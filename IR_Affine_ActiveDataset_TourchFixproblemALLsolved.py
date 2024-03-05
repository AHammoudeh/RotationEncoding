# A work-around to the bug in the official Torch library has been found, 
# The TORCH.NN.FUNCTIONAL.AFFINE_GRID does not transform the image correctly.
# This fix is here
# https://colab.research.google.com/drive/1JC8lJFFEQVLxWfcfUcSsiDU7DYkdAn-D


## conda activate dinov2
# cp -r '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/224/' '../../tmp/224/'
# cp -r '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/augmented/' '../../localdb/augmented/'
# cp -r '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/224/' '../../localdb/224/'

#Delete AE option if results are not beter than U-Net. The only difference between them is the shape of input: [3,2x128,128] Vs [2x3,128,128]

# imports
import os, sys
import numpy as np
from PIL import Image
import itertools
import glob
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.functional import relu as RLU
import torch.multiprocessing as mp
import time
import tqdm
import json
#-------------------------------------------------------------
device_num=0
dim = 128
dim0 =224
crop_ratio = dim/dim0

Fix_Torch_Wrap = False
BW_Position = False

if Fix_Torch_Wrap:
    Noise_level_dataset = 0.1
else:
    Noise_level_dataset = 0.5


DATASET_generation = 'active' #{'passive', #'active'}
registration_method = 'Rawblock' #{'Rawblock', 'matching_points', 'Additive_Recurence', 'Multiplicative_Recurence'} #'recurrent_matrix', 
imposed_point = 0

Arch = 'ResNet'#{'VTX', 'U-Net','ResNet', 'DINO' , }

if BW_Position:
    folder_suffix = '.BWPosition'
else:
    folder_suffix = ''

if 'Recurence' in registration_method :
    folder_suffix += '.completely_random'#'random_aff_param'#'completely_random' #'scheduledrandom' #zero

if registration_method == 'matching_points':
    with_epch_loss = True
    if imposed_point>0:
        folder_suffix += str(imposed_point)
else:
    with_epch_loss = False

if Fix_Torch_Wrap:
    folder_suffix += 'WrapFix'
else:
    folder_suffix += '_NoWrapFix'


with_shear = True
if with_shear:
    Intitial_Tx = ['angle', 'scale', 'translationX','translationY','shearX','shearY'] #,{'shearX','shearY'}
else:
    Intitial_Tx = ['angle', 'scale', 'translationX','translationY'] #,'shearX','shearY'] #,{'shearX','shearY'}

if Arch == 'VTX':
    batch_size = 16
elif Arch == 'ResNet':
    batch_size = 30
    if 'Recurence' in registration_method:
        batch_size = 80
elif Arch == 'U-Net':
    batch_size = 30
elif Arch == 'DINO':
    batch_size = 30

if registration_method=='matching_points':
    N_parameters = 18
else:
    N_parameters = 6

torch.cuda.set_device(device_num)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if DATASET_generation == 'active':
    root = '../../localdb/224/'
    #root = '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/224/'
    train_routes_source = glob.glob(root+'train/'+"**/*.JPEG", recursive = True)
    val_routes_source = glob.glob(root+'val/'+"**/*.JPEG", recursive = True)
    test_routes_source = glob.glob(root+'test/'+"**/*.JPEG", recursive = True)
else:
    #root = '../../localdb/augmented/'
    root = '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/augmented/'
    train_routes_source = glob.glob(root+'train/'+'source'+str(Noise_level_dataset)+"**/*.JPEG", recursive = True)
    val_routes_source = glob.glob(root+'val/'+'source'+str(Noise_level_dataset)+"**/*.JPEG", recursive = True)
    #test_routes_source = glob.glob(root+'test/'+'source'+str(Noise_level_dataset)+"**/*.JPEG", recursive = True)


file_savingfolder = '/home/ahmadh/MIR_savedmodel/{}_M0:{}_{}_{}_{}/'.format(
            registration_method,folder_suffix, Arch, DATASET_generation,Noise_level_dataset)
#file_savingfolder = file_savingfolder[:-1] +'shuffle_trainloader/' 
os.system('mkdir '+ file_savingfolder)
os.system('mkdir '+ file_savingfolder+ 'MIRexamples')

#train_routes = random.sample(train_routes, 10000)
#val_routes = random.sample(val_routes, 20000)
#test_routes = random.sample(test_routes, 20000)

#---------------Data--------------------------------
#---------------------------------------------------
#---------------------------------------------------
# Note when the input is in the form of dictionary, parallalism fails

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def wrap_imge_with_inverse(Affine_mtrx, source_img, ratio= 2):
    #edited version on 20240301
    Affine_mtrx = workaround_matrix(Affine_mtrx, acc = ratio).to('cpu')
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_img.shape,align_corners=False)
    wrapped_img = torch.nn.functional.grid_sample(source_img, grid=grd,
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)
    return wrapped_img

def wrap_imge0(Affine_mtrx, source_img):
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_img.shape,align_corners=False)
    wrapped_img = torch.nn.functional.grid_sample(source_img.to(device), grid=grd.to(device),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)
    return wrapped_img

def wrap_imge_cropped(Affine_mtrx, source_img, dim1=224, dim2=128):
  source_img224 = torch.nn.ZeroPad2d(int((dim1-dim2)/2))(source_img)
  grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_img224.shape,align_corners=False)
  wrapped_img = torch.nn.functional.grid_sample(source_img224, grid=grd,
                                                mode='bilinear', padding_mode='zeros', align_corners=False)
  wrapped_img = torchvision.transforms.CenterCrop((dim2, dim2))(wrapped_img)
  return wrapped_img


def workaround_matrix(Affine_mtrx0, acc = 2):
    # To find the equivalent torch-compatible matrix from a correct matrix set acc=2 #This will be needed for transforming an image
    # To find the correct Affine matrix from Torch compatible matrix set acc=0.5
    Affine_mtrx_adj = inv_AM(Affine_mtrx0)
    Affine_mtrx_adj[:,:,2]*=acc
    return Affine_mtrx_adj

def inv_AM(Affine_mtrx):
    AM3 = mtrx3(Affine_mtrx)
    AM_inv = torch.linalg.inv(AM3)
    return AM_inv[:,0:2,:]

def mtrx3(Affine_mtrx):
    mtrx_shape = Affine_mtrx.shape
    if len(mtrx_shape)==3:
        N_Mbatches = mtrx_shape[0]
        AM3 = torch.zeros( [N_Mbatches,3,3]).to(device)
        AM3[:,0:2,:] = Affine_mtrx
        AM3[:,2,2] = 1
    elif len(mtrx_shape)==2:
        N_Mbatches = 1
        AM3 = torch.zeros([3,3]).to(device)
        AM3[0:2,:] = Affine_mtrx
        AM3[2,2] = 1 
    return AM3

def move_dict2device(dictionary,device):
    for key in list(dictionary.keys()):
        dictionary[key] = dictionary[key].to(device)
    return dictionary

def standarize_point(d, dim=128, flip = False):
    if flip:
        d = -d
    return d/dim - 0.5

def destandarize_point(d, dim=128, flip = False):
    if flip:
        d = -d
    return dim*(d + 0.5)

def generate_standard_elips(N_samples = 100, a= 1,b = 1):
    radius = 0.25
    center = 0
    N_samples1 = int(N_samples/2 - 1)
    N_samples2 = N_samples - N_samples1
    x1 = torch.distributions.uniform.Uniform(center-radius,center + radius).sample([N_samples1])
    x1_ordered = torch.sort(x1).values
    y1 = center + b*torch.sqrt(radius**2 - ((x1_ordered-center)/a)**2)
    x2 = torch.distributions.uniform.Uniform(center-radius,center + radius).sample([N_samples2])
    x2_ordered = torch.sort(x2, descending=True).values
    y2 = center - b*torch.sqrt(radius**2 - ((x2_ordered-center)/a)**2)
    x = torch.cat([x1_ordered, x2_ordered])
    y = torch.cat([y1, y2])
    return x, y

def transform_standard_points(Affine_mat, x,y):
    XY = torch.ones([3,x.shape[0]])
    XY[0,:]= x
    XY[1,:]= y
    XYt = torch.matmul(Affine_mat.to('cpu').detach(),XY)
    xt0 = XYt[0]
    yt0 = XYt[1]
    return xt0, yt0


def save_examples(model, dataloader , n_examples = 4, plt_elipses=True,plt_imgs=False, time=1):
    with torch.no_grad():
        dataiter = iter(dataloader)
        X_batch, Y_batch = next(dataiter)
        X_batch = move_dict2device(X_batch,device)
        Y_batch = move_dict2device(Y_batch,device)
        pred = model(X_batch)
        Affine_mtrx = pred['Affine_mtrx']
        source = X_batch['source'].to(device)
        target = X_batch['target'].to(device)
        if Fix_Torch_Wrap:
            wrapped_img = wrap_imge0(Affine_mtrx, source)
            M_GroundTruth = workaround_matrix(Y_batch['Affine_mtrx'].detach(), acc = 0.5)
            M_Predicted = workaround_matrix(pred['Affine_mtrx'].detach(), acc = 0.5)
        else:
            wrapped_img = wrap_imge_cropped(Affine_mtrx, source)
            M_GroundTruth = workaround_matrix(Y_batch['Affine_mtrx'].detach(), acc = 0.5/crop_ratio)
            M_Predicted = workaround_matrix(pred['Affine_mtrx'].detach(), acc = 0.5/crop_ratio)
        #prepare figures
        figure = plt.figure()
        gs = figure.add_gridspec(n_examples, 3, hspace=0.05, wspace=0)
        axis = gs.subplots(sharex='col', sharey='row')
        figure.set_figheight(5*n_examples)
        figure.set_figwidth(15)
        for k in range(n_examples):
            if plt_elipses:
                x0_source, y0_source = generate_standard_elips(N_samples = 100)
                x0_target, y0_target = transform_standard_points(M_GroundTruth[k], x0_source, y0_source)
                x0_transformed, y0_transformed = transform_standard_points(M_Predicted[k], x0_source, y0_source)
                #x0_target, y0_target = transform_standard_points(Y_batch['Affine_mtrx'][k], x0_source, y0_source)
                #x0_transformed, y0_transformed = transform_standard_points(pred['Affine_mtrx'][k], x0_source, y0_source)
                #--------------destandarize for plotting purposes-------------
                x_source = destandarize_point(x0_source, dim=dim, flip = False)
                x_target = destandarize_point(x0_target, dim=dim, flip = False)
                x_transformed = destandarize_point(x0_transformed, dim=dim, flip = False)
                #flip the y-axis because when plotting the y axis is downward instead of upward
                y_source = destandarize_point(y0_source, dim=dim, flip = False)
                y_target = destandarize_point(y0_target, dim=dim, flip = False)
                y_transformed = destandarize_point(y0_transformed, dim=dim, flip = False)
                #plot
                axis[k, 0].plot(x_source,y_source, color ='black',marker='x', linewidth = 2)
                axis[k, 1].plot(x_target,y_target, color ='black',marker='x', linewidth = 2)
                axis[k, 2].plot(x_target,y_target, color ='black',marker='', linewidth = 1)
                axis[k, 2].plot(x_transformed,y_transformed, color ='red',marker='x', linewidth = 2)
            if plt_imgs:
                axis[k, 0].imshow(torchvision.transforms.ToPILImage()(source[k]))
                axis[k, 1].imshow(torchvision.transforms.ToPILImage()(target[k]))
                axis[k, 2].imshow(torchvision.transforms.ToPILImage()(wrapped_img[k]))
                
        for ax in figure.get_axes():
            ax.label_outer()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
        suffix = ''
        if plt_elipses:
                suffix += 'Elipse'
        if plt_imgs:
                suffix += 'Img'
        plt.savefig(file_savingfolder+'MIRexamples/{}_{}.jpeg'.format(suffix,time), bbox_inches='tight')
        plt.close()


def save_n_examples(model, loader,n_times=2, n_examples_per_time = 4 ):
    for m in range(n_times):
        save_examples(model, loader, n_examples = n_examples_per_time, plt_elipses=True, plt_imgs=False, time =m)
        save_examples(model, loader, n_examples = n_examples_per_time, plt_elipses=False, plt_imgs=True, time =m)
        save_examples(model, loader, n_examples = n_examples_per_time, plt_elipses=True, plt_imgs=True, time =m)

#save_n_examples(IR_Model_tst, testloader, n_times=2, n_examples_per_time = 4)

def pil_to_numpy(im):
    im.load()
    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)
    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))
    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data

def load_image_pil_accelerated(image_path):
    image = Image.open(image_path).convert("RGB")
    array = pil_to_numpy(image)
    tensor = torch.from_numpy(np.rollaxis(array,2,0)/255).to(torch.float32)
    #tensor = torchvision.transforms.Resize((dim,dim))(tensor)
    return tensor

#---------------------------------------------------
#---------------------------------------------------

def Load_augment(image_path, dim = 128): #measures=['angle', 'scale', 'translationX','translationY','shearX','shearY'], Noise_level=0,
  #enlargement = crop_ratio
  Outputs= {}
  #image0 = load_image_from_url(image_path, int(enlargement*dim))
  #image0 = (transforms.ToTensor()(image0)).unsqueeze(0).to(torch.float32)
  image0 = load_image_pil_accelerated(image_path).unsqueeze(0)
  transformed_img, Affine_mtrx, Affine_parameters = augment_img(image0)#, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx)
  image1 = torchvision.transforms.CenterCrop((dim, dim))(image0)
  transformed_img = torchvision.transforms.CenterCrop((dim, dim))(transformed_img)
  Outputs['Affine_mtrx'] = Affine_mtrx[0]
  Outputs['source'] = image1[0]
  Outputs['target'] = transformed_img[0]
  Outputs['Affine_parameters'] = Affine_parameters[0]
  return Outputs

def load_image_from_url(url, dim = 128):
    #if url.startswith('http'):
    #  url = urllib.request.urlopen(url)
    img = Image.open(url).convert("RGB")
    if dim!=224:
        img = img.resize((dim, dim))
    return img

#This function is created to pass Noise and the transformations which will be used to generate the affine matrix.
# THose parameters are not determined in the dataset class, so that they can be changed actively during training
#---------------------------------------------------
# Later another function(pass_parameters) should be defined to handle this instead, where when it is called it returns those changing parmeters: Noise, transformations, schedulers 
#---------------------------------------------------
#---------------------------------------------------
def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx): #,'shearX','shearY'
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
    return wrapped_img, Affine_mtrx, Affine_parameters

def pass_augment_img(image0, measure,NOISE_LEVEL=0, MODE='nearest'):
    try:
        dimw = image0.width
        dimh = image0.height
    except:
        dimw = image0.shape[2]
        dimh = image0.shape[3]
    #if no augmentation
    angle = 0
    scale_inv = (1 - 0.4)/1.3
    translationx = 0.5
    translationy = 0.5
    shearX=0
    shearY=0
    # if augmentation
    if 'angle' in measure: angle = np.random.uniform(0, 1) #1 is equivalent to 360 degrees or 2pi = 6.29
    if 'scale' in measure: scale_inv = np.random.uniform(0, 1)
    if 'translationX' in measure: translationx = np.random.uniform(0, 1)
    if 'translationY' in measure: translationy = np.random.uniform(0, 1)
    if 'shearX' in measure: shearX = np.random.uniform(0, 1)
    if 'shearY' in measure: shearY = np.random.uniform(0, 1)
    Affine_parameters = torch.tensor([[angle, scale_inv,translationx,translationy, shearX,shearY]]).to(torch.float32)
    Affine_mtrx = normalizedparameterline2Affine_matrx(Affine_parameters, device='cpu', Noise_level=NOISE_LEVEL)
    if Fix_Torch_Wrap:
        wrapped_img = wrap_imge_with_inverse(Affine_mtrx.detach(), image0, ratio= 2*crop_ratio)
        Affine_mtrx = workaround_matrix(Affine_mtrx.detach(), acc = 2).detach()
    else:
        wrapped_img = wrap_imge0(Affine_mtrx, image0)
    return wrapped_img, Affine_mtrx, Affine_parameters


#Later this function should be used in "pass_augment_img"
def Generate_random_AffineMatrix(measure=Intitial_Tx, NOISE_LEVEL=0, device='cpu'):
    angle = 0
    scale_inv = (1 - 0.4)/1.3
    translationx = 0.5
    translationy = 0.5
    shearX=0
    shearY=0
    # if augmentation
    if 'angle' in measure: angle = np.random.uniform(0, 1) #1 is equivalent to 360 degrees or 2pi = 6.29
    if 'scale' in measure: scale_inv = np.random.uniform(0, 1)
    if 'translationX' in measure: translationx = np.random.uniform(0, 1)
    if 'translationY' in measure: translationy = np.random.uniform(0, 1)
    if 'shearX' in measure: shearX = np.random.uniform(0, 1)
    if 'shearY' in measure: shearY = np.random.uniform(0, 1)
    Affine_parameters = torch.tensor([[angle, scale_inv,translationx,translationy, shearX,shearY]]).to(torch.float32)
    Affine_mtrx = normalizedparameterline2Affine_matrx(Affine_parameters, device=device, Noise_level=NOISE_LEVEL)
    return Affine_mtrx, Affine_parameters


def normalizedparameterline2Affine_matrx(line, device, Noise_level=0.0):
  N_batches = int(line.shape[0])
  Affine_mtrx = torch.ones(N_batches, 2,3).to(device)
  DeNormalize_parametersline = DeNormalize_AffineParameters(line)
  for i in range(N_batches):
    angle = DeNormalize_parametersline[i:i+1,0:1]
    scale_inv = DeNormalize_parametersline[i:i+1,1:2]
    translationx = DeNormalize_parametersline[i:i+1,2:3]
    translationy = DeNormalize_parametersline[i:i+1,3:4]
    shearx = DeNormalize_parametersline[i:i+1,4:5]
    sheary = DeNormalize_parametersline[i:i+1,5:6]
    Affine_Mtrx_i = Affine_parameters2matrx(angle,scale_inv,translationx,translationy, shearx, sheary)
    Affine_mtrx[i,:] = Affine_Mtrx_i#[:2,:]#.reshape(1,2,3)
  Affine_mtrx = Affine_mtrx+ Affine_mtrx*torch.normal(torch.zeros([N_batches,2,3]), Noise_level*torch.ones([N_batches,2,3]))
  return Affine_mtrx.to(torch.float32)

def Normalize_AffineParameters(parameters):
   Norm_parameters = parameters.clone()
   Norm_parameters[:,0]/= 6.29
   Norm_parameters[:,1] = (Norm_parameters[:,1] - 0.4)/1.3
   Norm_parameters[:,2] = (Norm_parameters[:,2] + 0.2)/0.4
   Norm_parameters[:,3] = (Norm_parameters[:,3] + 0.2)/0.4
   Norm_parameters[:,4] = (Norm_parameters[:,4] + 0.0)/0.1
   Norm_parameters[:,5] = (Norm_parameters[:,5] + 0.0)/0.1
   return Norm_parameters

def DeNormalize_AffineParameters(Normalized_Parameters):
   DeNormalized_Parameters = Normalized_Parameters.clone()
   DeNormalized_Parameters[:,0]*= 6.29
   DeNormalized_Parameters[:,1] = 1.3*DeNormalized_Parameters[:,1] + 0.4
   DeNormalized_Parameters[:,2] = 0.4*DeNormalized_Parameters[:,2] - 0.2
   DeNormalized_Parameters[:,3] = 0.4*DeNormalized_Parameters[:,3] - 0.2
   DeNormalized_Parameters[:,4] = 0.1*DeNormalized_Parameters[:,4] - 0.0
   DeNormalized_Parameters[:,5] = 0.1*DeNormalized_Parameters[:,5] - 0.0
   return DeNormalized_Parameters

def Affine_parameters2matrx(angle,scale_inv,translationx,translationy, shearx, sheary):
   N_batches = 1#int(line.shape[0])
   Affine_mtrx = torch.ones(N_batches, 2,3).to(device)
   for i in range(N_batches):
       Mat_shear = torch.tensor([[1.0, shearx, 0.0],[sheary,1.0, 0.0],[0.0, 0.0, 1.0]])
       Mat_translation = torch.tensor([[1.0, 0.0, translationx],[0.0, 1.0, translationy],[0.0, 0.0, 1.0]])
       Mat_scale = torch.tensor([[scale_inv, 0.0, 0.0],[0.0, scale_inv, 0.0],[0.0, 0.0, 1.0]])
       Mat_Rot = torch.tensor([[torch.cos(angle), torch.sin(angle), 0.0],
                               [-torch.sin(angle), torch.cos(angle), 0.0],
                               [0.0, 0.0, 1.0]]).float()
       Affine_Mtrx_i = torch.matmul(torch.matmul(torch.matmul(Mat_shear,Mat_scale), Mat_Rot),Mat_translation)
       Affine_mtrx[i,:] = Affine_Mtrx_i[:2,:].reshape(1,2,3)
   return Affine_mtrx

def Affine_mtrx2parameters(Affine_Mtrx):
   eps = 0.0000001
   alpha = X_delta(Affine_Mtrx[:,1,0]+Affine_Mtrx[:,0,1])
   N_batches = Affine_Mtrx.shape[0]
   parameters = torch.zeros(N_batches, 6).to(device)
   Translation_ = torch.matmul(torch.inverse(Affine_Mtrx[:,:2,:2]),Affine_Mtrx[:,:2,2:])
   tan_theta = (1-alpha)*(Affine_Mtrx[:,1,1]-Affine_Mtrx[:,0,0])/(eps +Affine_Mtrx[:,1,0]+Affine_Mtrx[:,0,1]) + alpha*(Affine_Mtrx[:,0,1]/Affine_Mtrx[:,0,0])
   Mx = (Affine_Mtrx[:,0,1]-Affine_Mtrx[:,0,0]*tan_theta)/(Affine_Mtrx[:,0,1]*tan_theta +Affine_Mtrx[:,0,0])
   My = (Affine_Mtrx[:,1,0]+Affine_Mtrx[:,1,1]*tan_theta)/(Affine_Mtrx[:,1,1]- Affine_Mtrx[:,1,0]*tan_theta)
   scale = torch.abs((Affine_Mtrx[:,0,0]+Affine_Mtrx[:,0,1]*tan_theta)/torch.sqrt(1+tan_theta**2))
   sin_theta = (Affine_Mtrx[:,0,1]-Affine_Mtrx[:,0,0]*Mx)/(scale*(1+Mx**2))#torch.tensor(1) #the sign of sin(a) to be derived
   parameters[:,1] = scale
   parameters[:,0] = torch.atan(tan_theta)+ torch.pi*RLU(-(torch.sign(tan_theta)))+ torch.pi*RLU(-(torch.sign(sin_theta)))
   parameters[:,2] = Translation_[:,0,0]
   parameters[:,3] = Translation_[:,1,0]
   parameters[:,4] = Mx
   parameters[:,5] = My
   Normalized_Parameters = Normalize_AffineParameters(parameters)
   return Normalized_Parameters

def X_delta(x):
 return torch.exp(torch.tensor(-100000*x**2))



def To_BWposition(img, dim):
    Pos_x = np.cos(np.pi*(np.arange(0,dim)/dim - 1/2))
    Pos_y = np.sin(np.pi*(np.arange(0,dim)/dim - 1/2))
    Grey_img = torchvision.transforms.Grayscale()(img)
    Position = torch.tensor(np.array([np.tile(Pos_x,[dim,1]).transpose(), np.tile(Pos_y,[dim,1])])).to(torch.float32)
    BWposition = torch.cat([Grey_img,Position], dim=0)
    return BWposition


def batch_ToBWposition(img_batch):
    dim = img_batch.shape[2]
    Pos_x = np.cos(np.pi*(np.arange(0,dim)/dim - 1/2))
    Pos_y = np.sin(np.pi*(np.arange(0,dim)/dim - 1/2))
    BWposition_batch = img_batch.detach()
    BWposition_batch[:,1:2,:,:] = torchvision.transforms.Grayscale()(img_batch)
    BWposition_batch[:,1,:,:] = torch.from_numpy(np.tile(Pos_x,[dim,1]).transpose())
    BWposition_batch[:,0,:,:] = torch.from_numpy(np.tile(Pos_y,[dim,1]))
    return BWposition_batch


class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_paths, dim = 128, batch_size = 25, DATASET='active', registration_method= 'Additive_Recurence', mode='train', folder_suffix = 'random'):
        self.list_paths = list_paths
        self.dim = dim
        self.dataset = DATASET
        self.batch_size = batch_size
        self.number_examples = len(self.list_paths)
        self.registration_method = registration_method
        self.mode = mode
        self.folder_suffix = folder_suffix
        #self.Noise_level = Noise_level
        #self.measures = measures
  def __len__(self):
        return int(self.batch_size*(self.number_examples // self.batch_size))
  def __getitem__(self, index):
        source_image_path = self.list_paths[index]
        if self.dataset == 'active':
            Outputs = Load_augment(source_image_path, dim = self.dim)
            source = Outputs['source'].to(torch.float32)
            target = Outputs['target'].to(torch.float32)
            Affine_mtrx = Outputs['Affine_mtrx']#.to(torch.float32)
            #Affine_parameters = Outputs['Affine_parameters']#.to(torch.float32)
        else:
            source_image_path = self.list_paths[index]
            source = load_image_pil_accelerated(source_image_path)
            target = load_image_pil_accelerated(source_image_path.replace('source','target'))
            with open(source_image_path.replace('source','matrix').replace('.JPEG','.npy'), 'rb') as f:
                Affine_mtrx = torch.from_numpy(np.load(f))
        #X_concatinated = torch.cat((source,target), dim=1).to(torch.float32)
        if 'Recurence' in self.registration_method:
            if 'completely_random' in self.folder_suffix :
                M_i = torch.normal(torch.zeros([2,3]), torch.ones([2,3]))
            elif 'random_aff_param' in self.folder_suffix:
                M_i =  Generate_random_AffineMatrix(measure=Intitial_Tx, NOISE_LEVEL=0.5, device='cpu')[0][0]
            elif 'random' in self.folder_suffix:
                if self.mode == 'train':
                    M_i = Affine_mtrx + Affine_mtrx*torch.normal(torch.zeros([2,3]), torch.rand([2,3]))
                else:
                    M_i = torch.normal(torch.zeros([2,3]), torch.rand([2,3]))
            else:
                if (self.mode == 'train') and (torch.rand([1])<0.5):
                    scheduled_noise = scheduled_parameters()
                    M_i = Affine_mtrx + Affine_mtrx*torch.normal(torch.zeros([2,3]), scheduled_noise*torch.ones([2,3]))
                else:
                    M_i = torch.zeros_like(Affine_mtrx)
            X = {'source':source,
                'target':target,
                'M_i' :M_i }
            if 'Additive' in self.registration_method:
                Y = {'Affine_mtrx': Affine_mtrx,
                   'Deviated_mtrx': Affine_mtrx - X['M_i'],}
            elif 'Multiplicative' in self.registration_method:
                Y = {'Affine_mtrx': Affine_mtrx,
                   'Deviated_mtrx': torch.matmul(mtrx3(Affine_mtrx), torch.linalg.inv(mtrx3(X['M_i'])))[0:2,:],}
        else:
            X = {'source':source,'target':target}
            Y = {'Affine_mtrx': Affine_mtrx,}
        return X,Y


train_set = Dataset(list_paths=train_routes_source,  batch_size=batch_size, DATASET=DATASET_generation,registration_method =registration_method,folder_suffix = folder_suffix, mode='train')
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,  shuffle=False) #num_workers=4,
#-------------
val_set = Dataset(list_paths=val_routes_source, batch_size=batch_size, DATASET=DATASET_generation,
                    registration_method =registration_method,folder_suffix = folder_suffix,  mode='valid')

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=True)
#-------------

X_item, Y_item = val_set.__getitem__(5)
dataiter = iter(valloader)
X_batch, Y_batch = next(dataiter)


#---------------Model--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#ResNet
#----------------------------------------------------------

class Build_IRmodel_Resnet(nn.Module):
    def __init__(self, resnet_model, registration_method = 'Rawblock', BW_Position=False):
        super(Build_IRmodel_Resnet, self).__init__()
        self.resnet_model = resnet_model
        self.BW_Position = BW_Position
        #self.N_parameters = N_parameters
        if registration_method=='matching_points':
            self.N_parameters = 18
        else:
            self.N_parameters = 6
        self.registration_method = registration_method
        self.fc1 =nn.Linear(6, 64)
        self.fc2 =nn.Linear(64, 128*3)
        #self.global_avg_pooling = nn.AdaptiveAvgPool2d(10)
        #self.fc1 =nn.Linear(10*10*2, 100)
        #self.fc2 =nn.Linear(100, 100)
        self.fc3 =nn.Linear(512, self.N_parameters)
    def forward(self, input_X_batch):
        source = input_X_batch['source']
        target = input_X_batch['target']
        if self.BW_Position:
            source = batch_ToBWposition(source)
            target = batch_ToBWposition(target)
        if 'Recurence' in self.registration_method:
            M_i = input_X_batch['M_i'].view(-1, 6)
            #-------------------------------------
            M_rep = F.relu(self.fc1(M_i))
            M_rep = F.relu(self.fc2(M_rep)).view(-1,3,1,128)
            #-------------------------------------
            concatenated_input = torch.cat((source,target,M_rep), dim=2)
        else:
            concatenated_input = torch.cat((source,target), dim=2)
        resnet_output = self.resnet_model(concatenated_input)
        #x = self.global_avg_pooling(Unet_output)
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        predicted_line = self.fc3(resnet_output)
        if self.registration_method=='matching_points':
            predcited_points = predicted_line.view(-1, 2, 3, 3)
            XY_source = predcited_points[:,1,:,:]
            if '1' in folder_suffix:
                XY_source[:,:,2] = 0.
            XY_source[:,2,:] = 1.
            XY_target = predcited_points[:,0,0:2,:]
            Prd_Affine_mtrx = torch.matmul(XY_target, torch.linalg.inv(XY_source)) 
            predction = {'Affine_mtrx': Prd_Affine_mtrx,
                        'XY_source':XY_source,
                       'XY_target':XY_target,}
        elif 'Recurence' in self.registration_method:
            predicted_part_mtrx = predicted_line.view(-1, 2, 3)
            if 'Additive' in self.registration_method:
                Prd_Affine_mtrx = predicted_part_mtrx + input_X_batch['M_i']
            elif 'Multiplicative' in self.registration_method:
                Prd_Affine_mtrx = torch.matmul(mtrx3(predicted_part_mtrx), mtrx3(input_X_batch['M_i']))[:,0:2,:]
            predction = {'predicted_part_mtrx':predicted_part_mtrx,
                            'Affine_mtrx': Prd_Affine_mtrx}
        else:
            Prd_Affine_mtrx = predicted_line.view(-1, 2, 3)
            predction = {'Affine_mtrx': Prd_Affine_mtrx}
        return predction


from torchvision.models import resnet18
if Arch == 'ResNet':
    core_model = resnet18(pretrained=True)#weights='ResNet18_Weights.IMAGENET1K_V1')
    core_model.fc = Identity()
    IR_Model = Build_IRmodel_Resnet(core_model, registration_method, BW_Position)
    core_model.to(device)
    IR_Model.to(device)


dataiter = iter(valloader)
X_batch, Y_batch = next(dataiter)

X_batch = move_dict2device(X_batch,device)
Y_batch = move_dict2device(Y_batch,device)

pred = IR_Model(X_batch)
Affine_mtrx = pred['Affine_mtrx']




#---------------Train--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------

def scheduled_parameters(epoch = 0, Total_epochs=15):
    sceduled_noise = epoch/Total_epochs
    return sceduled_noise


def eval_loss(loader, max_iterations=100, key = 'Affine_mtrx'):
    eval_loss_tot = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = IR_Model(inputs)
                eval_loss_tot += MSE_loss(labels[key], predections[key].detach())
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg

def threshold(numberX, max_threshold=2):
    if numberX>max_threshold:
        return max_threshold
    else:
        return numberX



Learning_rate = 0.001
print_every = 2000

MSE_loss = torch.nn.functional.mse_loss
optimizer = optim.AdamW(IR_Model.parameters(), lr=Learning_rate)#, momentum=0.9

training_loss_epochslist = []
validation_loss_epochslist = []
training_loss_iterationlist = []
validation_loss_iterationlist = []

TOTAL_Epochs = 12
best_loss = 100000000000000000000
for EPOCH in range(TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    if 'scheduledrandom' in folder_suffix:
        def scheduled_parameters(epoch = EPOCH, Total_epochs=TOTAL_Epochs):
            sceduled_noise = epoch/Total_epochs
            if epoch>12:
                sceduled_noise = 1
            return sceduled_noise
        print('epoch:', EPOCH,', scheduled noise:' , scheduled_parameters())
    for inputs, labels in tqdm.tqdm(trainloader):
        i+=1
        inputs = move_dict2device(inputs,device)
        labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IR_Model(inputs)
        loss_item = {}
        loss = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'])
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:    # print every 2000 mini-batches
                eval_loss_x = eval_loss(valloader,int(2000/batch_size),'Affine_mtrx').detach().item()
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}'
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                        torch.save(IR_Model.state_dict(), file_savingfolder+'IR_Model_bestVal.pth')
                        torch.save(core_model.state_dict(), file_savingfolder+'./core_model_bestVal.pth')
                running_loss = 0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    plt.legend()
    plt.savefig(file_savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(file_savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(file_savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")
    save_examples(IR_Model,valloader, n_examples = 6, plt_elipses=True,plt_imgs=True)
    if with_epch_loss:
        training_loss_epochslist.append(threshold(eval_loss(trainloader,int(10000/batch_size),'Affine_mtrx').detach().item()))
        validation_loss_epochslist.append(threshold(eval_loss(valloader,int(10000/batch_size),'Affine_mtrx').detach().item()))
        plt.plot(training_loss_epochslist, marker='o', label = 'epoch training loss')
        plt.plot(validation_loss_epochslist, marker='x', label = 'epoch validation loss')
        plt.legend()
        plt.savefig(file_savingfolder+'loss_epochs.png', bbox_inches='tight')
        plt.close()
        np.savetxt(file_savingfolder+'training_loss_epochslist.txt', training_loss_epochslist, delimiter=",", fmt="%.3f")
        np.savetxt(file_savingfolder+'validation_loss_epochslist.txt', validation_loss_epochslist, delimiter=",", fmt="%.3f")

print('Finished Training')
torch.save(IR_Model.state_dict(), file_savingfolder+'IR_Model_EndTraining.pth')
torch.save(core_model.state_dict(), file_savingfolder+'./core_model_EndTraining.pth')


if with_epch_loss:
    with open(file_savingfolder+'training_loss_epochslist.npy', 'wb') as f:
        np.save(f, np.array(training_loss_epochslist))
        
    with open(file_savingfolder+'validation_loss_epochslist.npy', 'wb') as f:
        np.save(f, np.array(validation_loss_epochslist))
        
    plt.plot(training_loss_epochslist, label = 'training loss')
    plt.plot(validation_loss_epochslist, label = 'validation loss')
    plt.legend()
    plt.savefig(file_savingfolder+'loss_epochs.png', bbox_inches='tight')
    plt.close()

with open(file_savingfolder+'training_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(training_loss_iterationlist))

with open(file_savingfolder+'validation_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(validation_loss_iterationlist))


plt.plot(training_loss_iterationlist, label = 'training loss')
plt.plot(validation_loss_iterationlist, label = 'validation loss')
plt.legend()
plt.savefig(file_savingfolder+'loss_iterations.png', bbox_inches='tight')
plt.close()

#---------------Test--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
    #Load best model saved on drive
ext = '_bestVal'#'_EndTraining' #_bestVal
if Arch == 'cnn':
    IR_Model = Build_IRmodel_CNN(registration_method)
elif Arch == 'ResNet':
    core_model_tst = resnet18(pretrained=True)
    core_model_tst.fc = Identity()
    core_model_tst.load_state_dict(torch.load(file_savingfolder+'core_model'+ext+'.pth'))
    core_model_tst.to(device)
    IR_Model_tst = Build_IRmodel_Resnet(core_model_tst, registration_method)
elif Arch == 'VTX':
    core_model_tst = vit_b_16(pretrained=True)
    core_model_tst.fc = Identity()
    core_model_tst.load_state_dict(torch.load(file_savingfolder+'core_model'+ext+'.pth'))
    core_model_tst.to(device)
    IR_Model_tst = Build_IRmodel_VTX(core_model_tst,registration_method)
elif Arch == 'U_net' or Arch == 'AE':
    core_model_tst = build_unet(dim = dim)
    core_model_tst.load_state_dict(torch.load(file_savingfolder+'core_model'+ext+'.pth'))
    core_model_tst.to(device)
    IR_Model_tst = Build_IRmodel_AE(core_model_tst, NN_Output, N_parameters)

IR_Model_tst.load_state_dict(torch.load(file_savingfolder+'IR_Model'+ext+'.pth'))
IR_Model_tst.to(device)
IR_Model_tst.eval()

#----------------------------
#Load test dataset

test_set = Dataset(list_paths=test_routes_source, batch_size=batch_size, DATASET=DATASET_generation,
            registration_method =registration_method,folder_suffix = folder_suffix, mode='test')
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True)

file_savingfolder_test = file_savingfolder+'test/'
os.system('mkdir '+ file_savingfolder_test)
os.system('mkdir '+ file_savingfolder_test+ 'MIRexamples')

file_savingfolder = file_savingfolder_test
#----------------------------

##------------------------------------------------------------------
## Plot deviation graph of each augmentation parameter from the the ground-truth
##------------------------------------------------------------------
# 1- fix three parameters and sample from one
# 2- plot the distrinution of the error for the variable parameter
# fixed values s =1, angle =0, translation =0
# MSE_AffineMatrix_1measure
#test_routes = glob.glob('../../localdb/224/test/'+"**/*.JPEG", recursive = True)
Measures_list = ['angle', 'scale', 'translationX','translationY','shearX','shearY']
num2txt= {  '0': 'angle', '1': 'scale', '2': 'translationX','3': 'translationY', '4': 'shearX','5': 'shearY'}
txt2num ={'angle':0, 'scale':1, 'translationX':2, 'translationY':3, 'shearX':4,'shearY':5}


def test_loss(model, loader=testloader, max_iterations=100, key = 'Affine_mtrx'):
    eval_loss_tot = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = model(inputs)
                eval_loss_tot += MSE_loss(labels[key], predections[key].detach())
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg



MSE_affine = test_loss(IR_Model_tst, testloader,200, key = 'Affine_mtrx').detach().item()
print(MSE_affine)
json.dump( MSE_affine, open( file_savingfolder+'Test_loss.txt', 'w' ) )
#np.savetxt(file_savingfolder+'Test_loss.txt', np.array([MSE_affine]), delimiter=",", fmt="%.5f")


save_examples(IR_Model_tst,testloader, n_examples = 6)
save_n_examples(IR_Model_tst, testloader, n_times=2, n_examples_per_time = 4 )



##------------------------------------------------------------------
## Recurrent estimation of the Affine matrix
##------------------------------------------------------------------
def recurrent_loss(model,loader = testloader, max_iterations=100, No_recurences = 5, key ='Affine_mtrx'):
    eval_loss_tot = {}
    eval_loss_avg = {}
    for j in range(No_recurences):
        eval_loss_tot[str(j)]=0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = model(inputs)
                eval_loss_tot[str(0)] += MSE_loss(labels[key].detach(), predections[key].detach()).item()
                for j in range(1, No_recurences):
                    inputs_j ={'source': inputs['source'],
                                'target': inputs['target'],
                                'M_i': predections['Affine_mtrx'].detach()}
                    predections = model(inputs_j)
                    eval_loss_tot[str(j)] += MSE_loss(labels[key].detach(), predections[key].detach()).item()
            else :
                for j in range(No_recurences):
                    eval_loss_avg[str(j)] = eval_loss_tot[str(j)]/max_iterations
                return eval_loss_avg

No_recurences = 5
if 'Recurence' in registration_method:
    Noise_level_testset=0.5
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx): #['angle', 'scale', 'translationX','translationY','shearX','shearY']
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
            return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_recurrent_sigmahalf = recurrent_loss(model=IR_Model_tst,loader=testloader, max_iterations=200, No_recurences = No_recurences)
    print('REcurrent MSE when Noise_level_testset=0.5:', MSE_recurrent_sigmahalf)
    json.dump( MSE_recurrent_sigmahalf, open( file_savingfolder+'MSE_recurrent_sigmahalf.txt', 'w' ) )
    #np.savetxt(file_savingfolder+'Test_loss_recurrences_Sigmahalf.txt', np.array([MSE_recurrent_sigmahalf]), delimiter=",", fmt="%.5f")


if 'Recurence' in registration_method:
    Noise_level_testset=0.0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx): #['angle', 'scale', 'translationX','translationY','shearX','shearY']
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
            return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_recurrent_sigma0 = recurrent_loss(model=IR_Model_tst,loader=testloader, max_iterations=200, No_recurences = No_recurences)
    print('MSE when Noise_level_testset=0.0:', MSE_recurrent_sigma0)
    json.dump( MSE_recurrent_sigma0, open( file_savingfolder+'MSE_recurrent_sigma0.txt', 'w' ) )
    #np.savetxt(file_savingfolder+'Test_loss_recurrences_Sigma0.txt', np.array([MSE_recurrent_sigma0]), delimiter=",", fmt="%.5f")


#Test Loss when sigma =0 and 1 transformation
#------------------------------------------------

Noise_level_testset=0.5
def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx): #['angle', 'scale', 'translationX','translationY','shearX','shearY']
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
        return wrapped_img, Affine_mtrx, Affine_parameters

Test_loss_sigma0 = test_loss(IR_Model_tst,testloader,200, key = 'Affine_mtrx').detach().item()
print(Test_loss_sigma0)
json.dump( Test_loss_sigma0, open( file_savingfolder+'Test_loss_sigma{}.txt'.format(Noise_level_testset), 'w' ) )


MSE_AffineMatrix_1measure = {}

for MEASURE in Measures_list:
    MEASURES= [MEASURE]#['angle', 'scale', 'translationX','translationY','shearX','shearY'] #['angle']#
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
        return wrapped_img, Affine_mtrx, Affine_parameters
    #test_set = Dataset(list_paths=test_routes_source, batch_size=batch_size, DATASET=DATASET_generation,  mode='test')
    #testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)
    MSE_AffineMatrix_1measure[MEASURE] = test_loss(IR_Model_tst,testloader,200, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_1measure)
json.dump( MSE_AffineMatrix_1measure, open( file_savingfolder+'Test_loss_1Transformation.txt', 'w' ) )


'''
max_iterations=200
eval_loss_tot = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        if i < max_iterations:
            inputs, labels = data
            inputs = move_dict2device(inputs,device)
            labels = move_dict2device(labels,device)
            #predections = IR_Model_tst(inputs)
            eval_loss_tot += MSE_loss(labels['Affine_mtrx'], inputs['M_i']).detach()
            eval_loss_avg = eval_loss_tot/i
        else:
            break
'''


#------------------------------------------------
# Plotting deviation of the affine parameters [not accurate]

Noise_level_testset=0.0
def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx): #['angle', 'scale', 'translationX','translationY','shearX','shearY']
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
        return wrapped_img, Affine_mtrx, Affine_parameters

N_test_examples =300
measurements_compare = {}
for ls in Measures_list:
    measurements_compare[ls] = torch.zeros(2,N_test_examples)

for MEASURE in Measures_list:
    MEASURES= [MEASURE]#['angle', 'scale', 'translationX','translationY','shearX','shearY'] #['angle']#
    #test_set0 = Dataset(list_paths=test_routes_source, batch_size=batch_size, DATASET=DATASET_generation,
    #        registration_method =registration_method,folder_suffix = folder_suffix, mode='test')
    #tesloader0 = torch.utils.data.DataLoader(test_set0, batch_size=batch_size,shuffle=False)
    dataiter = iter(testloader)
    with torch.no_grad():
        for k in range(N_test_examples//batch_size):
            inputs, labels  = next(dataiter)
            inputs = move_dict2device(inputs,device)
            labels = move_dict2device(labels,device)
            predections = IR_Model_tst(inputs)
            measurements_compare[MEASURE][0,k*batch_size:(k+1)*batch_size] =  Affine_mtrx2parameters(labels['Affine_mtrx'])[:, txt2num[MEASURE]]
            measurements_compare[MEASURE][1,k*batch_size:(k+1)*batch_size]  = Affine_mtrx2parameters(predections['Affine_mtrx'])[:, txt2num[MEASURE]]

def clip(x, x_min=torch.tensor(-1), x_max=torch.tensor(2)):
    return torch.max(torch.min(x, x_max), x_min)

for ls in Measures_list:
    plt.plot([0,1], [0,1])
    plt.scatter(measurements_compare[ls][0,:], 
                clip(measurements_compare[ls][1,:],torch.tensor(-1),torch.tensor(2)), alpha = 0.2, cmap='viridis')
    plt.title(ls)
    plt.xlabel("ground-truth")
    plt.ylabel("prediction")
    #plt.legend([str(d) for d in range(N_recurences+1)],loc='upper left')
    plt.savefig(file_savingfolder+ls+'.png')
    plt.close()

#------------------------------------------------
# MSE_Affine Parameters 1measure

def eval_AP_loss(loader, max_iterations=100):
    eval_loss_tot = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = IR_Model_tst(inputs)
                eval_loss_tot += MSE_loss(Affine_mtrx2parameters(labels['Affine_mtrx'])[:, txt2num[MEASURE]], Affine_mtrx2parameters(predections['Affine_mtrx'].detach())[:, txt2num[MEASURE]] )
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg

Affine_parameters_1measure = {}

for MEASURE in Measures_list:
    # Affine_parameters
    MEASURES= [MEASURE]#['angle', 'scale', 'translationX','translationY','shearX','shearY'] #['angle']#
    def augment_img(image0, NOISE_LEVEL=0, MODE='bilinear', ONE_MESURE=MEASURES): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
        return wrapped_img, Affine_mtrx, Affine_parameters
    Affine_parameters_1measure[MEASURE] = eval_AP_loss(testloader,200).detach().item()

print(Affine_parameters_1measure)
json.dump( Affine_parameters_1measure, open( file_savingfolder+'Test_loss_Affine_parameters_1measure.txt', 'w' ) )

{'angle': 0.103, 'scale': 0.246, 'translationX': 0.039, 'translationY': 0.036, 'shearX': 245797, 'shearY': 160380}

##------------------------------------------------------------------


'''
key = 'Affine_mtrx'
max_iterations=10
No_recurences = 10
eval_loss_tot = {}
eval_loss_avg = {}
for j in range(No_recurences):
    eval_loss_tot[str(j)]=0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = IR_Model_tst(inputs)
                eval_loss_tot[str(0)] += MSE_loss(labels[key], predections[key])
                for j in range(1, No_recurences):
                    inputs_j ={'source': inputs['source'],
                                'target': inputs['target'],
                                'M_i': predections['Affine_mtrx'].detach()}
                    predections = IR_Model_tst(inputs_j)
                    eval_loss_tot[str(j)] += MSE_loss(labels[key].detach(), predections[key].detach()).item()
            else :
                for j in range(No_recurences):
                    eval_loss_avg[str(j)] = eval_loss_tot[str(j)]/max_iterations
'''



##------------------------------------------------------------------
## Old Recurrent registration
##------------------------------------------------------------------


N_recurences = 3

Measures_list =['MSE[I]','NCC[I]']
avg_measure = {}
total_measurex = {}

for recurence in range(N_recurences):
    avg_measure[str(recurence)] = {}
    total_measurex[str(recurence)] = {}
    for ls in Measures_list:
        total_measurex[str(recurence)][ls] = 0.0
        avg_measure[str(recurence)][ls] = 0.0

pred_evolution ={}
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        if i>1:
            inputs, labels = data            
            for recurence in range(N_recurences):
                pred = IR_Model_tst(inputs)
                pred_evolution[str(recurence)]= pred
                inputs['source']= 0
                inputs['source'] = pred['wrapped'].detach()
                measure_val = {}
                measure_val['MSE[I]'] = MSE_loss(labels['target'], pred['wrapped'])
                measure_val['NCC[I]'] = NCC_loss(labels['target'], pred['wrapped'])
                for ls in Measures_list:
                    try: total_measurex[str(recurence)][ls] += measure_val[ls].item()
                    except: total_measurex[str(recurence)][ls] += measure_val[ls]
            if i%10 ==0:
                for ls in Measures_list:
                    txt=''
                    for recurence in range(N_recurences):
                        txt += '{}v{}: {} --  '.format(ls,recurence,np.round(total_measurex[str(recurence)][ls]/i,3))
                    print(txt)
                    
    for recurence in range(N_recurences):
        for ls in Measures_list:
            avg_measure[str(recurence)][ls] = total_measurex[ls]/i



'''
dataiter = iter(valloader)
X_batch, Y_batch = next(dataiter)

def save_examples(IR_Model, dataloader , n_examples = 10):
    with torch.no_grad():
        dataiter = iter(dataloader)
        X_batch, Y_batch = next(dataiter)
        X_batch = move_dict2device(X_batch,device)
        #Y_batch = move_dict2device(Y_batch,device)
        pred = IR_Model(X_batch)
        Affine_mtrx = pred['Affine_mtrx']
        source = X_batch['source'].to(device)
        target = X_batch['target'].to(device)
        wrapped_img = wrap_imge(Affine_mtrx, source)
        for k in range(n_examples):
            if registration_method=='matching_points':
                P1x = [dim*XY_source[k,0,0], dim*XY_target[k,0,0]]
                P1y = [dim*XY_source[k,1,0], dim*(1+XY_target[k,1,0])]
                P2x = [dim*XY_source[k,0,1], dim*XY_target[k,0,1]]
                P2y = [dim*XY_source[k,1,1], dim*(1+XY_target[k,1,1])]
                P3x = [dim*XY_source[k,0,2], dim*XY_target[k,0,2]]
                P3y = [dim*XY_source[k,1,2], dim*(1+XY_target[k,1,2])]
                plt.plot(P1x,P1y,P2x,P2y,P3x,P3y, color ='white',marker='x', linewidth = 2)
                plt.plot(P1x,P1y, color ='white',marker='x', linewidth = 2)
                plt.plot(P2x,P2y, color ='white',marker='o', linewidth = 2)
                plt.plot(P3x,P3y, color ='white',marker='s', linewidth = 2)
                plt.imshow(concatenated_PILimages)
                plt.savefig(file_savingfolder+'tst.png', bbox_inches='tight')
                plt.close()
            else:
                concatenated_images = torch.cat((source[k],target[k],wrapped_img[k]), dim =2 )
                concatenated_PILimages = torchvision.transforms.ToPILImage()(concatenated_images)
                concatenated_PILimages.save(file_savingfolder+'MIRexamples/{}.jpeg'.format(k))

dataiter = iter(dataloader)
X_batch, Y_batch = next(dataiter)


X_batch = move_dict2device(X_batch,device)
#Y_batch = move_dict2device(Y_batch,device)
pred = IR_Model(X_batch)
Affine_mtrx = pred['Affine_mtrx']
XY_target = pred['XY_target']
XY_source = pred['XY_source']

source = X_batch['source'].to(device)
target = X_batch['target'].to(device)


y1 = [XY_source[k,0,0], XY_source[k,0,1]]


k=0
concatenated_images = torch.cat((source[k],target[k],wrapped_img[k]), dim =2 )
concatenated_PILimages = torchvision.transforms.ToPILImage()(concatenated_images)
concatenated_images = X_batch


P1x = [dim*XY_source[k,0,0], dim*XY_target[k,0,0]]
P1y = [dim*XY_source[k,1,0], dim*(1+XY_target[k,1,0])]
P2x = [dim*XY_source[k,0,1], dim*XY_target[k,0,1]]
P2y = [dim*XY_source[k,1,1], dim*(1+XY_target[k,1,1])]
P3x = [dim*XY_source[k,0,2], dim*XY_target[k,0,2]]
P3y = [dim*XY_source[k,1,2], dim*(1+XY_target[k,1,2])]
plt.plot(P1x,P1y,P2x,P2y,P3x,P3y, color ='white',marker='x', linewidth = 2)
plt.plot(P1x,P1y, color ='white',marker='x', linewidth = 2)
plt.plot(P2x,P2y, color ='white',marker='o', linewidth = 2)
plt.plot(P3x,P3y, color ='white',marker='s', linewidth = 2)
plt.imshow(concatenated_PILimages)
plt.savefig(file_savingfolder+'tst.png', bbox_inches='tight')
plt.close()

wrapped_img = wrap_imge(Affine_mtrx, source)
for k in range(n_examples):
    concatenated_images = torch.cat((source[k],target[k],wrapped_img[k]), dim =2 )
    concatenated_PILimages = torchvision.transforms.ToPILImage()(concatenated_images)
    concatenated_PILimages.save(file_savingfolder+'MIRexamples/{}.jpeg'.format(k))
'''

'''
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,num_workers=4,shuffle=False)
dataiter = iter(valloader)
T={}
for n in range(10):
    T[str(n)] = 0

max_i = 100
ind = 0
T_s = 0
start0 = time.time()
end0 = time.time()
#for inputs, labels in valloader:
for ind in range(max_i):
    #ind +=1
    #if ind> max_i:
    #    break
    #--------------------------
    #time.sleep(0.2)
    start1 = time.time()
    inputs, labels = next(dataiter)
    time.sleep(0.1)
    T['1'] += ((time.time()-start1)*10**3)/max_i
    #--------------------------
    start7 = time.time()
    labels = labels.to(device)
    T['7'] += ((time.time()-start7)*10**3)/max_i
    start8 = time.time()
    inputs = inputs.to(device)
    T['8'] += ((time.time()-start8)*10**3)/max_i
    #--------------------------
    start2 = time.time()
    optimizer.zero_grad()
    T['2'] += ((time.time()-start2)*10**3)/max_i
    #--------------------------
    #time.sleep(0.2)
    start3 = time.time()        
    predections = IR_Model(inputs)
    #time.sleep(0.05)
    T['3'] += ((time.time()-start3)*10**3)/max_i
    #--------------------------
    start4 = time.time()
    loss = MSE_loss(labels, predections)
    T['4'] += ((time.time()-start4)*10**3)/max_i
    #--------------------------
    #time.sleep(0.2)
    start5 = time.time()
    loss.backward()
    #time.sleep(0.2)
    T['5'] += ((time.time()-start5)*10**3)/max_i
    #--------------------------
    start6 = time.time()
    optimizer.step()
    T['6'] += ((time.time()-start6)*10**3)/max_i
    T['9'] += ((time.time()-end0)*10**3)/max_i
    end0 = time.time()

T['0'] += ((time.time()-start0)*10**3)/max_i
T['0']
'''



# Old models
# -----------------------------------------



#DINO
#----------------------------------------------------------

if registration_method=='recurrent_matrix':
    matrix_size = 1
else:
    matrix_size = 0

DINO_transform = transforms.Compose([transforms.Resize(size=(14*16 - matrix_size,14*16 ), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

class Build_IRmodel_DINO(nn.Module):
    def __init__(self, DINO_model,registration_method = 'affine matrix'):
        super(Build_IRmodel_DINO, self).__init__()
        self.DINO_model = DINO_model
        self.registration_method = registration_method
        if registration_method=='matching_points':
            self.N_parameters = 18
        else:
            self.N_parameters = 6
        #self.N_parameters = N_parameters
        #self.device = device
        self.conv3 = nn.Conv2d(3, 3, kernel_size=14, padding='same', stride = 1)
        self.activation = nn.ELU()
        #self.global_avg_pooling = nn.AdaptiveAvgPool2d(10)
        self.fc1 =nn.Linear(6, 64)
        self.fc2 =nn.Linear(64, 128*3)
        self.fc3 =nn.Linear(384, 384)
        self.fc4 =nn.Linear(384, self.N_parameters)
    def forward(self, concatenated_input):
        source = input_X_batch['source']
        target = input_X_batch['target']
        concatenated_images = torch.cat((source,target), dim=2)
        concatenated_images = DINO_transform(concatenated_images)
        if self.registration_method=='recurrent_matrix':
            M_i = input_X_batch['M_i'].view(-1, 6)
            #-------------------------------------
            M_rep = F.relu(self.fc1(M_i))
            M_rep = F.relu(self.fc2(M_rep)).view(-1,3,1,128)
            #-------------------------------------
            concatenated_input = torch.cat((concatenated_images,M_rep), dim=2)
        else:
            concatenated_input = torch.cat((source,target), dim=2)
        #X = torch.zeros([batch_size, 3, 288, 200])
        #X[:, :, 16: 272, 36:164] = concatenated_input
        Tx_inputs = DINO_transform(concatenated_input)
        DINO_inputs = self.conv3(Tx_inputs)
        VTX_output = self.DINO_model(DINO_inputs)
        #x = self.global_avg_pooling(Unet_output)
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = VTX_output
        x = F.relu(self.fc3(x))
        predicted_line = self.fc4(x)
        if self.registration_method=='matching_points':
            predcited_points = predicted_line.view(-1, 2, 3, 3)
            XY_source = predcited_points[:,1,:,:]
            XY_source[:,2,:] = 1.
            XY_target = predcited_points[:,0,0:2,:]
            Affine_mtrx = torch.matmul(XY_target, torch.linalg.inv(XY_source)) 
            predction = {'Affine_mtrx': Affine_mtrx,
                        'XY_source':XY_source,
                       'XY_target':XY_target, }
        elif self.registration_method=='recurrent_matrix':
            predicted_mtrx = predicted_line.view(-1, 2, 3)
            Affine_mtrx = predicted_mtrx + input_X_batch['M_i']
            predction = {'predicted_mtrx':predicted_mtrx,
                            'Affine_mtrx': Affine_mtrx}
        else:
            Affine_mtrx = predicted_line.view(-1, 2, 3)
            predction = {'Affine_mtrx': Affine_mtrx}#{'wrapped': wrapped_img,'Affine_mtrx': Affine_mtrx}
        return predction


if Arch == 'DINO':
    core_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    count =0
    for param in core_model.parameters():
        count+=1
        if count>1 and count<173:
            param.requires_grad = False
            
    IR_Model = Build_IRmodel_DINO(core_model,registration_method )
    core_model.to(device)
    IR_Model.to(device)



#ViT
#----------------------------------------------------------
if registration_method=='recurrent_matrix':
    matrix_size = 1
else:
    matrix_size = 0

VTX_scale = torchvision.transforms.Resize([224-matrix_size,224], antialias=False)
class Build_IRmodel_VTX(nn.Module):
    def __init__(self, VTX_model,registration_method = 'affine matrix'):
        super(Build_IRmodel_VTX, self).__init__()
        self.VTX_model = VTX_model
        self.registration_method = registration_method
        if registration_method=='matching_points':
            self.N_parameters = 18
        else:
            self.N_parameters = 6
        #self.N_parameters = N_parameters
        #self.device = device
        #self.global_avg_pooling = nn.AdaptiveAvgPool2d(10)
        self.fc1 =nn.Linear(6, 64)
        self.fc2 =nn.Linear(64, 128*3)
        self.fc3 =nn.Linear(768, 768)
        self.fc4 =nn.Linear(768, self.N_parameters)
    def forward(self, concatenated_input):
        source = input_X_batch['source']
        target = input_X_batch['target']
        concatenated_images = torch.cat((source,target), dim=2)
        concatenated_images = DINO_transform(concatenated_images)
        if self.registration_method=='recurrent_matrix':
            M_i = input_X_batch['M_i'].view(-1, 6)
            #-------------------------------------
            M_rep = F.relu(self.fc1(M_i))
            M_rep = F.relu(self.fc2(M_rep)).view(-1,3,1,128)
            #-------------------------------------
            concatenated_input = torch.cat((concatenated_images,M_rep), dim=2)
        else:
            concatenated_input = torch.cat((source,target), dim=2)
        #X = torch.zeros([batch_size, 3, 288, 200])
        #X[:, :, 16: 272, 36:164] = concatenated_input
        VTX_inputs = VTX_scale(concatenated_input)
        VTX_output = self.VTX_model(VTX_inputs)
        #x = self.global_avg_pooling(Unet_output)
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = VTX_output
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        predicted_line = self.fc4(x)
        if self.registration_method=='matching_points':
            predcited_points = predicted_line.view(-1, 2, 3, 3)
            XY_source = predcited_points[:,1,:,:]
            XY_source[:,2,:] = 1.
            XY_target = predcited_points[:,0,0:2,:]
            Affine_mtrx = torch.matmul(XY_target, torch.linalg.inv(XY_source)) 
            predction = {'Affine_mtrx': Affine_mtrx,
                        'XY_source':XY_source,
                       'XY_target':XY_target, }
        elif self.registration_method=='recurrent_matrix':
            predicted_mtrx = predicted_line.view(-1, 2, 3)
            Affine_mtrx = predicted_mtrx + input_X_batch['M_i']
            predction = {'predicted_mtrx':predicted_mtrx,
                            'Affine_mtrx': Affine_mtrx}
        else:
            Affine_mtrx = predicted_line.view(-1, 2, 3)
            predction = {'Affine_mtrx': Affine_mtrx}#{'wrapped': wrapped_img,'Affine_mtrx': Affine_mtrx}
        return predction


from torchvision.models import vit_b_16
if Arch == 'VTX':
    core_model = vit_b_16(pretrained=True)
    count =0
    for param in core_model.parameters():
        count+=1
        if count>1:
            param.requires_grad = False
    core_model.heads.head = Identity()
    IR_Model = Build_IRmodel_VTX(core_model,registration_method )
    core_model.to(device)
    IR_Model.to(device)



## U-Net
#----------------------------------------------------------

class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_c)
    self.relu = nn.ReLU()
  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    return x

class encoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv = conv_block(in_c, out_c)
    self.pool = nn.MaxPool2d((2, 2))
  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)
    return x, p

class decoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    self.conv = conv_block(out_c+out_c, out_c)
  def forward(self, inputs, skip):
    x = self.up(inputs)
    x = torch.cat([x, skip], axis=1)
    x = self.conv(x)
    return x

class build_unet(nn.Module):
  def __init__(self, dim=128):
    super().__init__()
    self.dim = dim
    #self.layer_norm = torch.nn.LayerNorm([6, dim, dim])
    self.e1 = encoder_block(3, 32)
    self.e2 = encoder_block(32, 64)
    self.e3 = encoder_block(64, dim)
    self.b = conv_block(dim, dim)
    self.d1 = decoder_block(dim, dim)
    self.d2 = decoder_block(dim, 64)
    self.d3 = decoder_block(64, 32)
    self.outputs = nn.Conv2d(32, 2, kernel_size=5,stride = 2, padding=0)
  def forward(self, inputs):
    #inputs = self.layer_norm(inputs)
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    b = self.b(p3)
    d1 = self.d1(b, s3)
    d2 = self.d2(d1, s2)
    d3 = self.d3(d2, s1)
    outputs = self.outputs(d3)
    return outputs

class Build_IRmodel_AE(nn.Module):
    def __init__(self, unet,registration_method = 'affine matrix'):
        super(Build_IRmodel_AE, self).__init__()
        self.unet = unet
        if registration_method=='matching_points':
            self.N_parameters = 18
        else:
            self.N_parameters = 6
        #self.N_parameters = N_parameters
        #self.device = device
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(10)
        self.fc1 =nn.Linear(6, 64)
        self.fc2 =nn.Linear(64, 128*3)
        self.fc3 =nn.Linear(2*10*10, 100)
        self.fc4 =nn.Linear(100, 100)
        self.fc5 =nn.Linear(100, self.N_parameters)
    def forward(self, concatenated_input):
        source = input_X_batch['source']
        target = input_X_batch['target']
        if self.registration_method=='recurrent_matrix':
            M_i = input_X_batch['M_i'].view(-1, 6)
            #-------------------------------------
            M_rep = F.relu(self.fc1(M_i))
            M_rep = F.relu(self.fc2(M_rep)).view(-1,3,1,128)
            #-------------------------------------
            concatenated_input = torch.cat((source,target,M_rep), dim=2)
        else:
            concatenated_input = torch.cat((source,target), dim=1)
        Unet_output = self.unet(concatenated_input)
        x = self.global_avg_pooling(Unet_output)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        predicted_line = self.fc5(x)
        if self.registration_method=='matching_points':
            predcited_points = predicted_line.view(-1, 2, 3, 3)
            XY_source = predcited_points[:,1,:,:]
            XY_source[:,2,:] = 1.
            XY_target = predcited_points[:,0,0:2,:]
            Affine_mtrx = torch.matmul(XY_target, torch.linalg.inv(XY_source)) 
            predction = {'Affine_mtrx': Affine_mtrx,
                        'XY_source':XY_source,
                        'XY_target':XY_target, }
        elif self.registration_method=='recurrent_matrix':
            predicted_mtrx = predicted_line.view(-1, 2, 3)
            Affine_mtrx = predicted_mtrx + input_X_batch['M_i']
            predction = {'predicted_mtrx':predicted_mtrx,
                            'Affine_mtrx': Affine_mtrx}
        else:
            Affine_mtrx = predicted_line.view(-1, 2, 3)
            predction = {'Affine_mtrx': Affine_mtrx}
        return predction

if Arch == 'U-Net':
    core_model = build_unet(dim = dim)
    IR_Model = Build_IRmodel_AE(core_model, registration_method)
    core_model.to(device)
    IR_Model.to(device)


## CNN
#----------------------------------------------------------

class Build_IRmodel_CNN(nn.Module):
    def __init__(self, registration_method = 'affine matrix'):
      super(Build_IRmodel_CNN, self).__init__()
      if registration_method=='matching_points':
            self.N_parameters = 18
        else:
            self.N_parameters = 6
      self.registration_method = registration_method   
      self.conv1 = nn.Conv2d(6, 128, kernel_size=7,stride = 4, padding=3)
      self.conv2 = nn.Conv2d(128, 256, kernel_size=5,stride = 3, padding=3)
      self.conv3 = nn.Conv2d(256, 512, kernel_size=5,stride = 3, padding=1)
      self.conv4 = nn.Conv2d(512, 512, kernel_size=3,stride = 2, padding=1)
      self.bn1 = nn.BatchNorm2d(128)
      self.bn2 = nn.BatchNorm2d(256)
      self.bn3 = nn.BatchNorm2d(512)
      self.relu = nn.ReLU()
      self.activation = nn.ELU()
      self.mxpool = nn.MaxPool2d(3)
      self.global_avg_pooling = nn.AdaptiveAvgPool2d(10)
      self.fc1 =nn.Linear(6, 64)
      self.fc2 =nn.Linear(64, 128*3)
      self.fc3 =nn.Linear(512, self.N_parameters)
      self.fc4 =nn.Linear(512, 100)
      self.fc5 =nn.Linear(100, 100)
      self.fc6 =nn.Linear(100, self.N_parameters)
    def forward(self, inputs):
      source = input_X_batch['source']
      target = input_X_batch['target']
      if self.registration_method=='recurrent_matrix':
        M_i = input_X_batch['M_i'].view(-1, 6)
        #-------------------------------------
        M_rep = F.relu(self.fc1(M_i))
        M_rep = F.relu(self.fc2(M_rep)).view(-1,3,1,128)
        #-------------------------------------
        concatenated_input = torch.cat((source,target,M_rep), dim=2)
      else:
        concatenated_input = torch.cat((source,target), dim=2)
      cn1 = self.activation(self.bn1(self.conv1(concatenated_input)))
      cn2 = self.activation(self.bn2(self.conv2(cn1)))
      cn3 = self.activation(self.conv3(cn2))
      cn3 = self.mxpool(cn3)
      cn4 = self.activation(self.bn3(self.conv4(cn3)))
      cn4 = self.activation(self.conv4(cn3))
      cn_output = cn4
      #predction = cn4
      x = torch.flatten(cn_output, 1)
      x = self.activation(self.fc4(x))
      x = self.activation(self.fc5(x))
      predicted_line = self.fc6(x)
      if self.registration_method=='Rawblock':
       Affine_mtrx = predicted_line.view(-1, 2, 3)
       Affine_parameters = Affine_mtrxline2parameters(predicted_line)
      elif self.registration_method=='matching_points':
       predcited_points = predicted_line.view(-1, 2, 3, 3)
       XY_source = predcited_points[:,1,:,:]
       XY_source[:,2,:] = 1.
       XY_target = predcited_points[:,0,0:2,:]
       Affine_mtrx = torch.matmul(XY_target, torch.linalg.inv(XY_source))  
       predction = {'Affine_mtrx': Affine_mtrx,
                        'XY_source':XY_source,
                        'XY_target':XY_target, }       
      elif self.registration_method=='recurrent_matrix':
            predicted_mtrx = predicted_line.view(-1, 2, 3)
            Affine_mtrx = predicted_mtrx + input_X_batch['M_i']
            predction = {'predicted_mtrx':predicted_mtrx,'Affine_mtrx': Affine_mtrx}
      else:
       Affine_parameters = predicted_line
       Affine_mtrx = normalizedparameterline2Affine_matrx(predicted_line, device , Noise_level=0.0)#Affine_matrx_from_line(predicted_line)
       predction = {'Affine_mtrx': Affine_mtrx,'Affine_parameters': Affine_parameters}
      return predction

if Arch == 'cnn':
    IR_Model = Build_IRmodel_CNN(registration_method)
    IR_Model.to(device)



#----------------------------------------------------------


'''
def wrap_imge0(Affine_mtrx, source_img):
    #This function does not produce accurate grid.
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_img.shape,align_corners=False)
    wrapped_img = torch.nn.functional.grid_sample(source_img, grid=grd,
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)
    return wrapped_img
'''
