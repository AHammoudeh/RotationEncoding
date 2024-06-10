import os, sys
import numpy as np
from PIL import Image
import itertools
import glob
import random
import time
import json
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import relu as RLU
import torch.multiprocessing as mp
from torchvision.models import resnet18
import matplotlib.pyplot as plt
#sys.path.append(os.getcwd()) 
#from gthms import *
#destandarize_point(6)
device_num = 2
torch.cuda.set_device(device_num)

Intitial_Tx = ['angle', 'scale', 'translationX','translationY','shearX','shearY']
global Noise_level_dataset
Noise_level_dataset=0.5
batch_size = 80

overlap='vertical' #'horizontal'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATASET_generation_split = { 'train': 'active',
'val': 'active',
'test': 'active',
'test0':'active'}

DATASET_generation = DATASET_generation_split['train']#'active' #{'passive', #'active'}

registration_method = 'Additive_Recurence' #{'Rawblock', 'matching_points', 'Additive_Recurence', 'Multiplicative_Recurence'} #'recurrent_matrix',
imposed_point = 0

Arch = 'ResNet'#{'VTX', 'U-Net','ResNet', 'DINO' , }
Glopooling = 'avg'#'max', 'avg', 'none'

with_scheduler = True

IMG_noise = False
LYR_NORM = False
Fix_Torch_Wrap = False
BW_Position = False
BLOCKED_REGION = 0
dim = 128
dim0 =224
crop_ratio = dim/dim0

if Fix_Torch_Wrap:
    Noise_level_dataset = 0.1
else:
    Noise_level_dataset = 0.5

folder_suffix = ''

if BW_Position:
    folder_suffix += '.BWPosition'

if IMG_noise:
    IMG_noise_level = 0.1
    folder_suffix += 'ImgNoise{}_'.format(IMG_noise_level)

if 'Recurence' in registration_method :
    folder_suffix += '.completely_random'#'random_aff_param'#'completely_random' #'scheduledrandom' #zero

if registration_method == 'matching_points':
    with_epch_loss = True
    if imposed_point>0:
        folder_suffix += str(imposed_point)
else:
    with_epch_loss = False

if Fix_Torch_Wrap:
    folder_suffix += 'AdustMtrx_'
else:
    folder_suffix += 'Adjustplot_'

folder_suffix+= Glopooling+'Pool_'

if LYR_NORM:
    folder_suffix += 'LyrNORM_'

if with_scheduler:
    folder_suffix += 'LR_'

if overlap=='vertical':
    folder_suffix += 'Voverlap_'

if registration_method=='matching_points':
    N_parameters = 18
else:
    N_parameters = 6

activedata_root = '/gpfs/projects/acad/maiaone/dataset/224/'#'../../localdb/224/'
saveddata_root = 'False'

routes_source={}
for x in ['train', 'val', 'test','test0']:
    if DATASET_generation_split[x] == 'active':
        if x=='test0':
            if DATASET_generation_split[x]==DATASET_generation_split['test']:
                routes_source[x] = routes_source['test']
            else:
                routes_source[x] = glob.glob(activedata_root+'test'+"/**/*.JPEG", recursive = True)
        else:
            routes_source[x] = glob.glob(activedata_root+x+"/**/*.JPEG", recursive = True)
    else:
        routes_source[x] = glob.glob(saveddata_root+x+'/source'+str(Noise_level_dataset)+"**/*.JPEG", recursive = True)

'''
file_savingfolder = '/home/ahmadh/MIR_savedmodel/{}_Mi0:{}_{}_{}_{}/'.format(
            registration_method,folder_suffix, Arch, DATASET_generation,Noise_level_dataset)
os.system('mkdir '+ file_savingfolder)
os.system('mkdir '+ file_savingfolder+ 'MIRexamples')
'''
file_savingfolder = '/gpfs/home/acad/umons-artint/freepal/MIR_savedmodel/{}_Mi0:{}_{}_{}_{}/'.format(
            registration_method,folder_suffix, Arch, DATASET_generation,Noise_level_dataset)
#file_savingfolder = file_savingfolder[:-1] +'shuffle_trainloader/'
os.system('mkdir '+ file_savingfolder)
os.system('mkdir '+ file_savingfolder+ 'MIRexamples')


def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx): #,'shearX','shearY'
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL)
    return wrapped_img, Affine_mtrx, Affine_parameters




##-----------------------------------------
##---------------Gthms--------------------

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


def wrap_imge_uncropped(Affine_mtrx, source_img224, dim2=128):
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

def generate_standard_elips(N_samples = 100, a= 1,b = 1, with_direction=False):
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
    if with_direction:
        x_direction = torch.tensor([center]*5)
        y_direction = torch.tensor([radius-0.02, radius-0.06, radius-0.010, radius-0.05,radius-0.01])
        x = torch.cat([x1_ordered,x_direction, x2_ordered])
        y = torch.cat([y1, y_direction, y2])
    else:
        x = torch.cat([x1_ordered, x2_ordered])
        y = torch.cat([y1, y2])
    return x, y


def generate_standard_mark(N_samples = 100, a= 2,b = 1, start=-0.5):
    radius = 0.25
    center = 0
    a0 = start*radius*a
    N_samples1 = int(N_samples/2 - 1)
    N_samples2 = N_samples - N_samples1
    x1 = torch.linspace(center-radius*a-a0,center + radius*a,N_samples1 ) #torch.distributions.uniform.Uniform(center-radius*a-a0,center + radius*a).sample([N_samples1])
    x1_ordered = torch.sort(x1).values
    y1 = center + b*torch.sqrt(radius**2 - ((x1_ordered-center)/a)**2)
    x2 = torch.linspace(center-radius*a-a0,center + radius*a,N_samples2 ) #torch.distributions.uniform.Uniform(center-radius*a-a0,center + radius*a).sample([N_samples2])
    x2_ordered = torch.sort(x2, descending=True).values
    y2 = center - b*torch.sqrt(radius**2 - ((x2_ordered-center)/a)**2)
    dx=a0
    dy = center + b*torch.sqrt(radius**2 - ((torch.tensor(dx)-center)/a)**2)
    S=dy/dx
    N_pointstri=int(N_samples1/2)
    Xtri_samples1 = torch.linspace(dx,0,N_pointstri ) #torch.distributions.uniform.Uniform(dx,0).sample([N_pointstri])
    Xtri_samples1 = torch.sort(Xtri_samples1).values
    Ytri_samples1 = -S*Xtri_samples1
    Xtri_samples2 = torch.linspace(dx*0.5,0, int(N_pointstri/2)) #torch.distributions.uniform.Uniform(dx*0.5,0).sample([int(N_pointstri/2)])
    Xtri_samples2 = torch.sort(Xtri_samples2, descending=True).values
    Ytri_samples2 = S*Xtri_samples2
    Xline_samples = torch.linspace(dx*0.5,radius*a*0.8, int(N_pointstri/2)) #torch.distributions.uniform.Uniform(dx*0.5,radius*a*0.8).sample([int(N_pointstri/2)])
    Xline_samples = torch.sort(Xline_samples).values
    Yline_samples = 0*Xline_samples + radius*b*0.5
    x = torch.cat([x1_ordered, x2_ordered, Xtri_samples1, Xtri_samples2, Xline_samples])
    y = torch.cat([y1, y2, Ytri_samples1, Ytri_samples2, Yline_samples])
    return x, y

def transform_standard_points(Affine_mat, x,y):
    XY = torch.ones([3,x.shape[0]])
    XY[0,:]= x
    XY[1,:]= y
    XYt = torch.matmul(Affine_mat.to('cpu').detach(),XY)
    xt0 = XYt[0]
    yt0 = XYt[1]
    return xt0, yt0

def save_examples(model, loader , n_examples = 4, plt_elipses=True,plt_imgs=False, time=1, feed_origion=True, shadow=False, win = 5):
    #prepare figures
    if shadow:
        Pixel_NCC = Torch_NCC(win=win,stride = 1).ncc
        plots_per_example = 5
    else:
        plots_per_example = 3
    figure = plt.figure()
    gs = figure.add_gridspec(n_examples, plots_per_example, hspace=0.05, wspace=0)
    axis = gs.subplots(sharex='col', sharey='row')
    figure.set_figheight(5*n_examples)
    figure.set_figwidth(5*plots_per_example)
    with torch.no_grad():
        dataiter = iter(loader)
        for k in range(n_examples):
            for select in range(time):
                X_batch, Y_batch = next(dataiter)
            X_batch = move_dict2device(X_batch,device)
            Y_batch = move_dict2device(Y_batch,device)
            pred = model(X_batch)
            Affine_mtrx = pred['Affine_mtrx']
            if feed_origion:
                source_origion = X_batch['source_origion'].to(device)
            source = X_batch['source'].to(device)
            target = X_batch['target'].to(device)
            if Fix_Torch_Wrap:
                wrapped_img = wrap_imge0(Affine_mtrx, source)
                M_GroundTruth = workaround_matrix(Y_batch['Affine_mtrx'].detach(), acc = 0.5)
                M_Predicted = workaround_matrix(pred['Affine_mtrx'].detach(), acc = 0.5)
            else:
                if feed_origion:
                    wrapped_img = wrap_imge_uncropped(Affine_mtrx, source_origion)
                else:
                    wrapped_img = wrap_imge_cropped(Affine_mtrx, source)
                M_GroundTruth = workaround_matrix(Y_batch['Affine_mtrx'].detach(), acc = 0.5/crop_ratio)
                M_Predicted = workaround_matrix(pred['Affine_mtrx'].detach(), acc = 0.5/crop_ratio)
            if shadow:
                shadow_img = Pixel_NCC(target.detach(), wrapped_img.detach())
                #'''
                colored_shadow = torch.zeros_like(wrapped_img)
                Green = shadow_img.clone()
                Red = 1-Green
                #Red = shadow_img.clone()
                #Blue = shadow_img.clone()
                #Green[shadow_img<0.7] = 0
                #Red[shadow_img>0.3] = 0
                #Blue[torch.abs(shadow_img-0.5)>0.2] = 0
                colored_shadow[:,0:1,:,:] = Red
                colored_shadow[:,1:2,:,:] = Green
                #colored_shadow[:,2:3,:,:] = Blue#'''
                Error_map = torch.abs(target.detach()-wrapped_img.detach())
            if plt_elipses:
                x0_source, y0_source = generate_standard_mark()#N_samples = 50, a= 2,b = 1, start=-0.5)
                #x0_source, y0_source = generate_standard_elips(N_samples = 100)
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
                axis[k, 2].plot(x_target,y_target, color ='black',marker='', linewidth = 2)
                axis[k, 2].plot(x_transformed,y_transformed, color ='red',marker='x', linewidth = 2)
            if plt_imgs:
                axis[k, 0].imshow(torchvision.transforms.ToPILImage()(source[k]))
                axis[k, 1].imshow(torchvision.transforms.ToPILImage()(target[k]))
                axis[k, 2].imshow(torchvision.transforms.ToPILImage()(wrapped_img[k]))
            if shadow:
                axis[k, 3].imshow(torchvision.transforms.ToPILImage()(colored_shadow[k]),alpha=0.5)
                #axis[k, 3].imshow(torchvision.transforms.ToPILImage()(shadow_img[k]), cmap='gray')
                #axis[k, 3].imshow(torchvision.transforms.ToPILImage()(shadow_img[k]), cmap='RdYlGn')
                axis[k, 4].imshow(torchvision.transforms.ToPILImage()(Error_map[k]),alpha=0.9)
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

'''
shadow_img = Pixel_NCC(target.detach(), wrapped_img.detach())
colored_shadow = torch.zeros_like(wrapped_img)
Green = shadow_img.clone()
Red = shadow_img.clone()
Green[Green<=0.5] = 0
Red[Red>0.5] = 0
colored_shadow[:,0:1,:,:] = Red
colored_shadow[:,1:2,:,:] = Green
'''
#save_examples(IR_Model,testloader0, n_examples = 6, plt_elipses=True,plt_imgs=True,time=5, feed_origion=True, shadow=True, win=5)


def save_n_examples(model, loader,n_times=2, n_examples_per_time = 4 ):
    for m in range(n_times):
        save_examples(model, loader, n_examples = n_examples_per_time, plt_elipses=True, plt_imgs=False, time =m)
        save_examples(model, loader, n_examples = n_examples_per_time, plt_elipses=False, plt_imgs=True, time =m)
        save_examples(model, loader, n_examples = n_examples_per_time, plt_elipses=True, plt_imgs=True, time =m)

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
'''
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
'''

def pass_augment_img(image0, measure,NOISE_LEVEL=0, MODE='nearest', blocked_region =0 ):
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
    if 'angle' in measure: #angle = np.random.uniform(0, 1) #1 is equivalent to 360 degrees or 2pi = 6.29
        if np.random.rand()<0.5:
            angle = np.random.uniform(0, 0.5-blocked_region)
        else:
            angle = np.random.uniform(0.5+blocked_region, 1)
    if 'scale' in measure: scale_inv = np.random.uniform(0+blocked_region/2, 1-blocked_region/2)
    if 'translationX' in measure: translationx = np.random.uniform(0+blocked_region/2, 1-blocked_region/2)
    if 'translationY' in measure: translationy = np.random.uniform(0+blocked_region/2, 1-blocked_region/2)
    if 'shearX' in measure: shearX = np.random.uniform(0, 1-blocked_region)
    if 'shearY' in measure: shearY = np.random.uniform(0, 1-blocked_region)
    Affine_parameters = torch.tensor([[angle, scale_inv,translationx,translationy, shearX,shearY]]).to(torch.float32)
    Affine_mtrx = normalizedparameterline2Affine_matrx(Affine_parameters, device='cpu', Noise_level=NOISE_LEVEL)
    if Fix_Torch_Wrap:
        wrapped_img = wrap_imge_with_inverse(Affine_mtrx.detach(), image0, ratio= 2*crop_ratio)
        Affine_mtrx = workaround_matrix(Affine_mtrx.detach(), acc = 2).detach()
    else:
        wrapped_img = wrap_imge0(Affine_mtrx, image0)
    return wrapped_img, Affine_mtrx, Affine_parameters


def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx, blocked_region = BLOCKED_REGION): #,'shearX','shearY'
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region = blocked_region)
    return wrapped_img, Affine_mtrx, Affine_parameters



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
  Outputs['source_origion'] = image0[0]
  Outputs['source'] = image1[0]
  Outputs['target'] = transformed_img[0]
  Outputs['Affine_parameters'] = Affine_parameters[0]
  return Outputs


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

def add_noise2img(img0, sigma = 0.1):
    img0 = img0.detach().to('cpu')
    s1 = sigma*torch.rand(1)#.to(device)
    if torch.rand(1)>0.5:
        gamma = 0.2+0.8*torch.rand(1)+0.1#.to(device)
    else:
        gamma = 1+4*torch.rand(1)#.to(device)
    c = torch.normal(torch.zeros_like(img0), s1*torch.ones_like(img0))
    img0_clipped = torch.clip(img0*(1 + c),0,1)
    img_adjusted0 = (img0_clipped)**(gamma)
    img_adjusted_clipped = torch.clip(img_adjusted0,0,1)
    return img_adjusted_clipped.detach()

def Generate_Mi(folder_suffix='completely_random', mode='test', noise=0.5,Affine_mtrx=torch.zeros([2,3]) ):
    if 'completely_random' in folder_suffix :
        M_i = torch.normal(torch.zeros([2,3]), torch.ones([2,3]))
    elif 'random_aff_param' in folder_suffix:
        M_i =  Generate_random_AffineMatrix(measure=Intitial_Tx, NOISE_LEVEL=noise, device='cpu')[0][0]
    elif 'mix' in folder_suffix:
        if mode == 'train':
            M_i = Affine_mtrx + Affine_mtrx*torch.normal(torch.zeros([2,3]), torch.rand([2,3]))
        else:
            M_i = torch.normal(torch.zeros([2,3]), torch.rand([2,3]))
    else:
        if (mode == 'train') and (torch.rand([1])<0.5):
            scheduled_noise = scheduled_parameters()
            M_i = Affine_mtrx + Affine_mtrx*torch.normal(torch.zeros([2,3]), scheduled_noise*torch.ones([2,3]))
        else:
            M_i = torch.zeros_like(Affine_mtrx)
    return M_i

class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_paths, dim = 128,
                            batch_size = 25,
                            DATASET={'train':'active','val':'','test':''},
                            Img_noise = False,
                            registration_method= 'Additive_Recurence',
                            folder_suffix = 'random',
                            mode= 'train' ):
        self.list_paths = list_paths[mode]
        self.dim = dim
        self.mode = mode
        self.activedataset = DATASET[self.mode]
        self.batch_size = batch_size
        self.number_examples = len(self.list_paths)
        self.registration_method = registration_method
        self.folder_suffix = folder_suffix
        self.Img_noise = Img_noise
  def __len__(self):
        return int(self.batch_size*(self.number_examples // self.batch_size))
  def __getitem__(self, index):
        source_image_path = self.list_paths[index]
        if self.activedataset == 'active':
            Outputs = Load_augment(source_image_path, dim = self.dim)#,Img_noise=self.Img_noise)
            source_origion = Outputs['source_origion'].to(torch.float32)
            source = Outputs['source'].to(torch.float32)
            target = Outputs['target'].to(torch.float32)
            Affine_mtrx = Outputs['Affine_mtrx']#.to(torch.float32)
            #Affine_parameters = Outputs['Affine_parameters']#.to(torch.float32)
        else:
            source_image_path = self.list_paths[index]
            source = load_image_pil_accelerated(source_image_path).to(torch.float32)
            target = load_image_pil_accelerated(source_image_path.replace('source','target')).to(torch.float32)
            with open(source_image_path.replace('source','matrix').replace('.JPEG','.npy'), 'rb') as f:
                Affine_mtrx = torch.from_numpy(np.load(f))
        if self.Img_noise:
            source_origion = add_noise2img(source_origion)
            source = add_noise2img(source)
            target = add_noise2img(target)
        if 'Recurence' in self.registration_method:
            M_i = Generate_Mi(folder_suffix=self.folder_suffix, mode=self.mode,
                    noise=0.5, Affine_mtrx= Affine_mtrx )
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
        if 'test' in self.mode:
            X['source_origion'] = source_origion
        return X,Y

class Build_IRmodel_Resnet(nn.Module):
    def __init__(self, resnet_model, registration_method = 'Rawblock', Glopooling = 'avg', LYR_NORM=True, BW_Position=False, overlap='horizontal'):
        super(Build_IRmodel_Resnet, self).__init__()
        self.resnet_model = resnet_model
        self.LYR_NORM = LYR_NORM
        self.BW_Position = BW_Position
        self.Glopooling = Glopooling#'max'#'max', 'avg', 'none', ''
        self.adapmaxpool = nn.AdaptiveMaxPool2d((1,1))
        self.overlap=overlap
        self.Layer_Norm = nn.LayerNorm([3,128,128])
        if registration_method=='matching_points':
            self.N_parameters = 18
        else:
            self.N_parameters = 6
        self.registration_method = registration_method
        self.fc1 =nn.Linear(6, 64)
        if self.overlap == 'vertical':
            self.fc2 =nn.Linear(64, 128*6)
        else:
            self.fc2 =nn.Linear(64, 128*3)
        if self.Glopooling=='none':
            self.flat = torch.nn.Flatten()
            if 'Recurence' in self.registration_method:
                self.fci = nn.Linear(18432, 100)
                self.ReLU = nn.ReLU()
                self.fcii = nn.Linear(100, 100)
                self.fc3 =nn.Linear(100, self.N_parameters)
            else:
                self.fc3 =nn.Linear(16840, self.N_parameters) #check this number
        else:
             self.fc3 =nn.Linear(512, self.N_parameters)
        #self.global_avg_pooling = nn.AdaptiveAvgPool2d(10)
    def forward(self, input_X_batch):
        source = input_X_batch['source']
        target = input_X_batch['target']
        if self.LYR_NORM:
            source = self.Layer_Norm(source)
            target = self.Layer_Norm(target)
        if self.BW_Position:
            source = batch_ToBWposition(source)
            target = batch_ToBWposition(target)
        if 'Recurence' in self.registration_method:
            M_i = input_X_batch['M_i'].view(-1, 6)
            #-------------------------------------
            M_rep = F.relu(self.fc1(M_i))
            if self.overlap == 'vertical':
                M_rep = F.relu(self.fc2(M_rep)).view(-1,6,1,128)
                concatenated_input = torch.cat((source,target), dim=1)
                concatenated_input = torch.cat((concatenated_input,M_rep), dim=2)
            else:
                M_rep = F.relu(self.fc2(M_rep)).view(-1,3,1,128)
            #-------------------------------------
                concatenated_input = torch.cat((source,target,M_rep), dim=2)
        else:
            if self.overlap == 'vertical':
                concatenated_input = torch.cat((source,target), dim=1)
            else:
                concatenated_input = torch.cat((source,target), dim=2)
        resnet_output = self.resnet_model(concatenated_input)
        if self.Glopooling=='none':
            resnet_output = self.flat(resnet_output)
            resnet_output = self.ReLU(self.fci(resnet_output))
            resnet_output = self.ReLU(self.fcii(resnet_output))
        elif self.Glopooling=='max':
            resnet_output = self.adapmaxpool(resnet_output).squeeze()
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


#---------------Train--------------------------------
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

def test_loss(model, loader, max_iterations=100, key = 'Affine_mtrx'):
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

class Build_Ensemble_IRmodel(nn.Module):
    def __init__(self, model, Num_iterations=10, key = 'Affine_mtrx',mode='test', Matrix_initialization=folder_suffix,MI_noise=1,AM=torch.zeros([batch_size,2,3])):
        super(Build_Ensemble_IRmodel, self).__init__()
        self.model = model
        self.Matrix_initialization = Matrix_initialization
        self.MI_noise = MI_noise
        self.AM = AM
        self.mode = mode
        self.Num_iterations = Num_iterations
        self.key =key
    def forward(self, input_X_batch):
        predction= {}
        input_X_batch['M_i'] = Generate_Mi_batch(batch_size= batch_size, Matrix_initialization=self.Matrix_initialization, mode=self.mode,
                        MI_noise=self.MI_noise,Affine_mtrx=self.AM ).to(device)
        predction[self.key] = self.model(input_X_batch)[self.key]
        for i in range(1,self.Num_iterations):
            input_X_batch['M_i'] = Generate_Mi_batch(batch_size= batch_size, Matrix_initialization=self.Matrix_initialization, mode=self.mode,
                        MI_noise=self.MI_noise,Affine_mtrx=self.AM).to(device)
            #input_X_batch['M_i'] = torch.zeros_like(input_X_batch['M_i'])
            predction[self.key] += self.model(input_X_batch)[self.key].detach()
        predction[self.key]/= self.Num_iterations
        return predction










































#---------------Data--------------------------------
#---------------------------------------------------
#---------------------------------------------------
train_set = Dataset(list_paths=routes_source,  batch_size=batch_size, DATASET=DATASET_generation_split,Img_noise= IMG_noise,
            registration_method =registration_method,folder_suffix = folder_suffix, mode='train')
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,  shuffle=False) #num_workers=4,
#-------------
val_set = Dataset(list_paths=routes_source, batch_size=batch_size, DATASET=DATASET_generation_split,Img_noise= False,
                    registration_method =registration_method,folder_suffix = folder_suffix,  mode='val')

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=True)
#-------------
'''
X_item, Y_item = val_set.__getitem__(5)
dataiter = iter(valloader)
X_batch, Y_batch = next(dataiter)

dataiter = iter(trainloader)
X_batch, Y_batch = next(dataiter)
'''

#---------------Model--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#ResNet
#----------------------------------------------------------

if Arch == 'ResNet':
    core_model = resnet18(weights='ResNet18_Weights.DEFAULT')#(pretrained=True), (weights='ResNet18_Weights.IMAGENET1K_V1')
    if Glopooling == 'avg': #with_pooling:
        core_model.fc = Identity()
    else:
        core_model.avgpool = Identity()
        core_model = torch.nn.Sequential(*list(core_model.children())[:-2])
    if overlap=='vertical':
            # Get the original first convolutional layer 
        original_conv1 = core_model.conv1 
        # Create a new convolutional layer with 6 input channels 
        new_conv1 = nn.Conv2d(in_channels=6, out_channels=original_conv1.out_channels, kernel_size=original_conv1.kernel_size, stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias) 
        # Initialize the new convolutional layer's weights
        with torch.no_grad(): 
            new_conv1.weight[:, :3, :, :] = original_conv1.weight 
            new_conv1.weight[:, 3:, :, :] = original_conv1.weight 
        # Replace the first convolutional layer in ResNet-18 
        core_model.conv1 = new_conv1
    IR_Model = Build_IRmodel_Resnet(core_model, registration_method,Glopooling, LYR_NORM=LYR_NORM, BW_Position=False, overlap=overlap)
    core_model.to(device)
    IR_Model.to(device)

'''
dataiter = iter(valloader)
X_batch, Y_batch = next(dataiter)

X_batch = move_dict2device(X_batch,device)
Y_batch = move_dict2device(Y_batch,device)

pred = IR_Model(X_batch)
Affine_mtrx = pred['Affine_mtrx']
'''



















#---------------Train--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------

Learning_rate = 0.001
print_every = 2000

MSE_loss = torch.nn.functional.mse_loss
optimizer = optim.AdamW(IR_Model.parameters(), lr=Learning_rate)#, momentum=0.9

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

training_loss_epochslist = []
validation_loss_epochslist = []
training_loss_iterationlist = []
validation_loss_iterationlist = []

TOTAL_Epochs = 12
best_loss = 100000000000000000000
for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
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
            if i % print_every == 0:
                eval_loss_x = eval_loss(valloader,int(2000/batch_size),'Affine_mtrx').detach().item()
                if with_scheduler:
                    scheduler.step(eval_loss_x)
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
    #save_examples(IR_Model,valloader, n_examples = 6, plt_elipses=True,plt_imgs=True)
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
ext = '_EndTraining'#'_EndTraining' #_bestVal

if Arch == 'ResNet':
    core_model_tst = resnet18(pretrained=True)#weights='ResNet18_Weights.IMAGENET1K_V1')
    if Glopooling == 'avg': #with_pooling:
        core_model_tst.fc = Identity()
    else:
        core_model_tst.avgpool = Identity()
        core_model_tst = torch.nn.Sequential(*list(core_model_tst.children())[:-2])
    if overlap=='vertical':
            # Get the original first convolutional layer 
        original_conv1 = core_model_tst.conv1 
        # Create a new convolutional layer with 6 input channels 
        new_conv1 = nn.Conv2d(in_channels=6, out_channels=original_conv1.out_channels, kernel_size=original_conv1.kernel_size, stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias) 
        # Initialize the new convolutional layer's weights
        with torch.no_grad(): 
            new_conv1.weight[:, :3, :, :] = original_conv1.weight 
            new_conv1.weight[:, 3:, :, :] = original_conv1.weight 
        # Replace the first convolutional layer in ResNet-18 
        core_model_tst.conv1 = new_conv1
    core_model_tst.load_state_dict(torch.load(file_savingfolder+'core_model'+ext+'.pth'))
    IR_Model_tst = Build_IRmodel_Resnet(core_model_tst, registration_method,Glopooling,LYR_NORM=LYR_NORM, BW_Position=False, overlap=overlap)


core_model_tst.to(device)
IR_Model_tst.load_state_dict(torch.load(file_savingfolder+'IR_Model'+ext+'.pth'))
IR_Model_tst.to(device)
IR_Model_tst.eval()


#----------------------------
#Load test dataset

test_set = Dataset(list_paths=routes_source, batch_size=batch_size, DATASET=DATASET_generation_split,Img_noise= True,
            registration_method =registration_method,folder_suffix = folder_suffix, mode='test')
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)


test_set0 = Dataset(list_paths=routes_source, batch_size=batch_size, DATASET=DATASET_generation_split,Img_noise= False,
            registration_method =registration_method,folder_suffix = folder_suffix, mode='test')
testloader0 = torch.utils.data.DataLoader(test_set0, batch_size=batch_size,shuffle=False)


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

MSE_loss = torch.nn.functional.mse_loss

MSE_affine = test_loss(IR_Model_tst, testloader,100, key = 'Affine_mtrx').detach().item()
print(MSE_affine)
json.dump( MSE_affine, open( file_savingfolder+'Test_loss.txt', 'w' ) )

MSE_affine = test_loss(IR_Model_tst, testloader0,100, key = 'Affine_mtrx').detach().item()
print(MSE_affine)
json.dump( MSE_affine, open( file_savingfolder+'Test_loss0.txt', 'w' ) )

save_examples(IR_Model_tst,testloader, n_examples = 6, plt_elipses=True,plt_imgs=True, feed_origion=True)
save_n_examples(IR_Model_tst, testloader, n_times=2, n_examples_per_time = 4 )

save_examples(IR_Model,testloader0, n_examples = 6, plt_elipses=True,plt_imgs=True,time=2, feed_origion=True)



#----------------------------------------------------------
#--------------- Evaluation using a Marker --------------------------------
#----------------------------------------------------------



def evaluate_model(model, loader, max_iterations=100, measures = { 'AM_MSE':True, 'Marker_RMSE':True, 'Marker_HD95':False}):
    eval_measures={}
    MSE_mtrx_tot = 0
    Mark_RMSE_tot =0
    Mark_HD_tot =0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = model(inputs)
                if measures['AM_MSE']:
                    MSE_mtrx_tot += MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'].detach())
                    MSE_mtrx_avg = MSE_mtrx_tot/i
                    eval_measures['AM_MSE'] = MSE_mtrx_avg
                Mark_dists = evaluate_distances(labels,predections)
                # RMSE of markers
                if measures['Marker_RMSE']:
                    Mark_RMSE = torch.sqrt(torch.mean(Mark_dists['Square_Error']))
                    Mark_RMSE_tot += Mark_RMSE
                    Mark_RMSE_avg = Mark_RMSE_tot/i
                    eval_measures['Marker_RMSE'] = Mark_RMSE_avg
                # HD of markers
                if measures['Marker_HD95']:
                    Mark_HD_mean = torch.mean(Mark_dists['HD95'])
                    Mark_HD_tot += Mark_HD_mean
                    Mark_HD_avg = Mark_HD_tot/i
                    eval_measures['Marker_HD95'] = Mark_HD_avg
            else:
                return eval_measures
    return eval_measures

def evaluate_distances(labels,predections):
    x0_source, y0_source = generate_standard_mark(N_samples = 50)
    N_points = x0_source.shape[0]
    batch_point_cloud1 = torch.zeros(labels['Affine_mtrx'].shape[0], N_points,2).to(torch.float32).to(device)
    batch_point_cloud2 = torch.zeros(labels['Affine_mtrx'].shape[0], N_points,2).to(torch.float32).to(device)
    for k in range(labels['Affine_mtrx'].shape[0]):
        x0_target, y0_target = transform_standard_points(labels['Affine_mtrx'][k], x0_source, y0_source)
        x0_transformed, y0_transformed = transform_standard_points(predections['Affine_mtrx'][k], x0_source, y0_source)
        x_source = destandarize_point(x0_source, dim=dim, flip = False)
        x_target = destandarize_point(x0_target, dim=dim, flip = False)
        x_transformed = destandarize_point(x0_transformed, dim=dim, flip = False)
        y_source = destandarize_point(y0_source, dim=dim, flip = False)
        y_target = destandarize_point(y0_target, dim=dim, flip = False)
        y_transformed = destandarize_point(y0_transformed, dim=dim, flip = False)
        batch_point_cloud1[k,:,0]= x_target
        batch_point_cloud1[k,:,1]= y_target
        batch_point_cloud2[k,:,0]= x_transformed
        batch_point_cloud2[k,:,1]= y_transformed
    #RMSE
    SE = (batch_point_cloud1-batch_point_cloud2)**2
    # Hausorff distance
    HD_dist= batch_hausdorff_prcnt_distances(batch_point_cloud1, batch_point_cloud2, percentile = 0.95)
    return {'Square_Error':SE, 'HD95':HD_dist}

def batch_hausdorff_prcnt_distances(batch_point_cloud1, batch_point_cloud2, percentile = 0.95):
    distances = torch.norm(batch_point_cloud1[:,:, None, :] - batch_point_cloud2[:, None, :,:], dim=-1)
    dists1 = torch.min(distances, dim=-1).values
    dists2 = torch.min(distances, dim=-2).values
    # Calculate the 95th percentile distance
    percentile_95 = torch.quantile(torch.cat([dists1, dists2],axis=1), percentile, interpolation='linear', dim=1)
    return percentile_95

evaluate_model(IR_Model, testloader0, max_iterations = 100, measures = { 'AM_MSE':True, 'Marker_RMSE':True, 'Marker_HD95':True})





#----------------------------------------------------------
#--------------- Evaluation of basic transformations --------------------------------
#----------------------------------------------------------
Measures_list = ['angle', 'scale', 'translationX','translationY','shearX','shearY']
num2txt= {  '0': 'angle', '1': 'scale', '2': 'translationX','3': 'translationY', '4': 'shearX','5': 'shearY'}
txt2num ={'angle':0, 'scale':1, 'translationX':2, 'translationY':3, 'shearX':4,'shearY':5}
BLOCKED_REGION_options = {'Hard0%':0, 'Medium30%':0.3, 'Easy60%':0.6}

IR_Model_tst=IR_Model.eval()


##------------------------------------------------------------------
## Deviation of each augmentation parameter from the the ground-truth
##------------------------------------------------------------------
MSE_AffineMatrix_1measure = {}
for MEASURE in Measures_list:
    MEASURES= [MEASURE]
    Noise_level_testset = 0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, blocked_region = BLOCKED_REGION): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region = blocked_region)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_1measure[MEASURE] = test_loss(IR_Model_tst,testloader0,200, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_1measure)
json.dump(MSE_AffineMatrix_1measure, open( file_savingfolder+'Test_loss_1Transformation.txt', 'w' ) )

'''
MEASURES= ['scale']
def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES): 
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region = BLOCKED_REGION)
    return wrapped_img, Affine_mtrx, Affine_parameters

test_loss(IR_Model_tst,testloader0,100, key = 'Affine_mtrx').detach().item()
save_examples(IR_Model,testloader0, n_examples = 6, plt_elipses=True,plt_imgs=True,time=4, feed_origion=True)
'''

##------------------------------------------------------------------
## Evaluation meaures under smaller domain of basic transformation (variant test set difficulties) for all transformations
##------------------------------------------------------------------
Noise_level_testset = 0.5
MSE_AffineMatrix_difficulty_withSigma = {}
for Difficulty in ['Hard0%', 'Medium30%', 'Easy60%']:
    Blocked_region_difficulty= BLOCKED_REGION_options[Difficulty]
    MEASURES = ['angle', 'scale', 'translationX','translationY','shearX','shearY']
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, blocked_region = Blocked_region_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region = blocked_region)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_difficulty_withSigma[Difficulty] = test_loss(IR_Model_tst,testloader0,200, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_difficulty_withSigma)
json.dump(MSE_AffineMatrix_difficulty_withSigma, open( file_savingfolder+'MSE_AffineMatrix_VariantDifficulty_AllTx_Sigma{}.txt'.format(Noise_level_testset), 'w' ) )



Noise_level_testset = 0
MSE_AffineMatrix_difficulty = {}
for Difficulty in ['Hard0%', 'Medium30%', 'Easy60%']:
    Blocked_region_difficulty= BLOCKED_REGION_options[Difficulty]
    MEASURES = ['angle', 'scale', 'translationX','translationY','shearX','shearY']
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, blocked_region = Blocked_region_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region = blocked_region)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_difficulty[Difficulty] = test_loss(IR_Model_tst,testloader0,200, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_difficulty)
json.dump(MSE_AffineMatrix_difficulty, open( file_savingfolder+'MSE_AffineMatrix_VariantDifficulty_AllTx_Sigma{}.txt'.format(Noise_level_testset), 'w' ) )


##------------------------------------------------------------------
## Evaluation meaures under smaller domain of basic transformation (variant test set difficulties) for basic transformations
##------------------------------------------------------------------

MSE_AffineMatrix_difficulty = {}
for Difficulty in ['Hard0%', 'Medium30%', 'Easy60%']:
    Blocked_region_difficulty= BLOCKED_REGION_options[Difficulty]
    MSE_AffineMatrix_1measure = {}
    for MEASURE in Measures_list:
        MEASURES= [MEASURE]
        Noise_level_testset=0
        def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, blocked_region = Blocked_region_difficulty): 
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region = blocked_region)
            return wrapped_img, Affine_mtrx, Affine_parameters
        MSE_AffineMatrix_1measure[MEASURE] = test_loss(IR_Model_tst,testloader0,200, key = 'Affine_mtrx').detach().item()
    MSE_AffineMatrix_difficulty[Difficulty] =MSE_AffineMatrix_1measure

print(MSE_AffineMatrix_difficulty)
json.dump(MSE_AffineMatrix_difficulty, open( file_savingfolder+'MSE_AffineMatrix_VariantDifficulty_BasicTx.txt', 'w' ) )
























##------------------------------------------------------------------
## Recurrent estimation of the Affine matrix
##------------------------------------------------------------------

#Affine_mtrx_j = torch.matmul( mtrx3(predections[key]), mtrx3(Affine_mtrx_j))[:,:2,:]
#Affine_mtrx_j = torch.matmul(mtrx3(Affine_mtrx_j), mtrx3(predections[key]))[:,:2,:]
#Affine_mtrx_j =torch.matmul(mtrx3(Affine_mtrx_j).permute(0,2,1), mtrx3(predections[key]).permute(0,2,1)).permute(0,1,2)[:,:2,:]
#Affine_mtrx_j =((torch.matmul(mtrx3(Affine_mtrx_j).T, mtrx3(predections[key]).T)).T)[:,:2,:]
#print(Affine_mtrx_j.shape)

class Torch_NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None,stride = 1):
        self.win = win
        self.eps = 1e-5
        self.stride = stride
    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.shape) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        # set window size
        if self.win is None:
            self.pad = 9
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.pad = self.win
            self.win = [self.win] * ndims
        # get convolution function
        conv_fn = getattr(torch.nn.functional, 'conv%dd' % ndims)
        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
        # compute filters
        in_ch = Ji.shape[1]
        # sum_filt = tf.ones([*self.win, in_ch, 1])
        sum_filt = torch.ones([1,in_ch, *self.win], dtype=Ii.dtype).to(device)
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        # compute local sums via convolution
        padding = (self.pad - 1) // 2
        #padding = (padding_s, padding_s)
        #padding = 'same'
        I_sum = conv_fn(Ii, sum_filt, stride =self.stride, padding = padding)
        J_sum = conv_fn(Ji, sum_filt, stride =self.stride, padding = padding)
        I2_sum = conv_fn(I2, sum_filt, stride =self.stride, padding = padding)
        J2_sum = conv_fn(J2, sum_filt, stride =self.stride, padding = padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride =self.stride, padding = padding)
        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross[cross<self.eps] = self.eps
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var[I_var<self.eps] = self.eps
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var[J_var<self.eps] = self.eps
        cc = (cross * cross) / (I_var * J_var + self.eps)
        return cc
    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        cc = torch.mean(cc)
        return -cc

NCC_loss = Torch_NCC(win=128,stride = 128).loss
NCC_loss(target.detach(), wrapped_img.detach()).item()

pad=32
feed_origion = True
def recurrent_loss(model,loader = testloader, max_iterations=100, No_recurences = 3, key ='Affine_mtrx'):
    PXL_MSE_tot = {}
    PXL_MSE_avg = {}
    PXL_NCC_tot = {}
    PXL_NCC_avg = {}
    for j in range(No_recurences):
        PXL_NCC_tot[str(j)]=0
        PXL_MSE_tot[str(j)]=0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                source0 = inputs['source'].to(device)
                target = inputs['target'].to(device)
                source_origion = inputs['source_origion'].to(device)
                predections = model(inputs)
                #eval_loss_tot[str(0)] += MSE_loss(target, predections[key].detach()).item()
                inputs_j = inputs
                Affine_mtrx_j = predections['Affine_mtrx'].detach()
                grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion.shape,align_corners=False)
                source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                wrapped_img = torchvision.transforms.CenterCrop((128, 128))(source_origion_224j)
                #wrapped_img = wrap_imge_uncropped(Affine_mtrx_j, source_origion)
                #source_origion_224j = source_origion
                PXL_MSE_tot[str(0)] += MSE_loss(target.detach(), wrapped_img.detach()).item()
                PXL_NCC_tot[str(0)] += NCC_loss(target.detach(), wrapped_img.detach()).item()
                for j in range(1, No_recurences):
                    inputs_j ={'source': wrapped_img,
                                'target': target,
                                'M_i': torch.zeros([int(inputs['source'].shape[0]),2,3]).to(device)
                                }
                    predections = model(inputs_j)
                    Affine_mtrx_j = predections['Affine_mtrx'].detach()
                    grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion_224j.shape,align_corners=False)
                    source_origion_224j = torch.nn.functional.grid_sample(source_origion_224j, grid=grd,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
                    wrapped_img = torchvision.transforms.CenterCrop((128, 128))(source_origion_224j)
                    #eval_loss_tot[str(j)] += MSE_loss(labels[key].detach(), predections[key].detach()).item()
                    PXL_MSE_tot[str(j)] += MSE_loss(target[wrapped_img>0.05].detach(), wrapped_img[wrapped_img>0.05].detach()).item()
                    PXL_NCC_tot[str(j)] += NCC_loss(target[:,:,pad:128-pad,pad:128-pad].detach(), wrapped_img[:,:,pad:128-pad,pad:128-pad].detach()).item()
            else:
                for r in range(No_recurences):
                    #eval_loss_avg[str(r)] = eval_loss_tot[str(r)]/max_iterations
                    PXL_MSE_avg[str(r)] = PXL_MSE_tot[str(r)]/max_iterations
                    PXL_NCC_avg[str(r)] = PXL_NCC_tot[str(r)]/max_iterations
                return PXL_MSE_avg, PXL_NCC_avg



No_recurences = 5
Noise_level_testset=0.5
def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx, blocked_region = 0): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region=blocked_region)
        return wrapped_img, Affine_mtrx, Affine_parameters
PXL_MSE_avg, PXL_NCC_avg = recurrent_loss(model=IR_Model_tst,loader=testloader0, max_iterations=200, No_recurences = No_recurences)
print('Recurrent PXL_MSE_avg when Noise_level_testset=0.5:', PXL_MSE_avg)
print('Recurrent PXL_NCC_avg when Noise_level_testset=0.5:', PXL_NCC_avg)
json.dump( PXL_MSE_avg, open( file_savingfolder+'Recurrent_PXL_MSE_avg.txt', 'w' ) )
json.dump( PXL_NCC_avg, open( file_savingfolder+'Recurrent_PXL_NCC_avg.txt', 'w' ) )




No_recurences = 5
Noise_level_testset=0.5
Pixels_MSE_AffineMatrix_difficulty = {}
Pixels_NCC_AffineMatrix_difficulty = {}

for Difficulty in ['Hard0%', 'Medium30%', 'Easy60%']:
    Blocked_region_difficulty= BLOCKED_REGION_options[Difficulty]
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx, blocked_region = Blocked_region_difficulty): 
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, blocked_region=blocked_region)
            return wrapped_img, Affine_mtrx, Affine_parameters
    Pixels_MSE_AffineMatrix_difficulty[Difficulty], Pixels_NCC_AffineMatrix_difficulty[Difficulty] = recurrent_loss(model=IR_Model_tst,loader=testloader0, max_iterations=200, No_recurences = No_recurences)

print('Recurrent PXL_MSE_avg when Noise_level_testset=0.5:', Pixels_MSE_AffineMatrix_difficulty)
print('Recurrent PXL_NCC_avg when Noise_level_testset=0.5:', Pixels_NCC_AffineMatrix_difficulty)
json.dump( Pixels_MSE_AffineMatrix_difficulty, open( file_savingfolder+'Recurrent_PXL_MSE_difficulties.txt', 'w' ) )
json.dump( Pixels_NCC_AffineMatrix_difficulty, open( file_savingfolder+'Recurrent_PXL_NCC_difficulties.txt', 'w' ) )






save_examples(IR_Model,testloader0, n_examples = 6, plt_elipses=True,plt_imgs=True,time=6, feed_origion=True, shadow=True, win=9)












'''

eval_loss_tot = {}
eval_loss_avg = {}
for j in range(No_recurences):
    eval_loss_tot[str(j)]=0

loader = testloader0
model = IR_Model_tst
max_iterations= 20
key = 'Affine_mtrx'



torchvision.transforms.ToPILImage()(source0[1]).save(file_savingfolder+'MIRexamples/source.jpeg')
torchvision.transforms.ToPILImage()(target[1]).save(file_savingfolder+'MIRexamples/trgt.jpeg')

Affine_mtrx_j = predections['Affine_mtrx'].detach()
wrapped_img = wrap_imge_uncropped(Affine_mtrx_j, source_origion)
torchvision.transforms.ToPILImage()(wrapped_img[1]).save(file_savingfolder+'MIRexamples/wrp.jpeg')
inputs_j ={'source': source0,#wrapped_img,
            'target': target,
            'M_i': torch.zeros([int(inputs['source'].shape[0]),2,3]).to(device)
            }
predections = model(inputs_j)


grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_origion_224j.shape,align_corners=False)
source_origion_224j = torch.nn.functional.grid_sample(source_origion_224j, grid=grd,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)

source_img224 = torch.nn.ZeroPad2d(int((dim1-dim2)/2))(source_img)
wrapped_img = torch.nn.functional.grid_sample(source_img224, grid=grd,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
wrapped_img = torchvision.transforms.CenterCrop((dim2, dim2))(wrapped_img)

with torch.no_grad():
    for i, data in enumerate(loader, 0):
        if i < max_iterations:
            inputs, labels = data
            inputs = move_dict2device(inputs,device)
            labels = move_dict2device(labels,device)
            source0 = inputs['source'].to(device)
            target = inputs['target'].to(device)
            source_origion = inputs['source_origion'].to(device)
            predections = model(inputs)
            #eval_loss_tot[str(0)] += MSE_loss(target, predections[key].detach()).item()
            inputs_j = inputs
            Affine_mtrx_j = predections['Affine_mtrx'].detach()
            grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion.shape,align_corners=False)
            source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
            wrapped_img = torchvision.transforms.CenterCrop((128, 128))(source_origion_224j)
            #wrapped_img = wrap_imge_uncropped(Affine_mtrx_j, source_origion)
            #source_origion_224j = source_origion
            PXL_MSE_tot[str(0)] += MSE_loss(target.detach(), wrapped_img.detach()).item()
            PXL_NCC_tot[str(0)] += NCC_loss(target.detach(), wrapped_img.detach()).item()
            for j in range(1, No_recurences):
                inputs_j ={'source': wrapped_img,
                            'target': target,
                            'M_i': torch.zeros([int(inputs['source'].shape[0]),2,3]).to(device)
                            }
                predections = model(inputs_j)
                Affine_mtrx_j = predections['Affine_mtrx'].detach()
                grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion_224j.shape,align_corners=False)
                source_origion_224j = torch.nn.functional.grid_sample(source_origion_224j, grid=grd,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                wrapped_img = torchvision.transforms.CenterCrop((128, 128))(source_origion_224j)
                #eval_loss_tot[str(j)] += MSE_loss(labels[key].detach(), predections[key].detach()).item()
                PXL_MSE_tot[str(j)] += MSE_loss(target[target>10].detach(), wrapped_img[target>10].detach()).item()
                PXL_NCC_tot[str(j)] += NCC_loss(target[:,:,pad:128-pad,pad:128-pad].detach(), wrapped_img[:,:,pad:128-pad,pad:128-pad].detach()).item()
        else:
            for r in range(No_recurences):
                #eval_loss_avg[str(r)] = eval_loss_tot[str(r)]/max_iterations
                PXL_MSE_avg[str(r)] = PXL_MSE_tot[str(r)]/max_iterations
                PXL_NCC_avg[str(r)] = PXL_NCC_tot[str(r)]/max_iterations
            return PXL_MSE_avg, PXL_NCC_avg
'''