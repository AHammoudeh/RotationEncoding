'''
conda create --name Tx
conda actiavet Tx
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install conda-forge::matplotlib


apple/mobilevitv2-2.0-imagenet1k-256
apple/mobilevitv2-1.0-imagenet1k-256
apple/coreml-FastViT-T8
apple/mobilevitv2-1.0-voc-deeplabv3
'''
#conda activate Tx

#----------------------------------------------------------------
import os, sys
import numpy as np
from PIL import Image
import itertools
import glob
import random
import math
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
import transformers
#sys.path.append(os.getcwd()) 
#from gthms import *
#destandarize_point(6)


HPC = False
SM1 = True
device_num = 2
torch.cuda.set_device(device_num)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_aug = device#'cpu'


BLOCK_options = {'Full':(0,1), 'Hard70-100%':(0.7,1), 'Medium30-70':(0.3,0.7), 'Easy10-30%':(0.1,0.3), 'Easier5-10%':(0.05,0.1), 'Easiest0-5%':(0,0.05)}
FLIP = False
Affine_noise_intensity=0.0
IMG_noise_level=0.0

#Saving folder
#------------------------------------------------------------------------------------------------------------------------

folder_prefx ='Canonical_orientation_classification'

Arch = 'mobileVIT'

if HPC:
    savingfolder = '/gpfs/home/acad/umons-artint/freepal/MIR_savedmodel/{}:{}_{}_sI:{}_sA:{}/'.format(
                folder_prefx, Arch,IMG_noise_level, Affine_noise_intensity)
else:
    savingfolder = '/home/ahmadh/MIR_savedmodel/{}:{}_sI:{}_sA:{}/'.format(
            folder_prefx, Arch,IMG_noise_level, Affine_noise_intensity)

os.system('mkdir '+ savingfolder)


#Dataset loader
#------------------------------------------------------------------------------------------------------------------------

if HPC:
    activedata_root = '/gpfs/projects/acad/maiaone/dataset/224/'
else:
    if SM1:
        activedata_root ='../../localdb/224/'
    else:
        activedata_root = '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/224/'
    import tqdm

routes_source={}
for x in ['train', 'val', 'test']:
    routes_source[x] = glob.glob(activedata_root+x+"/**/*.JPEG", recursive = True)

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


class Dataset_source(torch.utils.data.Dataset):
  def __init__(self, list_paths, batch_size = 64):
        self.list_paths = list_paths
        #self.dim = dim
        self.batch_size = batch_size
        self.number_examples = len(self.list_paths)
  def __len__(self):
        return int(self.batch_size*(self.number_examples // self.batch_size))
  def __getitem__(self, index):
        source_image_path = self.list_paths[index]
        source_origion = load_image_pil_accelerated(source_image_path)
        #X = {'source_origion':source_origion}
        #Y = {'Affine_mtrx': 0}
        return source_origion

if SM1:
    batch_size=80
else:
    batch_size = 256

train_set = Dataset_source(list_paths=routes_source['train'], batch_size=batch_size)
trainloader = torch.utils.data.DataLoader( train_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
#-------------
val_set = Dataset_source(list_paths=routes_source['val'], batch_size=batch_size)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, pin_memory=False,shuffle=True, num_workers=4, )
#-------------
'''
X_item, Y_item = val_set.__getitem__(5)


dataiter = iter(valloader)
c = next(dataiter)

c = move_dict2device(c,device)

pred = IR_Model(X_batch)
Affine_mtrx = pred['Affine_mtrx']
'''

# Dataset Preperation functions
#------------------------------------------------------------------------------------------------------------------------
def center_crop(batch, dim = 128):
    dim0 = batch.shape[-2]
    dim_start = (dim0 - dim)//2
    dim_end = dim_start+dim
    batch_cropped = batch[:,:,dim_start:dim_end,dim_start:dim_end]
    return batch_cropped

def warp_batch(Affine_mtrx, source_imgs):
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_imgs.shape,align_corners=False)
    wrapped_imgs = torch.nn.functional.grid_sample(source_imgs, grid=grd,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
    return wrapped_imgs

def mtrx3(Affine_mtrx):
    mtrx_shape = Affine_mtrx.shape
    if len(mtrx_shape)==3:
        N_Mbatches = mtrx_shape[0]
        AM3 = torch.zeros( [N_Mbatches,3,3])
        if Affine_mtrx.device.type !='cpu':
            AM3 = AM3.to(device)
        AM3[:,0:2,:] = Affine_mtrx
        AM3[:,2,2] = 1
    elif len(mtrx_shape)==2:
        N_Mbatches = 1
        AM3 = torch.zeros([3,3])
        if Affine_mtrx.device.type !='cpu':
            AM3 = AM3.to(device)
        AM3[0:2,:] = Affine_mtrx
        AM3[2,2] = 1
    return AM3

MATRIX_THRESHOLD = 6.0
def inv_AM(Affine_mtrx, threshold=MATRIX_THRESHOLD):
    AM3 = mtrx3(Affine_mtrx)
    AM_inv = torch.linalg.inv(AM3)
    AM_inv = torch.clamp(AM_inv, min=-threshold, max=threshold)
    return AM_inv[:,0:2,:]

# Add illumination noise to the images
def add_noise2batch(batch, sigma = IMG_noise_level, device='cpu'):
    N_imgs = batch.shape[0]
    batch_shape = batch.shape
    selector = torch.rand(N_imgs)
    gamma = torch.rand(N_imgs,1,1,1)
    gamma[selector<0.5]= 0.3+ 0.8*gamma[selector<0.5]
    gamma[selector>=0.5]= 1+ 4*gamma[selector>=0.5]
    if device == 'cpu':
        c = sigma*torch.randn(batch_shape)
        batch_clipped = torch.clip(batch*(1 + c),0,1)
        batch_adjusted0 = torch.pow(batch_clipped,gamma)
    else:
        c = sigma*torch.randn(batch_shape).to(device)
        batch_clipped = torch.clip(batch*(1 + c),0,1)
        batch_adjusted0 = torch.pow(batch_clipped.to(device),gamma.to(device))
    batch_adjusted_clipped = torch.clip(batch_adjusted0,0,1)
    return batch_adjusted_clipped

def add_noise2AffineMatrix(Affine_batch0, sigma = 0.1):
    batch_size = Affine_batch0.shape[0]
    random_component = torch.normal(torch.zeros([batch_size,2,3]), sigma*torch.ones([batch_size,2,3]))
    #truncate values that exceeds a threshold
    random_component = torch.clamp(random_component, min=-3.0*sigma, max=3.0*sigma)
    Affine_batch = Affine_batch0+ Affine_batch0*(random_component)
    return Affine_batch

def Batch_matrice_from_parameters(angles,scalesX,scalesY,shearsX,shearsY ,reflections ,translationsX,translationsY):
    batch_size = angles.shape[0]
    affine_matrices = torch.zeros(batch_size, 2, 3)
    # Compute the rotation matrices
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    rotation_matrices = torch.stack([
        torch.stack([cos_angles, -sin_angles], dim=1),
        torch.stack([sin_angles, cos_angles], dim=1)], dim=1)
    # Compute the scaling matrices
    scaling_matrices = torch.zeros(batch_size, 2, 2)
    scaling_matrices[:, 0, 0] = scalesX
    scaling_matrices[:, 1, 1] = scalesY
    # Compute the shearing matrices
    shear_matrices = torch.ones(batch_size, 2, 2)
    shear_matrices[:, 0, 1] = shearsX
    shear_matrices[:, 1, 0] = shearsY
    # Compute the reflection matrices
    reflection_matrices = torch.zeros(batch_size, 2, 2)#torch.eye(2).repeat(batch_size, 1, 1)
    reflection_matrices[:, 0, 0] = reflections[:, 0]
    reflection_matrices[:, 1, 1] = reflections[:, 1]
    # Combine transformations: scale -> shear -> rotate -> reflect
    transform_matrices = torch.bmm(torch.bmm(torch.bmm(scaling_matrices, shear_matrices), rotation_matrices), reflection_matrices)
    #transform_matrices = torch.bmm(torch.bmm(shear_matrices, torch.bmm(reflection_matrices, scaling_matrices)), rotation_matrices)
    # Assign to affine matrices
    affine_matrices[:, :2, :2] = transform_matrices
    affine_matrices[:, 0, 2] = translationsX
    affine_matrices[:, 1, 2] = translationsY
    return affine_matrices

def Generate_Affine_batch(batch_size, Uniscale=False):
    angle_range = [-math.pi, math.pi]
    scale_range = [0.2,1.8]
    shear_range = [-0.5,0.5]
    translation_range = [-0.25,0.25]
    angles = sample_from_2distributions(angle_range, N_samples= batch_size, PivotPoint = 0)
    scalesX = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
    if Uniscale:
        scalesY= scalesX
    else:
        scalesY = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
    shearsX = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
    shearsY = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
    translationsX = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
    translationsY = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
    reflections = 2*reflections_dist.sample((batch_size, 2))-1
    affine_matrices = Batch_matrice_from_parameters(angles,scalesX,scalesY,shearsX,shearsY,reflections,translationsX,translationsY)
    return affine_matrices


def Generate_Affine_batch_selecTx(batch_size, Tx = Intitial_Tx, Uniscale=False):
    angle_range = [-math.pi, math.pi]
    scale_range = [0.2,1.8]
    shear_range = [-0.5,0.5]
    translation_range = [-0.25,0.25]
    paramtrs={}
    if 'angle' in Tx:
        angles = sample_from_2distributions(angle_range, N_samples= batch_size, PivotPoint = 0)
        paramtrs['angles']=angles
    else:
        angles = torch.zeros(batch_size)
    if 'scaleX' in Tx:
        scalesX = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
        paramtrs['scalesX']=scalesX
    else:
        scalesX = torch.ones(batch_size)
    if 'scaleY' in Tx:
        if Uniscale:
            scalesY = scalesX
        else:
            scalesY = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
        paramtrs['scalesY']=scalesY
    else:
        scalesY = torch.ones(batch_size)
    if 'shearX' in Tx:
        shearsX = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
        paramtrs['shearsX']=shearsX
    else:
        shearsX = torch.zeros(batch_size)
    if 'shearY' in Tx:
        shearsY = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
        paramtrs['shearsY']=shearsY
    else:
        shearsY = torch.zeros(batch_size)
    if 'translationX' in Tx:
        translationsX = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
        paramtrs['translationsX']=translationsX
    else:
        translationsX = torch.zeros(batch_size)
    if 'translationY' in Tx:
        translationsY = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
        paramtrs['translationsY']=translationsY
    else:
        translationsY = torch.zeros(batch_size)
    if 'reflection' in Tx:
        reflections = 2*reflections_dist.sample((batch_size, 2))-1
        paramtrs['reflections']=reflections
    else:
        reflections = torch.ones(batch_size, 2)
    affine_matrices = Batch_matrice_from_parameters(angles,scalesX,scalesY,shearsX,shearsY,reflections,translationsX,translationsY)
    return {'affine_matrices':affine_matrices, 'param':paramtrs}

def find_sampling_boundries(dist_range, PivotPoint = 0,
                            difficulty_min = 0, difficulty_max =1):
    r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
    r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
    r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
    r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
    return r1_min, r1_max, r2_min, r2_max

def sample_from_2distributions(dist_range, N_samples= batch_size, PivotPoint = 0):
    N_samplesFromDistribution1 = random.randint(0, N_samples)
    N_samplesFromDistribution2 = N_samples - N_samplesFromDistribution1
    r1_min, r1_max, r2_min, r2_max = find_sampling_boundries(dist_range, PivotPoint=PivotPoint)
    samples = torch.cat([ torch.FloatTensor(N_samplesFromDistribution1).uniform_(r1_min, r1_max),
                        torch.FloatTensor(N_samplesFromDistribution2).uniform_(r2_min, r2_max) ])
    return samples

def generate_registration_batches(source_origion, device='cpu', Affine_noise_intensity=0,
                                IMG_noise_level=0, dim=128, Tx_select=False, Uniscale=False, with_param=False):
    batch_size = source_origion.shape[0]
    if Tx_select:
        out = Generate_Affine_batch_selecTx(batch_size, Tx = Tx_select, Uniscale=Uniscale)
        Affine_batch = out['affine_matrices']
        paramtrs = out['param']
    else:
        Affine_batch = Generate_Affine_batch(batch_size, Uniscale=Uniscale)
    #
    data_device = source_origion.device.type
    if data_device != device:
        source_origion = source_origion.to(device)
    if Affine_noise_intensity>0:
        Affine_batch = add_noise2AffineMatrix(Affine_batch, sigma= Affine_noise_intensity)
    if device !='cpu':
        #source_origion = source_origion.to(device)
        Affine_batch = Affine_batch.to(device)
    warped_img_dim0 = warp_batch(Affine_batch, source_origion)
    target = center_crop(warped_img_dim0, dim = dim)
    if IMG_noise_level>0:
        source_origion = add_noise2batch(source_origion, sigma = IMG_noise_level, device=device)
        target = add_noise2batch(target, sigma = IMG_noise_level, device=device)
    source = center_crop(source_origion, dim = dim)
    X_batch = {'source':source,'target':target,'source_origion':source_origion}
    Y_batch = {'Affine_mtrx': Affine_batch}
    if with_param:
        Y_batch['param']=paramtrs
    return X_batch, Y_batch



# Registration Dataset Preperation
#------------------------------------------------------------------------------------------------------------------------

#difficulty Initialization
dim = 128
dim0 =224
crop_ratio = dim/dim0

DIFFICULTY_MIN =0.0
DIFFICULTY_MAX = 1.0

if FLIP:
    prob_flip = 0.5*(DIFFICULTY_MAX+DIFFICULTY_MIN)
else:
    prob_flip =0

reflections_dist = torch.distributions.Categorical(torch.tensor([prob_flip, 1-prob_flip]))

def find_sampling_boundries(dist_range, PivotPoint = 0,
            difficulty_min = DIFFICULTY_MIN, difficulty_max =DIFFICULTY_MAX):
    r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
    r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
    r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
    r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
    return r1_min, r1_max, r2_min, r2_max

dataiter = iter(valloader)
source_origion = next(dataiter)

Intitial_Tx = ['angle']#, 'scaleX','scaleY',]#False, 'translationX','translationY','shearX','shearY', 'reflection'
UNISCALE = True 
WITH_PARAM = True

X_batch, Y_batch = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE, with_param=WITH_PARAM)


#------------------------------------------------------------------------------------------------------------------------


#Modeling
#------------------------------------------------------------------------------------------------------------------------

# IR MobileVIT 1 input images
#------------------------------------------------------------------------------------------------------------------------
Precision  = 0.05
N_classes = int(1/Precision)


FREEZE_stage1 = False

from transformers import MobileViTV2ForImageClassification
core_model_classification = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
original_conv1 = core_model_classification.mobilevitv2.conv_stem.convolution
new_conv1 = torch.nn.Conv2d(3,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
core_model_classification.mobilevitv2.conv_stem.convolution = new_conv1
core_model_classification.classifier = torch.nn.Linear(512, N_classes)
core_model_classification.to(device)

class Build_IR_Model_classification(torch.nn.Module):
    def __init__(self, core_model_classification):
        super(Build_IR_Model_classification, self).__init__()
        self.core_model_classification = core_model_classification
    def forward(self, input_X_batch):
        #source = input_X_batch['source']
        target = input_X_batch['target']
        #source_origion = input_X_batch['source_origion']
        #Affine_mtrx = labels['Affine_mtrx'][:,:2,:2]
        angles_classes = self.core_model_classification(target).logits
        outcome = {'angles':angles_classes}
        return outcome

IR_Model_stage1 = Build_IR_Model_classification(core_model_classification)
IR_Model_stage1.to(device)



'''
from transformers import MobileViTV2ForImageClassification
core_model_regression = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
original_conv1 = core_model_regression.mobilevitv2.conv_stem.convolution
core_model_regression.classifier = torch.nn.Linear(512, 4)
core_model_regression.to(device)


class Build_IR_Model_regression(torch.nn.Module):
    def __init__(self, core_model_regression):
        super(Build_IR_Model, self).__init__()
        self.core_model_regression = core_model_regression
    def forward(self, input_X_batch):
        source = input_X_batch['source']
        Affine_mtrx = self.core_model_regression(source).logits
        outcome = {'Affine_mtrx':Affine_mtrx}
        return outcome

IR_Model_stage1 = Build_IR_Model_regression(core_model_regression)
IR_Model_stage1.to(device)

'''

# freeze weights of the first model
if FREEZE_stage1:
    IR_Model_stage1.eval()
    for paramx in IR_Model_stage1.parameters():
        paramx.requires_grad = False

def threshold(numberX, max_threshold=2):
    if numberX>max_threshold:
        return max_threshold
    else:
        return numberX

def move_dict2device(dictionary,device):
    for key in list(dictionary.keys()):
        dictionary[key] = dictionary[key].to(device)
    return dictionary


def test_loss(model, loader, max_iterations=100, key = 'angles'):
    eval_loss_tot = 0
    with torch.no_grad():
        for i, source_origion in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE, with_param=WITH_PARAM)
                #inputs = move_dict2device(inputs,device)
                #labels = move_dict2device(labels,device)
                predections = model(inputs)
                paramtrs = labels['param']
                Y_scaled = 0.5*(paramtrs['angles']/math.pi+ 1)
                Y_classes = (Y_scaled//Precision).long()
                Y_classes_1h = one_hot(Y_classes, N_classes)
                eval_loss_tot += criterion(predections[key].detach(), Y_classes_1h.to(device))
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg

def one_hot(x,n_classes=20):
    x1h = torch.eye(n_classes)
    return x1h[x]


#Training
#------------------------------------------------------------------------------------------------------------------------
Learning_rate = 0.0001

print_every = int(2000*80/batch_size)
MSE_loss = torch.nn.functional.mse_loss
criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.AdamW(IR_Model_stage1.parameters(), lr=Learning_rate)#, momentum=0.9

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
if not SM1:
    IR_Model_stage1 = torch.compile(IR_Model_stage1)




#def train_IR_model(IR_Model, trainloader, TOTAL_Epochs = 12,):
training_loss_iterationlist = []
validation_loss_iterationlist = []
TOTAL_Epochs = 100
best_loss = 100000000000000000000
with_scheduler=True
for EPOCH in range(3, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for source_origion in loop:
        i+=1
        inputs, labels = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE, with_param=WITH_PARAM)
        if WITH_PARAM:
            paramtrs = labels['param']
            Y_scaled = 0.5*(paramtrs['angles']/math.pi+ 1)
            Y_classes = (Y_scaled//Precision).long()
            Y_classes_1h = one_hot(Y_classes, N_classes)
        #inputs = move_dict2device(inputs,device)
        #labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IR_Model_stage1(inputs)
        loss = criterion(predections['angles'], Y_classes_1h.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IR_Model_stage1, valloader, max_iterations=int(2000/batch_size),key = 'angles').detach().item()
                scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( running_loss/print_every)
                validation_loss_iterationlist.append(eval_loss_x)
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")



print('Finished Training')
    #return IR_Model, training_loss_iterationlist, validation_loss_iterationlist

torch.save(IR_Model_stage1.state_dict(), savingfolder+'./IR_Model_stage1_EndTraining.pth')
torch.save(core_model_classification.state_dict(), savingfolder+'./core_model_classification_EndTraining.pth')

with open(savingfolder+'training_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(training_loss_iterationlist))

with open(savingfolder+'validation_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(validation_loss_iterationlist))

plt.plot(training_loss_iterationlist, label = 'training loss')
plt.plot(validation_loss_iterationlist, label = 'validation loss')
plt.ylim(0,1)
plt.legend()
plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
plt.close()

os.system('mkdir '+ savingfolder+ 'MIRexamples')



for source_origion in loop:
inputs, labels = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE, with_param=WITH_PARAM)

paramtrs = labels['param']
Y_scaled = 0.5*(paramtrs['angles']/math.pi+ 1)
Y_classes = (Y_scaled//Precision).long()
Y_classes_1h = one_hot(Y_classes, N_classes)

predections = IR_Model_stage1(inputs)




# Evaluation

#Confusion matrix and probability mass matrix

IR_Model_stage1.eval()
conf_mtrx = np.zeros([N_classes,N_classes])
prob_mtrx = torch.zeros([N_classes, N_classes])
softmx = torch.nn.Softmax(dim=0)
with torch.no_grad():
    for source_origion in tqdm.tqdm(test_loader):
        inputs, labels = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE, with_param=WITH_PARAM)
                                        
        paramtrs = labels['param']
        Y_scaled = 0.5*(paramtrs['angles']/math.pi+ 1)
        GT_class = (Y_scaled//Precision).long()
        outputs = IR_Model_stage1(inputs)
        pred_class = torch.argmax(outputs, axis=-1)
    for item in range(outputs.shape[0]):
        conf_mtrx[int(GT_class[item]), int(pred_class[item])] +=1
        prob_mtrx[int(GT_class[item])] += softmx(outputs[item,:].detach())

prob_mtrx = np.around(prob_mtrx.numpy(),3)
prob_mtrx.tofile( savingfolder+f'prob_mtrx.txt', sep=', ', format='%.3f')
conf_mtrx.tofile( savingfolder+f'conf_mtrx.txt', sep=', ', format='%.3f')



def plot_table(data,save_file ,alpha =0.5):
    # Normalize the data to range between 0 and 1
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # Define a colormap
    cmap = plt.cm.Blues
    # Plotting the 2D array as a table with colored cells
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data.round(3), loc='center', cellLoc='center', colWidths=[0.1] * data.shape[1])
    # Color cells based on normalized data
    for (i, j), cell in table.get_celld().items():
        clr = cmap(norm_data[i, j])
        clr2 = (clr[0], clr[1], clr[2], alpha)
        cell.set_facecolor(clr2)
        cell.set_text_props(color='black')
    # Save the plot to a folder
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1) 
    plt.close()


plot_table(conf_mtrx, savingfolder+f'conf_mtrx_total.png')
plot_table(prob_mtrx, savingfolder+f'prob_matrx_total.png')















from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

def test_loss_x(model, loader, max_iterations=100, key = 'angles'):
    eval_loss_tot = 0
    acc = 0
    precision =0
    f1=0
    with torch.no_grad():
        for i, source_origion in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE, with_param=WITH_PARAM)
                predections = model(inputs)
                predected_classes = torch.argmax(predections[key], 1)
                Y_scaled = (labels['param'][key]/2/math.pi)+0.5
                Y_classes = (Y_scaled.long()//Precision).to(device)
                eval_loss_tot += criterion(predections[key].detach(), Y_classes)
                gt = Y_classes.numpy()
                pred = predected_classes.numpy()
                acc += accuracy_score(gt, pred)
                precision += f1_score(gt, pred, average='macro')
                f1 += accuracy_score(gt, pred, average='macro')
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg






import seaborn as sns


# Sample data (logits and labels)
logits = torch.tensor([[2.5, 0.3, 2.0], [1.0, 3.2, 1.2], [2.1, 0.5, 3.3], [1.2, 2.1, 1.5]], dtype=torch.float32)
labels = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
# Convert logits to predictions
_, preds = torch.max(logits, 1)

accuracy = accuracy_score(labels.numpy(), preds.numpy())
# Compute precision and F1-score
precision = precision_score(labels.numpy(), preds.numpy(), average='macro')
f1 = f1_score(labels.numpy(), preds.numpy(), average='macro'

# Compute confusion matrix
cm = confusion_matrix(labels.numpy(), preds.numpy())


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')#, xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()









































































#--------------------------------------------------------------------------------------------------------------------------------------------------------



'''
#----------------------------------------------------------------
import os, sys
import numpy as np
from PIL import Image
import itertools
import glob
import random
import math
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
import transformers
#sys.path.append(os.getcwd()) 
#from gthms import *
#destandarize_point(6)

device_num = 2
torch.cuda.set_device(device_num)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_aug = device#'cpu'

#Saving folder
#------------------------------------------------------------------------------------------------------------------------

HPC = False
SM1 = True
Intitial_Tx = ['angle']#, 'scaleX','scaleY']#,]#False, 'translationX','translationY','shearX','shearY', 'reflection'
UNISCALE = True 
FLIP = False

folder_prefx ='SD_universalAlign_regression_rotationUniscale_2imgs'

Affine_noise_intensity=0.0
IMG_noise_level=0.00
Arch = 'mobileVIT'

if HPC:
    savingfolder = '/gpfs/home/acad/umons-artint/freepal/MIR_savedmodel/{}:{}_{}_sI:{}_sA:{}/'.format(
                folder_prefx, Arch,IMG_noise_level, Affine_noise_intensity)
else:
    savingfolder = '/home/ahmadh/MIR_savedmodel/{}:{}_sI:{}_sA:{}/'.format(
            folder_prefx, Arch,IMG_noise_level, Affine_noise_intensity)

os.system('mkdir '+ savingfolder)


#Dataset loader
#------------------------------------------------------------------------------------------------------------------------

if HPC:
    activedata_root = '/gpfs/projects/acad/maiaone/dataset/224/'
else:
    if SM1:
        activedata_root ='../../localdb/224/'
    else:
        activedata_root = '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/224/'
    import tqdm

routes_source={}
for x in ['train', 'val', 'test']:
    routes_source[x] = glob.glob(activedata_root+x+"/**/*.JPEG", recursive = True)

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


class Dataset_source(torch.utils.data.Dataset):
  def __init__(self, list_paths, batch_size = 64):
        self.list_paths = list_paths
        #self.dim = dim
        self.batch_size = batch_size
        self.number_examples = len(self.list_paths)
  def __len__(self):
        return int(self.batch_size*(self.number_examples // self.batch_size))
  def __getitem__(self, index):
        source_image_path = self.list_paths[index]
        source_origion = load_image_pil_accelerated(source_image_path)
        #X = {'source_origion':source_origion}
        #Y = {'Affine_mtrx': 0}
        return source_origion

if SM1:
    batch_size= 80
else:
    batch_size = 256

train_set = Dataset_source(list_paths=routes_source['train'], batch_size=batch_size)
trainloader = torch.utils.data.DataLoader( train_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
#-------------
val_set = Dataset_source(list_paths=routes_source['val'], batch_size=batch_size)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, pin_memory=True,shuffle=True, num_workers=4, )

#-------------
#X_item, Y_item = val_set.__getitem__(5)
#dataiter = iter(valloader)
#c = next(dataiter)
#c = move_dict2device(c,device)

# Dataset Preperation functions
#------------------------------------------------------------------------------------------------------------------------
def center_crop(batch, dim = 128):
    dim0 = batch.shape[-2]
    dim_start = (dim0 - dim)//2
    dim_end = dim_start+dim
    batch_cropped = batch[:,:,dim_start:dim_end,dim_start:dim_end]
    return batch_cropped

def warp_batch(Affine_mtrx, source_imgs):
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_imgs.shape,align_corners=False)
    wrapped_imgs = torch.nn.functional.grid_sample(source_imgs, grid=grd,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
    return wrapped_imgs

# Add illumination noise to the images
def add_noise2batch(batch, sigma = IMG_noise_level, device='cpu'):
    N_imgs = batch.shape[0]
    batch_shape = batch.shape
    selector = torch.rand(N_imgs)
    gamma = torch.rand(N_imgs,1,1,1)
    gamma[selector<0.5]= 0.3+ 0.8*gamma[selector<0.5]
    gamma[selector>=0.5]= 1+ 4*gamma[selector>=0.5]
    if device == 'cpu':
        c = sigma*torch.randn(batch_shape)
        batch_clipped = torch.clip(batch*(1 + c),0,1)
        batch_adjusted0 = torch.pow(batch_clipped,gamma)
    else:
        c = sigma*torch.randn(batch_shape).to(device)
        batch_clipped = torch.clip(batch*(1 + c),0,1)
        batch_adjusted0 = torch.pow(batch_clipped.to(device),gamma.to(device))
    batch_adjusted_clipped = torch.clip(batch_adjusted0,0,1)
    return batch_adjusted_clipped

def add_noise2AffineMatrix(Affine_batch0, sigma = 0.1):
    batch_size = Affine_batch0.shape[0]
    random_component = torch.normal(torch.zeros([batch_size,2,3]), sigma*torch.ones([batch_size,2,3]))
    #truncate values that exceeds a threshold
    random_component = torch.clamp(random_component, min=-3.0*sigma, max=3.0*sigma)
    Affine_batch = Affine_batch0+ Affine_batch0*(random_component)
    return Affine_batch

def Batch_matrice_from_parameters(angles,scalesX,scalesY,shearsX,shearsY ,reflections ,translationsX,translationsY):
    batch_size = angles.shape[0]
    affine_matrices = torch.zeros(batch_size, 2, 3)
    # Compute the rotation matrices
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    rotation_matrices = torch.stack([
        torch.stack([cos_angles, -sin_angles], dim=1),
        torch.stack([sin_angles, cos_angles], dim=1)], dim=1)
    # Compute the scaling matrices
    scaling_matrices = torch.zeros(batch_size, 2, 2)
    scaling_matrices[:, 0, 0] = scalesX
    scaling_matrices[:, 1, 1] = scalesY
    # Compute the shearing matrices
    shear_matrices = torch.ones(batch_size, 2, 2)
    shear_matrices[:, 0, 1] = shearsX
    shear_matrices[:, 1, 0] = shearsY
    # Compute the reflection matrices
    reflection_matrices = torch.zeros(batch_size, 2, 2)#torch.eye(2).repeat(batch_size, 1, 1)
    reflection_matrices[:, 0, 0] = reflections[:, 0]
    reflection_matrices[:, 1, 1] = reflections[:, 1]
    # Combine transformations: scale -> shear -> rotate -> reflect
    transform_matrices = torch.bmm(torch.bmm(torch.bmm(scaling_matrices, shear_matrices), rotation_matrices), reflection_matrices)
    #transform_matrices = torch.bmm(torch.bmm(shear_matrices, torch.bmm(reflection_matrices, scaling_matrices)), rotation_matrices)
    # Assign to affine matrices
    affine_matrices[:, :2, :2] = transform_matrices
    affine_matrices[:, 0, 2] = translationsX
    affine_matrices[:, 1, 2] = translationsY
    return affine_matrices

def Generate_Affine_batch(batch_size):
    angle_range = [-math.pi, math.pi]
    scale_range = [0.2,1.8]
    shear_range = [-0.5,0.5]
    translation_range = [-0.25,0.25]
    angles = sample_from_2distributions(angle_range, N_samples= batch_size, PivotPoint = 0)
    scalesX = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
    scalesY = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
    shearsX = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
    shearsY = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
    translationsX = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
    translationsY = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
    reflections = 2*reflections_dist.sample((batch_size, 2))-1
    affine_matrices = Batch_matrice_from_parameters(angles,scalesX,scalesY,shearsX,shearsY,reflections,translationsX,translationsY)
    return affine_matrices


Intitial_Tx = ['angle', 'scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflection']
BLOCK_options = {'Full':(0,1), 'Hard70-100%':(0.7,1), 'Medium30-70':(0.3,0.7), 'Easy10-30%':(0.1,0.3), 'Easier5-10%':(0.05,0.1), 'Easiest0-5%':(0,0.05)}

def Generate_Affine_batch_selecTx(batch_size, Tx = Intitial_Tx, Uniscale=False):
    angle_range = [-math.pi, math.pi]
    scale_range = [0.2,1.8]
    shear_range = [-0.5,0.5]
    translation_range = [-0.25,0.25]
    if 'angle' in Tx:
        angles = sample_from_2distributions(angle_range, N_samples= batch_size, PivotPoint = 0)
    else:
        angles = torch.zeros(batch_size)
    if 'scaleX' in Tx:
        scalesX = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
    else:
        scalesX = torch.ones(batch_size)
    if 'scaleY' in Tx:
        if Uniscale:
            scalesY = scalesX
        else:
            scalesY = sample_from_2distributions(scale_range, N_samples= batch_size, PivotPoint = 1)
    else:
        scalesY = torch.ones(batch_size)
    if 'shearX' in Tx:
        shearsX = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
    else:
        shearsX = torch.zeros(batch_size)
    if 'shearY' in Tx:
        shearsY = sample_from_2distributions(shear_range, N_samples= batch_size, PivotPoint = 0)
    else:
        shearsY = torch.zeros(batch_size)
    if 'translationX' in Tx:
        translationsX = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
    else:
        translationsX = torch.zeros(batch_size)
    if 'translationY' in Tx:
        translationsY = sample_from_2distributions(translation_range, N_samples= batch_size, PivotPoint = 0)
    else:
        translationsY = torch.zeros(batch_size)
    if 'reflection' in Tx:
        reflections = 2*reflections_dist.sample((batch_size, 2))-1
    else:
        reflections = torch.ones(batch_size, 2)
    affine_matrices = Batch_matrice_from_parameters(angles,scalesX,scalesY,shearsX,shearsY,reflections,translationsX,translationsY)
    return affine_matrices

def sample_from_2distributions(dist_range, N_samples= batch_size, PivotPoint = 0):
    N_samplesFromDistribution1 = random.randint(0, N_samples)
    N_samplesFromDistribution2 = N_samples - N_samplesFromDistribution1
    r1_min, r1_max, r2_min, r2_max = find_sampling_boundries(dist_range, PivotPoint=PivotPoint)
    samples = torch.cat([ torch.FloatTensor(N_samplesFromDistribution1).uniform_(r1_min, r1_max),
                        torch.FloatTensor(N_samplesFromDistribution2).uniform_(r2_min, r2_max) ])
    return samples

def generate_registration_batches(source_origion, device='cpu', Affine_noise_intensity=0, IMG_noise_level=0, dim=128, Tx_select=False, Uniscale=False):
    batch_size = source_origion.shape[0]
    if Tx_select:
        Affine_batch = Generate_Affine_batch_selecTx(batch_size, Tx = Tx_select, Uniscale=Uniscale)
    else:
        Affine_batch = Generate_Affine_batch(batch_size)
    #
    data_device = source_origion.device.type
    if data_device != device:
        source_origion = source_origion.to(device)
    if Affine_noise_intensity>0:
        Affine_batch = add_noise2AffineMatrix(Affine_batch, sigma= Affine_noise_intensity)
    if device !='cpu':
        #source_origion = source_origion.to(device)
        Affine_batch = Affine_batch.to(device)
    warped_img_dim0 = warp_batch(Affine_batch, source_origion)
    target = center_crop(warped_img_dim0, dim = dim)
    if IMG_noise_level>0:
        source_origion = add_noise2batch(source_origion, sigma = IMG_noise_level, device=device)
        target = add_noise2batch(target, sigma = IMG_noise_level, device=device)
    source = center_crop(source_origion, dim = dim)
    X_batch = {'source':source,'target':target,'source_origion':source_origion}
    Y_batch = {'Affine_mtrx': Affine_batch}
    return X_batch, Y_batch



# Registration Dataset Preperation
#------------------------------------------------------------------------------------------------------------------------

#difficulty Initialization
dim = 128
dim0 =224
crop_ratio = dim/dim0
DIFFICULTY_MIN =0.0
DIFFICULTY_MAX = 1.0

if FLIP:
    prob_flip = 0.5*(DIFFICULTY_MAX+DIFFICULTY_MIN)
else:
    prob_flip = 0.0

reflections_dist = torch.distributions.Categorical(torch.tensor([prob_flip, 1-prob_flip]))

def find_sampling_boundries(dist_range, PivotPoint = 0,
            difficulty_min = DIFFICULTY_MIN, difficulty_max =DIFFICULTY_MAX):
    r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
    r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
    r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
    r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
    return r1_min, r1_max, r2_min, r2_max

dataiter = iter(valloader)
source_origion = next(dataiter)

X_batch, Y_batch = generate_registration_batches(source_origion, device=device_aug,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE)


#------------------------------------------------------------------------------------------------------------------------


#Modeling
#------------------------------------------------------------------------------------------------------------------------
class Build_IRmodel_ViT(torch.nn.Module):
    def __init__(self, core_model, registration_method = 'Rawblock', overlap='vertical', Arch = 'mobileVIT'):
        super(Build_IRmodel_ViT, self).__init__()
        self.core_model = core_model
        self.overlap = overlap
        self.registration_method = registration_method
        #self.N_parameters = 6
        if 'Recurence' in self.registration_method:
            self.M0_width = 4
            self.fc1 =nn.Linear(6, 64)
            if self.overlap == 'vertical':
                self.fc2 =nn.Linear(64, self.M0_width*128*6)
            else:
                self.fc2 =nn.Linear(64, self.M0_width*128*3)
    def forward(self, input_X_batch):
        source = input_X_batch['source']
        target = input_X_batch['target']
        if 'Recurence' in self.registration_method:
            M_i = input_X_batch['M_i'].view(-1, 6)
            M_rep = F.relu(self.fc1(M_i))
            if self.overlap == 'vertical':
                M_rep = F.relu(self.fc2(M_rep)).view(-1,6,self.M0_width,128)
                concatenated_input = torch.cat((source,target), dim=1)
                concatenated_input = torch.cat((concatenated_input,M_rep), dim=2)
            else:
                M_rep = F.relu(self.fc2(M_rep)).view(-1,3,self.M0_width,128)
                concatenated_input = torch.cat((source,target,M_rep), dim=2)
        else:
            if self.overlap == 'vertical':
                concatenated_input = torch.cat((source,target), dim=1)
            else:
                concatenated_input = torch.cat((source,target), dim=2)
        if 'mobileVIT' in Arch:
            core_model_output = self.core_model(concatenated_input).logits
        else:
            core_model_output = self.core_model(concatenated_input)
            
        if 'Recurence' in self.registration_method:
            predicted_part_mtrx = core_model_output.view(-1, 2, 3)
            if 'Additive' in self.registration_method:
                Prd_Affine_mtrx = predicted_part_mtrx + input_X_batch['M_i']
            predction = {'predicted_part_mtrx':predicted_part_mtrx,
                            'Affine_mtrx': Prd_Affine_mtrx}
        else:
            Prd_Affine_mtrx = core_model_output.view(-1, 2, 3)
            predction = {'Affine_mtrx': Prd_Affine_mtrx}
        return predction





def mtrx3(Affine_mtrx):
    mtrx_shape = Affine_mtrx.shape
    if len(mtrx_shape)==3:
        N_Mbatches = mtrx_shape[0]
        AM3 = torch.zeros( [N_Mbatches,3,3])
        if Affine_mtrx.device.type !='cpu':
            AM3 = AM3.to(device)
        AM3[:,0:2,:] = Affine_mtrx
        AM3[:,2,2] = 1
    elif len(mtrx_shape)==2:
        N_Mbatches = 1
        AM3 = torch.zeros([3,3])
        if Affine_mtrx.device.type !='cpu':
            AM3 = AM3.to(device)
        AM3[0:2,:] = Affine_mtrx
        AM3[2,2] = 1
    return AM3

MATRIX_THRESHOLD = 6.0
def inv_AM(Affine_mtrx, threshold=MATRIX_THRESHOLD):
    AM3 = mtrx3(Affine_mtrx)
    AM_inv = torch.linalg.inv(AM3)
    AM_inv = torch.clamp(AM_inv, min=-threshold, max=threshold)
    return AM_inv[:,0:2,:]

# freeze weights of the first model
if FREEZE_stage1:
    IR_Model_stage1.eval()
    for param in IR_Model_stage1.parameters():
        param.requires_grad = False


#------------------------------------------------------------------------------------------------------------------------
def threshold(numberX, max_threshold=2):
    if numberX>max_threshold:
        return max_threshold
    else:
        return numberX

def test_loss(model, loader, max_iterations=100, key = 'Affine_mtrx', Tx_select=Intitial_Tx, Uniscale=UNISCALE):
    eval_loss_tot = 0
    with torch.no_grad():
        for i, source_origion in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = generate_registration_batches(source_origion, device=device,
                                            Affine_noise_intensity=Affine_noise_intensity,
                                            IMG_noise_level=IMG_noise_level,
                                            dim=dim, Tx_select=Tx_select, Uniscale=Uniscale)
                #inputs = move_dict2device(inputs,device)
                #labels = move_dict2device(labels,device)
                predections = model(inputs)
                eval_loss_tot += MSE_loss(labels[key], predections[key].detach())
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg


from transformers import MobileViTV2ForImageClassification
core_model_regression = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
original_conv1 = core_model_regression.mobilevitv2.conv_stem.convolution
core_model_regression.classifier = torch.nn.Linear(512, 6)
core_model_regression.to(device)

class Build_IR_Model_regression(torch.nn.Module):
    def __init__(self, core_model_regression):
        super(Build_IR_Model_regression, self).__init__()
        self.core_model_regression = core_model_regression
    def forward(self, input_X_batch):
        source = input_X_batch['source']
        core_model_output = self.core_model_regression(source).logits
        Affine_mtrx = core_model_output.view(-1, 2, 3)
        outcome = {'Affine_mtrx':Affine_mtrx}
        return outcome

IR_Model_stage1 = Build_IR_Model_regression(core_model_regression)
IR_Model_stage1.to(device)


#-------------
#dataiter = iter(valloader)
#source_origion = next(dataiter)
#X_batch, Y_batch = generate_registration_batches(source_origion, device=device_aug,
#                                        Affine_noise_intensity=Affine_noise_intensity,
#                                        IMG_noise_level=IMG_noise_level,
#                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE)
X_batch = move_dict2device(X_batch,device)
pred = IR_Model_stage1(X_batch)




Learning_rate = 0.001
optimizer = optim.AdamW(IR_Model_stage1.parameters(), lr=Learning_rate)#, momentum=0.9
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

print_every = int(2000*80/batch_size)
MSE_loss = torch.nn.functional.mse_loss
if not SM1:
    IR_Model_stage1 = torch.compile(IR_Model_stage1)

#torch.set_float32_matmul_precision('high')
with_Difficultyadj_loss=False
with_scheduler=True
training_loss_iterationlist = []
validation_loss_iterationlist = []
TOTAL_Epochs = 200
best_loss = 100000000000000000000


#Single example
#------------------------------------------------------------------------------------------------------------------------

for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    Learning_rate = 0.00005 + 0.001*math.sin(math.pi*EPOCH/TOTAL_Epochs)
    optimizer = torch.optim.AdamW(IR_Model_stage1.parameters(), lr=Learning_rate)#, momentum=0.9
    if EPOCH==TOTAL_Epochs//2:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for source_origion in loop:
        i+=1
        inputs, labels = generate_registration_batches(source_origion, device=device,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE)
        #inputs = move_dict2device(inputs,device)
        #labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IR_Model_stage1(inputs)
        loss = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'])
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IR_Model_stage1, valloader,int(2000/batch_size),'Affine_mtrx',
                                        Tx_select=Intitial_Tx, Uniscale=UNISCALE).detach().item()
                scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    plt.ylim(0,0.5)
    plt.legend()
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")



#Double examples
#------------------------------------------------------------------------------------------------------------------------

for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    Learning_rate = 0.00005 + 0.005*math.sin(math.pi*EPOCH/TOTAL_Epochs)
    optimizer = torch.optim.AdamW(IR_Model_stage1.parameters(), lr=Learning_rate)#, momentum=0.9
    if EPOCH==TOTAL_Epochs//2:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for source_origion in loop:
        i+=1
        inputs, labels = generate_registration_batches(source_origion, device=device,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE)
        #inputs = move_dict2device(inputs,device)
        #labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IR_Model_stage1(inputs)
        loss1 = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'])
        inputs, labels = generate_registration_batches(source_origion, device=device,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=Intitial_Tx, Uniscale=UNISCALE)
        predections = IR_Model_stage1(inputs)
        loss2 = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'])
        loss = 0.5*(loss1 + loss2)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IR_Model_stage1, valloader,int(2000/batch_size),'Affine_mtrx',
                                        Tx_select=Intitial_Tx, Uniscale=UNISCALE).detach().item()
                scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    plt.legend()
    plt.ylim(0,0.5)
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")



torch.save(IR_Model_stage1.state_dict(), savingfolder+'./IR_Model_stage1_EndTraining.pth')
torch.save(core_model_regression.state_dict(), savingfolder+'./core_model_regression_EndTraining.pth')
'''