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

device_num = 1
torch.cuda.set_device(device_num)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_aug = device#'cpu'

#Saving folder
#------------------------------------------------------------------------------------------------------------------------

HPC = False
SM1 = False

folder_prefx ='Global_GravityReference_'

Affine_noise_intensity=0.0
IMG_noise_level=0.05
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
    batch_size = 128

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

def generate_registration_batches(source_origion, device='cpu', Affine_noise_intensity=0, IMG_noise_level=0, dim=128, Tx_select=False):
    batch_size = source_origion.shape[0]
    if Tx_select:
        Affine_batch = Generate_Affine_batch_selecTx(batch_size, Tx = Tx_select, Uniscale=False)
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
prob_flip = 0.5*(DIFFICULTY_MAX+DIFFICULTY_MIN)
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
                                        dim=dim, Tx_select=False)

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

overlap='vertical'
registration_method='Rawblock'
FREEZE_stage1 = False
if 'VIT' in Arch:
    if 'mobileVIT' in Arch:
        from transformers import MobileViTV2ForImageClassification
        core_model_stage1 = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        original_conv1 = core_model_stage1.mobilevitv2.conv_stem.convolution
        new_conv1 = torch.nn.Conv2d(6,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Initialize the new convolutional layer's weights
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            new_conv1.weight[:, 3:, :, :] = original_conv1.weight
        core_model_stage1.mobilevitv2.conv_stem.convolution = new_conv1
        core_model_stage1.classifier = torch.nn.Linear(512, 6)
    elif Arch == 'rawVIT':
        core_model_stage1 = rawViT(img_size=128,in_channels=6,patch_size=12,embed_dim=384,hidden_dim=768,num_heads=12,num_layers=4, num_classes=6)
    IR_Model_stage1 = Build_IRmodel_ViT(core_model_stage1,registration_method = registration_method, overlap=overlap, Arch = Arch)
    core_model_stage1.to(device)
    if FREEZE_stage1:
        IR_Model_stage1.load_state_dict(torch.load(file_loadingfolder+'IR_Model'+ext+'.pth'))
    IR_Model_stage1.to(device)


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

MATRIX_THRESHOLD = 5.0
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

class Build_IRmodel_wRef(torch.nn.Module):
    def __init__(self,IR_Model_stage1):
        super(Build_IRmodel_wRef, self).__init__()
        self.IR_Model_stage1 = IR_Model_stage1
        self.Global_reference = torch.nn.Parameter(torch.rand(3, dim, dim))
        #self.overlap = overlap
        #self.No_recurences = No_recurences
    def forward(self, input_X_batch):
        Global_reference_batches = self.Global_reference.unsqueeze(0)
        Global_reference_batches = Global_reference_batches.repeat([input_X_batch['source'].shape[0],1,1,1])
        source = input_X_batch['source']
        target = input_X_batch['target']
        predections_A = self.IR_Model_stage1(input_X_batch)
        pred_matrix_A = predections_A['Affine_mtrx']
        input_X_batch_B = {'source':target, 'target':Global_reference_batches}
        predections_B = self.IR_Model_stage1(input_X_batch_B)
        pred_matrix_B = predections_B['Affine_mtrx']
        input_X_batch_C = {'source':source, 'target':Global_reference_batches}
        predections_C = self.IR_Model_stage1(input_X_batch_C)
        pred_matrix_C = predections_C['Affine_mtrx']
        input_X_batch_D = {'source':Global_reference_batches, 'target':target}
        predections_D = self.IR_Model_stage1(input_X_batch_D)
        pred_matrix_D = predections_C['Affine_mtrx']
        outcome = {'Affine_mtrx':pred_matrix_A, 'pred_matrix_A':pred_matrix_A, 'pred_matrix_B':pred_matrix_B,
                     'pred_matrix_C':pred_matrix_C,'pred_matrix_D':pred_matrix_D, 'Global_reference':self.Global_reference}
        return outcome


IRmodel_wRef = Build_IRmodel_wRef(IR_Model_stage1)
IRmodel_wRef.to(device)
Global_reference0 = IRmodel_wRef.Global_reference
torchvision.transforms.ToPILImage()(Global_reference0).save(savingfolder+f'Global_reference_inital.png')

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of trainable paprameters: ',count_trainable_parameters(IRmodel_wRef)/1000000, 'Millions')
print('Number of trainable paprameters: ',count_trainable_parameters(IR_Model_stage1)/1000000, 'Millions')

pytorch_total_params = sum(p.numel() for p in IRmodel_wRef.parameters())
print('Number of paprameters: ',pytorch_total_params/1000000, 'Millions')

#------------------------------------------------------------------------------------------------------------------------
def threshold(numberX, max_threshold=2):
    if numberX>max_threshold:
        return max_threshold
    else:
        return numberX

def test_loss(model, loader, max_iterations=100, key = 'Affine_mtrx', 
            Affine_noise_intensity=Affine_noise_intensity, IMG_noise_level=IMG_noise_level,
            Tx_select=False):
    eval_loss_tot = 0
    with torch.no_grad():
        for i, source_origion in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = generate_registration_batches(source_origion, device=device,
                                            Affine_noise_intensity=Affine_noise_intensity,
                                            IMG_noise_level=IMG_noise_level,
                                            dim=dim, Tx_select=Tx_select)
                predections = model(inputs)
                eval_loss_tot += MSE_loss(labels[key], predections[key].detach())
                eval_loss_avg = eval_loss_tot/i
            else :
                return eval_loss_avg
    return eval_loss_avg




'''
I_avg = torch.zeros_like(source_origion[0])
L=0
for source_origion in loop:
    I_avg += torch.mean(source_origion, dim=0)
    L+=1

I_avg = I_avg/L
torchvision.transforms.ToPILImage()(I_avg).save(savingfolder+f'I_average_{EPOCH}.png')


ident = torch.tensor([[1., 0., 0.], [0, 1., 0]], dtype=torch.float32).to(device)
Trivial_B = torch.clamp(torch.mean(torch.abs(predections['pred_matrix_B'])-ident), min=0.4, max=100)
torch.mean(torch.abs(labels['Affine_mtrx'])-ident)
'''
#Training
#------------------------------------------------------------------------------------------------------------------------
if Arch == 'rawVIT':
    Learning_rate = 0.0001
else:
    Learning_rate = 0.001

print_every = int(2000*80/batch_size)
MSE_loss = torch.nn.functional.mse_loss

Curriculum_learning=False
if Curriculum_learning:
    optimizer = optim.SGD(IRmodel_wRef.parameters(), lr=Learning_rate)#, momentum=0.9
else:
    optimizer = optim.AdamW(IRmodel_wRef.parameters(), lr=Learning_rate)#, momentum=0.9

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

if not SM1:
    IRmodel_wRef = torch.compile(IRmodel_wRef)

#torch.set_float32_matmul_precision('high')

with_Difficultyadj_loss=False
Avoid_trivial= True
with_scheduler=True
training_loss_iterationlist = []
validation_loss_iterationlist = []
Loss_A_iterationlist = []
Loss_B_iterationlist = []
Loss_C_iterationlist = []
Loss_D_iterationlist = []
trivial_B_iterationlist = []
trivial_C_iterationlist = []
trivial_D_iterationlist = []
TOTAL_Epochs = 24
best_loss = 100000000000000000000

'''
for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    loss_prnt_A= 0.0
    loss_prnt_C= 0.0
    loss_prnt_trivial_B =0.0
    loss_prnt_trivial_C =0.0
    if Curriculum_learning:
        Difficulty_factor = min(1, (EPOCH+1)/TOTAL_Epochs)
        #Difficulty_factor = min(1, np.log(0.5*EPOCH+1)/np.log(0.5*TOTAL_Epochs-7))
        prob_flip = 0.5*(0+Difficulty_factor)
        reflections_dist = torch.distributions.Categorical(torch.tensor([prob_flip, 1-prob_flip]))
        def find_sampling_boundries(dist_range, PivotPoint = 0,
            difficulty_min = 0, difficulty_max = Difficulty_factor):
            r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
            r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
            r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
            r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
            return r1_min, r1_max, r2_min, r2_max
    if with_Difficultyadj_loss:
        weight_loss = Difficulty_factor
    else:
        weight_loss = 0.2
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for source_origion in loop:
        i+=1
        inputs, labels = generate_registration_batches(source_origion, device=device,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=False)
        #inputs = move_dict2device(inputs,device)
        #labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IRmodel_wRef(inputs)
        loss_A = MSE_loss(labels['Affine_mtrx'], predections['pred_matrix_A'])
        Transient_matrix_C = torch.matmul(mtrx3(labels['Affine_mtrx']), mtrx3(predections['pred_matrix_B']))[:,0:2,:]
        #Transient_matrix_C = torch.matmul(mtrx3(predections['pred_matrix_B']), mtrx3(labels['Affine_mtrx']))[:,0:2,:]
        #Transient_matrix_A = torch.matmul(mtrx3(predections['pred_matrix_A']), mtrx3(predections['pred_matrix_B']))[:,0:2,:]
        loss_C = MSE_loss(Transient_matrix_C, predections['pred_matrix_C'])
        loss = (1-weight_loss)*loss_A + weight_loss*loss_C
        #to avoid a trivial solution (when B = zero, loss_C becomes 0)
        if Avoid_trivial:
            Trivial_B = -torch.clamp(torch.mean(torch.abs(predections['pred_matrix_B'][:,:,:2])), min=0, max=0.35) + torch.clamp(
                        torch.mean(torch.abs(predections['pred_matrix_B'][:,:,2])), min=0.35, max=1948)
            Trivial_C = -torch.clamp(torch.mean(torch.abs(predections['pred_matrix_C'][:,:,:2])), min=0, max=0.35) + torch.clamp(
                        torch.mean(torch.abs(predections['pred_matrix_C'][:,:,2])), min=0.35, max=1948)
            loss += (Trivial_B+Trivial_C)
            loss_prnt_trivial_B += Trivial_B.detach().item()
            loss_prnt_trivial_C += Trivial_C.detach().item()
        loss_prnt_A += loss_A.detach().item()
        loss_prnt_C += loss_C.detach().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IRmodel_wRef, valloader,int(2000/batch_size),'Affine_mtrx').detach().item()
                scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                Loss_A_iterationlist.append(threshold(loss_prnt_A/print_every))
                Loss_C_iterationlist.append(threshold(loss_prnt_C/print_every))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '
                printed_text += f'loss_A: {loss_prnt_A/print_every:.3f}, loss_C:{loss_prnt_C/print_every:.3f}, '
                if Avoid_trivial:
                    trivial_B_iterationlist.append(threshold(loss_prnt_trivial_B/print_every))
                    trivial_C_iterationlist.append(threshold(loss_prnt_trivial_C/print_every))
                    printed_text += f'loss_TrivialB: {loss_prnt_trivial_B/print_every:.3f}, loss_TrivialC: {loss_prnt_trivial_C/print_every:.3f}, '
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
                loss_prnt_A= 0.0
                loss_prnt_C= 0.0
                loss_prnt_trivial_B=0.0
                loss_prnt_trivial_C=0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    plt.plot(Loss_A_iterationlist, label = 'Loss A')
    plt.plot(Loss_C_iterationlist, label = 'Loss C-estimated')
    if Avoid_trivial:
        plt.plot(trivial_B_iterationlist, label = 'trivial B')
        plt.plot(trivial_C_iterationlist, label = 'trivial C')
        np.savetxt(savingfolder+'trivial_B_iterationlist.txt', trivial_B_iterationlist, delimiter=",", fmt="%.3f")
        np.savetxt(savingfolder+'trivial_C_iterationlist.txt', trivial_C_iterationlist, delimiter=",", fmt="%.3f")
    plt.legend()
    plt.ylim(0,0.5)
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'Loss_A_iterationlist.txt', Loss_A_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'Loss_C-estimated_iterationlist.txt', Loss_C_iterationlist, delimiter=",", fmt="%.3f")
    torchvision.transforms.ToPILImage()(IRmodel_wRef.Global_reference).save(savingfolder+f'Global_reference_{EPOCH}.png')
'''

#Version 3: Gravity 
for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    loss_prnt_A= 0.0
    loss_prnt_B= 0.0
    loss_prnt_C= 0.0
    loss_prnt_D= 0.0
    loss_prnt_trivial_B =0.0
    loss_prnt_trivial_C =0.0
    Identity = torch.zeros(batch_size,2,3).to(device)
    Identity[:,0,0] = 1.
    Identity[:,1,1] = 1.
    if Curriculum_learning:
        Difficulty_factor = min(1, (EPOCH+1)/TOTAL_Epochs)
        #Difficulty_factor = min(1, np.log(0.5*EPOCH+1)/np.log(0.5*TOTAL_Epochs-7))
        prob_flip = 0.5*(0+Difficulty_factor)
        reflections_dist = torch.distributions.Categorical(torch.tensor([prob_flip, 1-prob_flip]))
        def find_sampling_boundries(dist_range, PivotPoint = 0,
            difficulty_min = 0, difficulty_max = Difficulty_factor):
            r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
            r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
            r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
            r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
            return r1_min, r1_max, r2_min, r2_max
    if with_Difficultyadj_loss:
        weight_loss = Difficulty_factor
    else:
        weight_loss = 0.2
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for source_origion in loop:
        i+=1
        inputs, labels = generate_registration_batches(source_origion, device=device,
                                        Affine_noise_intensity=Affine_noise_intensity,
                                        IMG_noise_level=IMG_noise_level,
                                        dim=dim, Tx_select=False)
        #inputs = move_dict2device(inputs,device)
        #labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IRmodel_wRef(inputs)
        loss_A = MSE_loss(labels['Affine_mtrx'], predections['pred_matrix_A'])
        loss_B = MSE_loss(inv_AM(labels['Affine_mtrx']), predections['pred_matrix_B'])
        loss_C = MSE_loss(Identity, predections['pred_matrix_C'])
        loss_D = MSE_loss(labels['Affine_mtrx'], predections['pred_matrix_D'])
        #Transient_matrix_C = torch.matmul(mtrx3(labels['Affine_mtrx']), mtrx3(predections['pred_matrix_B']))[:,0:2,:]
        #Transient_matrix_C = torch.matmul(mtrx3(predections['pred_matrix_B']), mtrx3(labels['Affine_mtrx']))[:,0:2,:]
        #Transient_matrix_A = torch.matmul(mtrx3(predections['pred_matrix_A']), mtrx3(predections['pred_matrix_B']))[:,0:2,:]
        #loss_C = MSE_loss(Transient_matrix_C, predections['pred_matrix_C'])
        loss = (1-weight_loss)*loss_A + weight_loss*(loss_B+loss_C)
        #to avoid a trivial solution (when B = zero, loss_C becomes 0)
        if Avoid_trivial:
            Trivial_B = -torch.clamp(torch.mean(torch.abs(predections['pred_matrix_B'][:,:,:2])), min=0, max=0.35) + torch.clamp(
                        torch.mean(torch.abs(predections['pred_matrix_B'][:,:,2])), min=0.35, max=1948)
            #Trivial_C = -torch.clamp(torch.mean(torch.abs(predections['pred_matrix_C'][:,:,:2])), min=0, max=0.35) + torch.clamp(
            #            torch.mean(torch.abs(predections['pred_matrix_C'][:,:,2])), min=0.35, max=1948)
            loss += Trivial_B #+Trivial_C)
            loss_prnt_trivial_B += Trivial_B.detach().item()
            #loss_prnt_trivial_C += Trivial_C.detach().item()
        loss_prnt_A += loss_A.detach().item()
        loss_prnt_B += loss_B.detach().item()
        loss_prnt_C += loss_C.detach().item()
        loss_prnt_D += loss_D.detach().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IRmodel_wRef, valloader,int(2000/batch_size),'Affine_mtrx').detach().item()
                scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                Loss_A_iterationlist.append(threshold(loss_prnt_A/print_every))
                Loss_B_iterationlist.append(threshold(loss_prnt_B/print_every))
                Loss_C_iterationlist.append(threshold(loss_prnt_C/print_every))
                Loss_D_iterationlist.append(threshold(loss_prnt_D/print_every))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '
                printed_text += f'loss_A: {loss_prnt_A/print_every:.3f}, '
                printed_text += f'loss_B: {loss_prnt_B/print_every:.3f}, '
                printed_text += f'loss_C: {loss_prnt_C/print_every:.3f}, '
                printed_text += f'loss_D: {loss_prnt_D/print_every:.3f}, '
                if Avoid_trivial:
                    trivial_B_iterationlist.append(threshold(loss_prnt_trivial_B/print_every))
                    printed_text += f'loss_TrivialB: {loss_prnt_trivial_B/print_every:.3f},'
                    #trivial_C_iterationlist.append(threshold(loss_prnt_trivial_C/print_every))
                    #printed_text += f'loss_TrivialC: {loss_prnt_trivial_C/print_every:.3f}, '
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
                loss_prnt_A= 0.0
                loss_prnt_B= 0.0
                loss_prnt_C= 0.0
                loss_prnt_D= 0.0
                loss_prnt_trivial_B=0.0
                loss_prnt_trivial_C=0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    plt.plot(Loss_A_iterationlist, label = 'Loss A')
    plt.plot(Loss_B_iterationlist, label = 'Loss B')
    plt.plot(Loss_C_iterationlist, label = 'Loss C')
    plt.plot(Loss_D_iterationlist, label = 'Loss D')
    if Avoid_trivial:
        plt.plot(trivial_B_iterationlist, label = 'trivial B')
        #plt.plot(trivial_C_iterationlist, label = 'trivial C')
        np.savetxt(savingfolder+'trivial_B_iterationlist.txt', trivial_B_iterationlist, delimiter=",", fmt="%.3f")
        #np.savetxt(savingfolder+'trivial_C_iterationlist.txt', trivial_C_iterationlist, delimiter=",", fmt="%.3f")
    plt.legend()
    plt.ylim(0,0.5)
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'Loss_A_iterationlist.txt', Loss_A_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'Loss_B_iterationlist.txt', Loss_B_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'Loss_C_iterationlist.txt', Loss_C_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'Loss_D_iterationlist.txt', Loss_D_iterationlist, delimiter=",", fmt="%.3f")
    torchvision.transforms.ToPILImage()(IRmodel_wRef.Global_reference).save(savingfolder+f'Global_reference_{EPOCH}.png')

IRmodel_wRef.Global_reference




IRmodel_wRef.Global_reference


print('Finished Training')
    #return IR_Model, training_loss_iterationlist, validation_loss_iterationlist

#IR_Model_tst, training_loss_iterationlist, validation_loss_iterationlist = train_IR_model(IRmodel_finetuned, trainloader, TOTAL_Epochs = 12,)
torchvision.transforms.ToPILImage()(IRmodel_wRef.Global_reference)).save(savingfolder+'Global_reference.png')

torch.save(IRmodel_wRef.state_dict(), savingfolder+'IRmodel_wRef.pth')
torch.save(IR_Model_stage1.state_dict(), savingfolder+'./IR_Model_stage1_EndTraining.pth')
torch.save(core_model_stage1.state_dict(), savingfolder+'./core_model_stage1_EndTraining.pth')

with open(savingfolder+'training_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(training_loss_iterationlist))

with open(savingfolder+'validation_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(validation_loss_iterationlist))


plt.plot(training_loss_iterationlist, label = 'training loss')
plt.plot(validation_loss_iterationlist, label = 'validation loss')
plt.ylim(0,0.5)
plt.legend()
plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
plt.close()




# Testing

os.system('mkdir '+ savingfolder+ 'MIRexamples')

test_set = Dataset_source(list_paths=routes_source['test'], batch_size=batch_size)
testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, pin_memory=True,shuffle=True, num_workers=4, )

Rep_Test_loss = test_loss(IR_Model_stage1, testloader, max_iterations=100, key = 'Affine_mtrx',
            Affine_noise_intensity=Affine_noise_intensity, IMG_noise_level=IMG_noise_level,
            Tx_select=False)

json.dump( Rep_Test_loss, open( savingfolder+'Rep_Test_loss.txt', 'w' ) )

Rep_Test_loss0 = test_loss(IR_Model_stage1, testloader, max_iterations=100, key = 'Affine_mtrx',
            Affine_noise_intensity=0, IMG_noise_level=0,
            Tx_select=False)

json.dump( Rep_Test_loss, open( savingfolder+'Rep_Test_loss0.txt', 'w' ) )



AM_recurrent_loss(IR_Model_stage2.eval(),testloader0 , max_iterations=100, No_recurences = 7, key ='Affine_mtrx', plot=True, plot_batach = 5, prefix = 'randomdifficulty')

test_loss(model, loader, max_iterations=100, key = 'Affine_mtrx', 
            Affine_noise_intensity=Affine_noise_intensity, IMG_noise_level=IMG_noise_level,
            Tx_select=False)




Measures_list = Intitial_Tx#['angle', 'scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflectionX', 'reflectionY']
BLOCK_options = {'Full':(0,1), 'Hard70-100%':(0.7,1), 'Medium30-70':(0.3,0.7), 'Easy10-30%':(0.1,0.3), 'Easier5-10%':(0.05,0.1), 'Easiest0-5%':(0,0.05)}

# test various difficulties
MSE_AffineMatrix_recurrent_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    prob_flip = 0.5*(Block_min_difficulty+Block_max_difficulty)
    reflections_dist = torch.distributions.Categorical(torch.tensor([prob_flip, 1-prob_flip]))
    def find_sampling_boundries(dist_range, PivotPoint = 0,
                difficulty_min = Block_min_difficulty, difficulty_max =Block_max_difficulty):
        r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
        r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
        r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
        r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
        return r1_min, r1_max, r2_min, r2_max
        MSE_AffineMatrix_recurrent_difficulty[Difficulty] =test_loss(
                IR_Model_stage1.eval(), testloader, max_iterations=70, key = 'Affine_mtrx',
                Affine_noise_intensity=0, IMG_noise_level=0,  Tx_select=Measures_list)

print(MSE_AffineMatrix_recurrent_difficulty)
json.dump(MSE_AffineMatrix_recurrent_difficulty, open( savingfolder+'MSE_AffineMatrix_recurrent_difficulty.txt', 'w' ) )


def plot_examples(source,target,wrapped_img,AM_Predicted=0,AM_Previous=0, AM_GroundTruth=0,
                    n_examples = 6, iteration = 0, prefix= 'randomd' ,
                    plt_markrs=True,plt_imgs=True, shadow=True):
    #prepare figures
    if shadow:
        plots_per_example = 4
    else:
        plots_per_example = 3
    figure = plt.figure()
    gs = figure.add_gridspec(n_examples, plots_per_example, hspace=0.05, wspace=0)
    axis = gs.subplots(sharex='col', sharey='row')
    figure.set_figheight(5*n_examples)
    figure.set_figwidth(5*plots_per_example)
    with torch.no_grad():
        if shadow:
            Error_map = torch.abs(target.detach()-wrapped_img.detach())
        for k in range(n_examples):
            if plt_markrs:
                M_GroundTruth = workaround_matrix(AM_GroundTruth.detach(), acc = 0.5/crop_ratio)
                M_Predicted = workaround_matrix(AM_Predicted.detach(), acc = 0.5/crop_ratio)
                M_Previous = workaround_matrix(AM_Previous.detach(), acc = 0.5/crop_ratio)
                x0, y0 = generate_standard_mark()
                x0_target, y0_target = transform_standard_points(M_GroundTruth[k], x0, y0)
                x0_source, y0_source = transform_standard_points(M_Previous[k], x0, y0)
                x0_transformed, y0_transformed = transform_standard_points(M_Predicted[k], x0_source, y0_source)
                #Another way of finding the transformed marker
                #M_accumilative = torch.matmul(mtrx3(M_Predicted), mtrx3(M_Previous))[:,0:2,:]
                #x0_transformed, y0_transformed = transform_standard_points(M_accumilative[k], x0, y0)
                #--------------destandarize for plotting purposes-------------
                x_source = destandarize_point(x0_source, dim=dim, flip = False)
                x_target = destandarize_point(x0_target, dim=dim, flip = False)
                x_transformed = destandarize_point(x0_transformed, dim=dim, flip = False)
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
                axis[k, 3].imshow(torchvision.transforms.ToPILImage()(Error_map[k]),alpha=0.9)
        for ax in figure.get_axes():
            ax.label_outer()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    suffix = ''
    if plt_markrs:
            suffix += 'Elipse'
    if plt_imgs:
        suffix += 'Img'
    plt.savefig(savingfolder+'MIRexamples/{}_{}_{}.jpeg'.format(prefix, suffix,iteration), bbox_inches='tight')
    plt.close()






def plot_global_represetations(model,loader , max_iterations=2, No_recurences = 1,
                        plot=True, plot_batach = 1, prefix = 'debug', dim=dim, key ='Affine_mtrx',
                        Affine_noise_intensity=0, IMG_noise_level=0, Tx_select=False):
    model = model.eval()
    with torch.no_grad():
        Global_reference_batches = IRmodel_wRef.Global_reference.detach().unsqueeze(0)
        for j in range(No_recurences):
            for i, source_origion in enumerate(loader, 0):
                if i < max_iterations:
                    inputs, labels = generate_registration_batches(source_origion, device=device,
                                                Affine_noise_intensity=Affine_noise_intensity,
                                                IMG_noise_level=IMG_noise_level,
                                                dim=dim, Tx_select=Tx_select)
                    source0 = inputs['target']#.to(device)
                    target = Global_reference_batches.repeat([source0.shape[0],1,1,1])
                    #target = inputs['source']#.to(device)
                    source_origion = torch.nn.ZeroPad2d(int((224-dim)/2))(source0)
                    #inputs['source_origion']#.to(device)
                    predections = model(inputs)
                    inputs_j = inputs
                    Affine_mtrx_j = predections['Affine_mtrx'].detach()
                    Accumilative_Affine_matrix = torch.zeros_like(Affine_mtrx_j).detach()
                    Accumilative_Affine_matrix[:,0,0]=1.
                    Accumilative_Affine_matrix[:,1,1]=1.
                    New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix.detach()), mtrx3(Affine_mtrx_j.detach()))[:,0:2,:]
                    grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion.shape,align_corners=False)
                    source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
                    wrapped_img = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
                    if plot:
                        if i== plot_batach:
                            plot_examples(source0,target,wrapped_img, AM_Predicted=Affine_mtrx_j,AM_Previous=Accumilative_Affine_matrix,
                                    AM_GroundTruth=labels[key], n_examples = 6, iteration = 0, plt_markrs=False, prefix= prefix)
                    for j in range(1, No_recurences):
                        inputs_j ={'source': wrapped_img,
                                    'source_origion': source_origion,
                                    'target': target,
                                    'M_i': torch.zeros([int(inputs['source'].shape[0]),2,3]).to(device)
                                    }
                        predections = model(inputs_j)
                        Affine_mtrx_j = predections['Affine_mtrx'].detach()
                        Accumilative_Affine_matrix = New_Accumilative_Affine_matrix.detach()
                        New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix.detach()), mtrx3(Affine_mtrx_j.detach()))[:,0:2,:]
                        grd_acc = torch.nn.functional.affine_grid(New_Accumilative_Affine_matrix, size=source_origion.shape,align_corners=False)
                        source_origion_224j_acc = torch.nn.functional.grid_sample(source_origion.detach(), grid=grd_acc,
                                                mode='bilinear', padding_mode='zeros', align_corners=False)
                        wrapped_img_acc = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j_acc)
                        wrapped_img = wrapped_img_acc.detach()
                        if plot:
                            if i== plot_batach:
                                plot_examples(inputs_j['source'],target,wrapped_img,AM_Predicted=Affine_mtrx_j, AM_Previous=Accumilative_Affine_matrix,
                                    AM_GroundTruth=labels[key], n_examples = 4, iteration = j, plt_markrs=False, prefix= prefix)
                else:
                    return 'Done'





plot_global_represetations(IR_Model_stage1,testloader , max_iterations=2, No_recurences = 1,
                        plot=True, plot_batach = 1, prefix = 'debug', dim=dim, 
                        Affine_noise_intensity=0, IMG_noise_level=0, Tx_select=False)




def AM_recurrent_loss(model,loader , max_iterations=2, No_recurences = 3,
                        key ='Affine_mtrx', plot=False, plot_batach = 1, prefix = 'debug', dim=dim, 
                        Affine_noise_intensity=0,
                        IMG_noise_level=0, Tx_select=False):
    AM_MSE_tot = {}
    AM_MSE_avg = {}
    model = model.eval()
    for j in range(No_recurences):
        AM_MSE_tot[str(j)]=0
    with torch.no_grad():
        for i, source_origion in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = generate_registration_batches(source_origion, device=device,
                                            Affine_noise_intensity=Affine_noise_intensity,
                                            IMG_noise_level=IMG_noise_level,
                                            dim=dim, Tx_select=Tx_select)
                #inputs = move_dict2device(inputs,device)
                #labels = move_dict2device(labels,device)
                source0 = inputs['source']#.to(device)
                target = inputs['target']#.to(device)
                source_origion = inputs['source_origion']#.to(device)
                predections = model(inputs)
                inputs_j = inputs
                Affine_mtrx_j = predections['Affine_mtrx'].detach()
                Accumilative_Affine_matrix = torch.zeros_like(Affine_mtrx_j).detach()
                Accumilative_Affine_matrix[:,0,0]=1.
                Accumilative_Affine_matrix[:,1,1]=1.
                New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix.detach()), mtrx3(Affine_mtrx_j.detach()))[:,0:2,:]
                AM_MSE_tot[str(0)] += MSE_loss(labels[key].detach(), New_Accumilative_Affine_matrix.detach()).item()
                grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion.shape,align_corners=False)
                source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                wrapped_img = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
                if plot:
                    if i== plot_batach:
                        plot_examples(source0,target,wrapped_img, AM_Predicted=Affine_mtrx_j,AM_Previous=Accumilative_Affine_matrix, AM_GroundTruth=labels[key], n_examples = 6, iteration = 0, plt_markrs=True, prefix= prefix)
                for j in range(1, No_recurences):
                    inputs_j ={'source': wrapped_img,
                                'source_origion': source_origion,
                                'target': target,
                                'M_i': torch.zeros([int(inputs['source'].shape[0]),2,3]).to(device)
                                }
                    predections = model(inputs_j)
                    Affine_mtrx_j = predections['Affine_mtrx'].detach()
                    Accumilative_Affine_matrix = New_Accumilative_Affine_matrix.detach()
                    New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix.detach()), mtrx3(Affine_mtrx_j.detach()))[:,0:2,:]
                    grd_acc = torch.nn.functional.affine_grid(New_Accumilative_Affine_matrix, size=source_origion.shape,align_corners=False)
                    source_origion_224j_acc = torch.nn.functional.grid_sample(source_origion.detach(), grid=grd_acc,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
                    wrapped_img_acc = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j_acc)
                    wrapped_img = wrapped_img_acc.detach()
                    if plot:
                        if i== plot_batach:
                            plot_examples(inputs_j['source'],target,wrapped_img,AM_Predicted=Affine_mtrx_j, AM_Previous=Accumilative_Affine_matrix, AM_GroundTruth=labels[key], n_examples = 4, iteration = j, plt_markrs=True, prefix= prefix)
                    AM_MSE_tot[str(j)] += MSE_loss(labels[key].detach(), New_Accumilative_Affine_matrix.detach()).item()
            else:
                for r in range(No_recurences):
                    AM_MSE_avg[str(r)] = np.round(AM_MSE_tot[str(r)]/max_iterations, 4)
                return AM_MSE_avg


AM_recurrent_loss(IR_Model_stage1,testloader , max_iterations=70, No_recurences = 4,
                        key ='Affine_mtrx', plot=False, plot_batach = 1, prefix = 'random',
                        Affine_noise_intensity=Affine_noise_intensity,
                        IMG_noise_level=IMG_noise_level, dim=dim, Tx_select=Measures_list)


AM_recurrent_loss(IR_Model_stage1,testloader , max_iterations=70, No_recurences = 4,
                        key ='Affine_mtrx', plot=False, plot_batach = 1, prefix = 'random',
                        Affine_noise_intensity=0,
                        IMG_noise_level=0, dim=dim, Tx_select=Measures_list)



MSE_AffineMatrix_recurrent_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    prob_flip = 0.5*(Block_min_difficulty+Block_max_difficulty)
    reflections_dist = torch.distributions.Categorical(torch.tensor([prob_flip, 1-prob_flip]))
    def find_sampling_boundries(dist_range, PivotPoint = 0,
                difficulty_min = Block_min_difficulty, difficulty_max =Block_max_difficulty):
        r1_min = PivotPoint - difficulty_max*(abs(PivotPoint-dist_range[0]))
        r1_max = PivotPoint - difficulty_min*(abs(PivotPoint-dist_range[0]))
        r2_min = PivotPoint + difficulty_min*(abs(dist_range[1]-PivotPoint))
        r2_max = PivotPoint + difficulty_max*(abs(dist_range[1]-PivotPoint))
        return r1_min, r1_max, r2_min, r2_max
    MSE_AffineMatrix_recurrent_difficulty[Difficulty] = AM_recurrent_loss(IR_Model_stage1,testloader,
                        max_iterations=70, No_recurences = 3,
                        key ='Affine_mtrx', plot=False, plot_batach = 1, prefix = Difficulty,
                        Affine_noise_intensity=0,
                        IMG_noise_level=0, dim=dim,
                        Tx_select=False)
 
print(MSE_AffineMatrix_recurrent_difficulty)
json.dump(MSE_AffineMatrix_recurrent_difficulty, open( savingfolder+'MSE_AffineMatrix_recurrent_difficulty.txt', 'w' ) )






#--------------------Plot-----------------------------------


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




class Torch_NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None,stride = 1):
        self.win = win
        self.eps = 1e-2
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
        cross[torch.abs(cross<self.eps)] = self.eps
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var[torch.abs(I_var<self.eps)] = self.eps
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var[torch.abs(J_var<self.eps)] = self.eps
        cc = ((cross * cross )+ self.eps) / ((I_var * J_var) + self.eps)
        return cc
    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        cc = torch.mean(cc)
        return -cc

NCC_loss = Torch_NCC(win=128,stride = 128).loss

def save_examples(model, loader , n_examples = 4, plt_markrs=True,plt_imgs=False, time=1,
                    Affine_noise_intensity=0, IMG_noise_level=0, dim=128, Tx_select= False,
                    feed_origion=True, shadow=False, win = 5, Fix_Torch_Wrap=False):
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
                source_origion = next(dataiter)
                X_batch, Y_batch =  generate_registration_batches(source_origion, device=device,
                                            Affine_noise_intensity=Affine_noise_intensity,
                                            IMG_noise_level=IMG_noise_level,
                                            dim=dim, Tx_select=Tx_select)
            #X_batch = move_dict2device(X_batch,device)
            #Y_batch = move_dict2device(Y_batch,device)
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
                colored_shadow = torch.zeros_like(wrapped_img)
                Green = shadow_img.clone()
                Red = 1-Green
                colored_shadow[:,0:1,:,:] = Red
                colored_shadow[:,1:2,:,:] = Green
                Error_map = torch.abs(target.detach()-wrapped_img.detach())
            if plt_markrs:
                x0_source, y0_source = generate_standard_mark()#N_samples = 50, a= 2,b = 1, start=-0.5)
                x0_target, y0_target = transform_standard_points(M_GroundTruth[k], x0_source, y0_source)
                x0_transformed, y0_transformed = transform_standard_points(M_Predicted[k], x0_source, y0_source)
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
                axis[k, 4].imshow(torchvision.transforms.ToPILImage()(Error_map[k]),alpha=0.9)
        for ax in figure.get_axes():
            ax.label_outer()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        suffix = ''
        if plt_markrs:
                suffix += 'Elipse'
        if plt_imgs:
                suffix += 'Img'
        plt.savefig(savingfolder+'MIRexamples/{}_{}.jpeg'.format(suffix,time), bbox_inches='tight')
        plt.close()
#-------------------------------------------------------


save_examples(IR_Model_stage1, testloader , n_examples = 6, plt_markrs=True,plt_imgs=True, time=4,
                    Affine_noise_intensity=0, IMG_noise_level=0, dim=dim, Tx_select= Measures_list,
                    feed_origion=True, shadow=True, win = 9)



























































































































dim = 128
dim0 =224
crop_ratio = dim/dim0



Fine_tuned = False
with_Global_reference= True
Curriculum_learning = False
Avoid_trivial = True
with_global_loss = True
with_Cyclic_losses = False
Active_learning = False
with_Difficultyadj_loss = False
include_loss_model1 = False
Arch = 'mobileVIT'#{#'mobileVIT'#'rawVIT'#'ResNet''VTX', 'U-Net','ResNet', 'DINO' ,'rawVIT' }#'mobileVITwithMI'
overlap='vertical' #'vertical' #'horizontal'
registration_method = 'Rawblock'#'Additive_Recurence' #{'Additive_Recurence','Rawblock', 'matching_points', 'Additive_Recurence', 'Multiplicative_Recurence'} #'recurrent_matrix',
Uniscale = False
with_scheduler = True
IMG_noise = True
SWITCH = False

global Noise_level_dataset
Noise_level_dataset=0.1
LYR_NORM = False
Fix_Torch_Wrap = False
BW_Position = False
#BLOCKED_REGION = 0
BLOCK_MIN = 0
BLOCK_MAX = 1


if Fine_tuned:
    folder_prefx = 'Tr_FT2_'
    if with_Cyclic_losses:
        folder_prefx += 'cyclic_'
    if with_global_loss:
        folder_prefx += 'global_'
elif with_Global_reference:
    folder_prefx = 'GR_'
    if Avoid_trivial:
        folder_prefx += 'NoTrivial_'
else:
    folder_prefx = 'tst'#'12Epochs'

if Curriculum_learning:
    folder_prefx += 'CurriculumL_'

if Active_learning:
    folder_prefx += 'ActiveL_'

if Uniscale:
    Intitial_Tx = ['angle', 'scale','scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflectionX', 'reflectionY']
else:
    Intitial_Tx = ['angle', 'scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflectionX', 'reflectionY'] #'scale'

DATASET_generation_split = { 'train': 'active',
'val': 'active',
'test': 'active',
'test0':'active'}

DATASET_generation = DATASET_generation_split['train']#'active' #{'passive', #'active'}

imposed_point = 0
Glopooling = 'avg'#'max', 'avg', 'none'

if Fix_Torch_Wrap:
    Noise_level_dataset = 0.1
elif SWITCH:
    Noise_level_dataset = 0.1

if SWITCH:
    folder_prefx += '_bidirectional_'

if 'scale' in Intitial_Tx:
    folder_prefx += '_uniscale_'

if BW_Position:
    folder_prefx += '.BWPosition'

if IMG_noise:
    IMG_noise_level = 0.2
    batch_size = 128
    folder_prefx += 'ImgNoise{}_'.format(IMG_noise_level)
else:
    batch_size = 128

if 'Recurence' in registration_method :
    folder_prefx += '.completely_random'#'random_aff_param'#'completely_random' #'scheduledrandom' #zero

if registration_method == 'matching_points':
    with_epch_loss = True
    if imposed_point>0:
        folder_prefx += str(imposed_point)
else:
    with_epch_loss = False

if Fix_Torch_Wrap:
    folder_prefx += 'AdustMtrx_'
else:
    folder_prefx += 'AdjustPlot_'

folder_prefx+= Glopooling+'Pool_'

if LYR_NORM:
    folder_prefx += 'LyrNORM_'

if with_scheduler:
    folder_prefx += 'LR_'

if overlap=='vertical':
    folder_prefx += 'V.concat_'
else:
    folder_prefx += 'H.concat_'

if registration_method=='matching_points':
    N_parameters = 18
else:
    N_parameters = 6

if HPC:
    activedata_root = '/gpfs/projects/acad/maiaone/dataset/224/'
else:
    if SM1:
        activedata_root ='../../localdb/224/'
    else:
        activedata_root = '/home/ahmadh/imagenet-object-localization-challenge/ILSVRC/Data/Standard_Size/224/'
    import tqdm

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


if HPC:
    savingfolder = '/gpfs/home/acad/umons-artint/freepal/MIR_savedmodel/{}_Mi0:{}_{}_{}_{}/'.format(
                registration_method,folder_prefx, Arch, DATASET_generation,Noise_level_dataset)
else:
    savingfolder = '/home/ahmadh/MIR_savedmodel/{}_Mi0:{}_{}_{}_{}/'.format(
            registration_method,folder_prefx, Arch, DATASET_generation,Noise_level_dataset)

#savingfolder = savingfolder[:-1] +'shuffle_trainloader/'
os.system('mkdir '+ savingfolder)
#os.system('mkdir '+ savingfolder+ 'MIRexamples')

##-----------------------------------------
##---------------Gthms--------------------

# we need to apply the augmentation to a batch of images instead of applying it to each image alone
# Thus the augmentation will take place in the dataset

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def wrap_imge_with_inverse(Affine_mtrx, source_img, ratio= 2):
    #edited version on 20240301
    Affine_mtrx = workaround_matrix(Affine_mtrx, acc = ratio).to(device_aug)
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_img.shape,align_corners=False)
    wrapped_img = torch.nn.functional.grid_sample(source_img, grid=grd,
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)
    return wrapped_img

def wrap_imge0(Affine_mtrx, source_img, device = 'cpu'):
    grd = torch.nn.functional.affine_grid(Affine_mtrx, size=source_img.shape,align_corners=False)
    if source_img.device.type=='cpu':
       wrapped_img = torch.nn.functional.grid_sample(source_img, grid=grd,
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)
    else:
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

def save_examples(model, loader , n_examples = 4, plt_markrs=True,plt_imgs=False, time=1, feed_origion=True, shadow=False, win = 5):
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
            if plt_markrs:
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
        if plt_markrs:
                suffix += 'Elipse'
        if plt_imgs:
                suffix += 'Img'
        plt.savefig(savingfolder+'MIRexamples/{}_{}.jpeg'.format(suffix,time), bbox_inches='tight')
        plt.close()


def save_n_examples(model, loader,n_times=2, n_examples_per_time = 4 ):
    for m in range(n_times):
        save_examples(model, loader, n_examples = n_examples_per_time, plt_markrs=True, plt_imgs=False, time =m)
        save_examples(model, loader, n_examples = n_examples_per_time, plt_markrs=False, plt_imgs=True, time =m)
        save_examples(model, loader, n_examples = n_examples_per_time, plt_markrs=True, plt_imgs=True, time =m)

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

def Generate_random_AffineMatrix(measure=Intitial_Tx, NOISE_LEVEL=0, block_min =0, block_max=1, device=device_aug):
    angle = 0
    #scale_inv = 0.5#(1 - 0.3)/1.4
    scaleX_inv = 0.5
    scaleY_inv = 0.5
    translationx = 0.5
    translationy = 0.5
    shearx=0.5
    sheary=0.5
    reflectionX=1.0
    reflectionY=1.0
    # if augmentation
    if 'angle' in measure: #angle = np.random.uniform(0, 1) #1 is equivalent to 360 degrees or 2pi = 6.29
        if np.random.rand()<0.5:
            angle = np.random.uniform(block_min/2, block_max/2)
        else:
            angle = np.random.uniform(1-block_max/2, 1-block_min/2)
    if 'scaleX' in measure:
        if np.random.rand()<0.5:
            scaleX_inv = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            scaleX_inv = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
    if 'scaleY' in measure:
        if np.random.rand()<0.5:
            scaleY_inv = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            scaleY_inv = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
    if 'scale' in measure:
        if np.random.rand()<0.5:
            scale_inv = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            scale_inv = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
        scaleX_inv=scale_inv
        scaleY_inv=scale_inv
    if 'translationX' in measure:
        if np.random.rand()<0.5:
            translationx = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            translationx = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
    if 'translationY' in measure:
        if np.random.rand()<0.5:
            translationy = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            translationy = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
    if 'shearX' in measure:
        if np.random.rand()<0.5:
            shearx = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            shearx = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
    if 'shearY' in measure:
        if np.random.rand()<0.5:
            sheary = np.random.uniform(0.5-block_max/2, 0.5-block_min/2)
        else:
            sheary = np.random.uniform(0.5+block_min/2, 0.5+block_max/2)
    if 'reflectionX' in measure:
        if block_max> 0.5:
            if np.random.rand()<0.5*block_max:
                reflectionX = -1.0
            else:
                reflectionX = 1.0
    if 'reflectionY' in measure:
        if block_max> 0.5:
            if np.random.rand()<0.5*block_max:
                reflectionY = -1.0
            else:
                reflectionY = 1.0
    #if shearx = np.random.uniform(block_min, 1-block_max)
    #if 'sheary' in measure: sheary = np.random.uniform(block_min, 1-block_max)
    Affine_parameters = torch.tensor([[angle, scaleX_inv, scaleY_inv, translationx,translationy, shearx, sheary,reflectionX,reflectionY]]).to(torch.float32)
    Affine_mtrx = normalizedparameterline2Affine_matrx(Affine_parameters, device=device, Noise_level=NOISE_LEVEL)
    return Affine_mtrx, Affine_parameters

def pass_augment_img(image0, measure,NOISE_LEVEL=0, MODE='nearest', block_min =0, block_max=1 ):
    try:
        dimw = image0.width
        dimh = image0.height
    except:
        dimw = image0.shape[2]
        dimh = image0.shape[3]
    Affine_mtrx, Affine_parameters = Generate_random_AffineMatrix(measure=measure, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max, device=device_aug)
    if Fix_Torch_Wrap:
        wrapped_img = wrap_imge_with_inverse(Affine_mtrx.detach(), image0, ratio= 2*crop_ratio)
        Affine_mtrx = workaround_matrix(Affine_mtrx.detach(), acc = 2).detach()
    else:
        wrapped_img = wrap_imge0(Affine_mtrx, image0)
    return wrapped_img, Affine_mtrx, Affine_parameters


def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx, block_min =BLOCK_MIN, block_max=BLOCK_MAX): #,'shearx','sheary'
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
    return wrapped_img, Affine_mtrx, Affine_parameters


def Load_augment(image_path, dim = 128): #measures=['angle', 'scale', 'translationX','translationY','shearx','sheary'], Noise_level=0,
  #enlargement = crop_ratio
  Outputs= {}
  #image0 = load_image_from_url(image_path, int(enlargement*dim))
  #image0 = (transforms.ToTensor()(image0)).unsqueeze(0).to(torch.float32)
  image0 = load_image_pil_accelerated(image_path).unsqueeze(0)
  transformed_img0, Affine_mtrx, Affine_parameters = augment_img(image0)#, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx)
  image1 = torchvision.transforms.CenterCrop((dim, dim))(image0)
  transformed_img = torchvision.transforms.CenterCrop((dim, dim))(transformed_img0)
  if SWITCH:
    Switch_threshold = 0.5
  else:
    Switch_threshold = 10
  if np.random.rand()<=Switch_threshold:
      Outputs['Affine_mtrx'] = Affine_mtrx[0]
      Outputs['source_origion'] = image0[0]
      Outputs['source'] = image1[0]
      Outputs['target'] = transformed_img[0]
      Outputs['Affine_parameters'] = Affine_parameters[0]
  else:
      Outputs['Affine_mtrx'] = inv_AM(Affine_mtrx)[0].detach()
      Outputs['source_origion'] = transformed_img0[0].detach()
      Outputs['source'] = transformed_img[0].detach()
      Outputs['target'] = image1[0].detach()
  return Outputs


def normalizedparameterline2Affine_matrx(line, device, Noise_level=0.0):
    N_batches = int(line.shape[0])
    Affine_mtrx = torch.ones(N_batches, 2,3)#.to(device)
    DeNormalize_parametersline = DeNormalize_AffineParameters(line)
    for i in range(N_batches):
        angle = DeNormalize_parametersline[i:i+1,0:1]
        scaleX_inv = DeNormalize_parametersline[i:i+1,1:2]
        scaleY_inv = DeNormalize_parametersline[i:i+1,2:3]
        translationx = DeNormalize_parametersline[i:i+1,3:4]
        translationy = DeNormalize_parametersline[i:i+1,4:5]
        shearx = DeNormalize_parametersline[i:i+1,5:6]
        sheary = DeNormalize_parametersline[i:i+1,6:7]
        reflectionX = DeNormalize_parametersline[i:i+1,7:8]
        reflectionY = DeNormalize_parametersline[i:i+1,8:9]
        Affine_Mtrx_i = Affine_parameters2matrx(angle, scaleX_inv, scaleY_inv, translationx,translationy, shearx, sheary,reflectionX,reflectionY)
        Affine_mtrx[i,:] = Affine_Mtrx_i#[:2,:]#.reshape(1,2,3)
    random_component = torch.normal(torch.zeros([N_batches,2,3]), Noise_level*torch.ones([N_batches,2,3]))
    #truncate values that exceeds a threshold
    threshold = 3.0*Noise_level
    random_component = torch.clamp(random_component, min=-threshold, max=threshold)
    Affine_mtrx = Affine_mtrx+ Affine_mtrx*(random_component)#.to(device_aug))
    return Affine_mtrx.to(torch.float32)

def Normalize_AffineParameters(parameters):
   Norm_parameters = parameters.clone()
   Norm_parameters[:,0]/= 6.29
   Norm_parameters[:,1] = (Norm_parameters[:,1] - 0.2)/1.6
   Norm_parameters[:,2] = (Norm_parameters[:,2] - 0.2)/1.6
   Norm_parameters[:,3] = (Norm_parameters[:,3] + 0.25)/0.5
   Norm_parameters[:,4] = (Norm_parameters[:,4] + 0.25)/0.5
   Norm_parameters[:,5] = (Norm_parameters[:,5] + 0.5)/1.0
   Norm_parameters[:,6] = (Norm_parameters[:,6] + 0.5)/1.0
   Norm_parameters[:,7] = Norm_parameters[:,7]
   Norm_parameters[:,8] = Norm_parameters[:,8]
   return Norm_parameters

def DeNormalize_AffineParameters(Normalized_Parameters):
   DeNormalized_Parameters = Normalized_Parameters.clone()
   DeNormalized_Parameters[:,0]*= 6.29
   DeNormalized_Parameters[:,1] = 1.6*DeNormalized_Parameters[:,1] + 0.2
   DeNormalized_Parameters[:,2] = 1.6*DeNormalized_Parameters[:,2] + 0.2
   DeNormalized_Parameters[:,3] = 0.5*DeNormalized_Parameters[:,3] - 0.25
   DeNormalized_Parameters[:,4] = 0.5*DeNormalized_Parameters[:,4] - 0.25
   DeNormalized_Parameters[:,5] = 1.0*DeNormalized_Parameters[:,5] - 0.5
   DeNormalized_Parameters[:,6] = 1.0*DeNormalized_Parameters[:,6] - 0.5
   DeNormalized_Parameters[:,7] = DeNormalized_Parameters[:,7]
   DeNormalized_Parameters[:,8] = DeNormalized_Parameters[:,8]
   return DeNormalized_Parameters

def Affine_parameters2matrx(angle, scaleX_inv, scaleY_inv, translationx,translationy, shearx, sheary,reflectionX,reflectionY):
   N_batches = 1#int(line.shape[0])
   Affine_mtrx = torch.ones(N_batches, 2,3)#.to(device)
   for i in range(N_batches):
       Mat_shear = torch.tensor([[1.0, shearx, 0.0],[sheary,1.0, 0.0],[0.0, 0.0, 1.0]])
       Mat_translation = torch.tensor([[1.0, 0.0, translationx],[0.0, 1.0, translationy],[0.0, 0.0, 1.0]])
       Mat_scale_reflection = torch.tensor([[reflectionX*scaleX_inv, 0.0, 0.0],[0.0, reflectionY*scaleY_inv, 0.0],[0.0, 0.0, 1.0]])
       Mat_Rot = torch.tensor([[torch.cos(angle), torch.sin(angle), 0.0],
                               [-torch.sin(angle), torch.cos(angle), 0.0],
                               [0.0, 0.0, 1.0]]).float()
       Affine_Mtrx_i = torch.matmul(torch.matmul(torch.matmul(Mat_shear,Mat_scale_reflection), Mat_Rot),Mat_translation)
       Affine_mtrx[i,:] = Affine_Mtrx_i[:2,:].reshape(1,2,3)
   return Affine_mtrx

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

def add_noise2img(img0, sigma = IMG_noise_level):
    img0 = img0.detach()#.to(device_aug)
    s1 = sigma*torch.rand(1)#.to(device_aug)
    if torch.rand(1)>0.5:
        gamma = 0.2+0.8*torch.rand(1)+0.1
    else:
        gamma = 1+4*torch.rand(1)
    c = torch.randn(img0.shape)*s1#torch.normal(torch.zeros_like(img0), s1*torch.ones_like(img0))#.to(device_aug)
    img0_clipped = torch.clip(img0*(1 + c),0,1)
    img_adjusted0 = (img0_clipped)**(gamma)#.to(device_aug))
    img_adjusted_clipped = torch.clip(img_adjusted0,0,1)
    return img_adjusted_clipped.detach()

def Generate_Mi(folder_prefx='completely_random', mode='test', noise=0, block_min =0, block_max=1, Affine_mtrx=torch.zeros([2,3]) ):
    if 'completely_random' in folder_prefx :
        M_i = torch.normal(torch.zeros([2,3]), torch.ones([2,3]))
    elif 'random_aff_param' in folder_prefx:
        M_i =  Generate_random_AffineMatrix(measure=Intitial_Tx, NOISE_LEVEL=noise, block_min =block_min, block_max=block_max, device=device_aug)[0][0]
    elif 'mix' in folder_prefx:
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
                            folder_prefx = 'random',
                            mode= 'train' ):
        self.list_paths = list_paths[mode]
        self.dim = dim
        self.mode = mode
        self.activedataset = DATASET[self.mode]
        self.batch_size = batch_size
        self.number_examples = len(self.list_paths)
        self.registration_method = registration_method
        self.folder_prefx = folder_prefx
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
            source = torchvision.transforms.CenterCrop((self.dim, self.dim))(source_origion).to(torch.float32)
            target = add_noise2img(target)
        if 'Recurence' in self.registration_method:
            M_i = Generate_Mi(folder_prefx=self.folder_prefx, mode=self.mode,
                    noise=Noise_level_dataset, Affine_mtrx= Affine_mtrx )
            X = {'source':source,
                'target':target,
                'M_i' :M_i }
            if 'Additive' in self.registration_method:
                Y = {'Affine_mtrx': Affine_mtrx,
                   'Deviated_mtrx': Affine_mtrx - X['M_i']},
            elif 'Multiplicative' in self.registration_method:
                Y = {'Affine_mtrx': Affine_mtrx,
                   'Deviated_mtrx': torch.matmul(mtrx3(Affine_mtrx), torch.linalg.inv(mtrx3(X['M_i'])))[0:2,:]}
        else:
            X = {'source':source,'target':target}
            Y = {'Affine_mtrx': Affine_mtrx}
        #if 'test' in self.mode or Fine_tuned:
            #X['source_origion'] = source_origion.to(device)
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
            if '1' in folder_prefx:
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



class Build_IRmodel_ViT(torch.nn.Module):
    def __init__(self, core_model, registration_method = 'Rawblock', overlap='horizontal', Arch = 'mobileVIT'):
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


class CreatePatches(nn.Module):
    def __init__(self, channels=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
    def forward(self, x):
        # Flatten along dim = 2 to maintain channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        return patches

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.pre_norm(x)
        # MultiheadAttention returns attention output and weights,
        # we need only the outputs, so [0] index.
        x = x + self.attention(x_norm, x_norm, x_norm)[0]
        x = x + self.MLP(self.norm(x))
        return x

class rawViT(nn.Module):
    def __init__(
        self,
        img_size=128,
        in_channels=6,
        patch_size=16,
        embed_dim=768,
        hidden_dim=3072,
        num_heads=12,
        num_layers=12,
        dropout=0.0,
        num_classes=6
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size//patch_size) ** 2
        self.patches = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.attn_layers.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x = self.patches(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.ln(x)
        x = x[:, 0]
        return self.head(x)


class Torch_NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None,stride = 1):
        self.win = win
        self.eps = 1e-2
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
        cross[torch.abs(cross<self.eps)] = self.eps
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var[torch.abs(I_var<self.eps)] = self.eps
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var[torch.abs(J_var<self.eps)] = self.eps
        cc = ((cross * cross )+ self.eps) / ((I_var * J_var) + self.eps)
        return cc
    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        cc = torch.mean(cc)
        return -cc

NCC_loss = Torch_NCC(win=128,stride = 128).loss
#NCC_loss(target.detach(), wrapped_img.detach()).item()

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
    def __init__(self, model, Num_iterations=10, key = 'Affine_mtrx',mode='test', Matrix_initialization=folder_prefx,MI_noise=1,AM=torch.zeros([batch_size,2,3])):
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

from transformers import MobileViTV2ForImageClassification
if 'VIT' in Arch:
    if 'mobileVIT' in Arch:
        core_model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        if overlap=='vertical':
            original_conv1 = core_model.mobilevitv2.conv_stem.convolution
            new_conv1 = torch.nn.Conv2d(6,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # Initialize the new convolutional layer's weights
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3:, :, :] = original_conv1.weight
            core_model.mobilevitv2.conv_stem.convolution = new_conv1
            core_model.classifier = torch.nn.Linear(512, 6)
    elif Arch == 'rawVIT':
        core_model = rawViT(img_size=128,in_channels=6,patch_size=12,embed_dim=384,hidden_dim=768,num_heads=12,num_layers=4, num_classes=6)
    IR_Model = Build_IRmodel_ViT(core_model, registration_method = registration_method, overlap=overlap, Arch = Arch)
    core_model.to(device)
    IR_Model.to(device)


pytorch_total_params = sum(p.numel() for p in IR_Model.parameters())
print('Number of paprameters: ',pytorch_total_params/1000000, 'Millions')



#---------------Train--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------

#from transformers import TrainingArguments
#https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#data-preloading

if Arch == 'rawVIT':
    Learning_rate = 0.0001
else:
    Learning_rate = 0.001


print_every = int(2000*80/batch_size)


#IR_Model = torch.compile(IR_Model)
MSE_loss = torch.nn.functional.mse_loss
optimizer = torch.optim._multi_tensor.AdamW(IR_Model.parameters(), lr=Learning_rate)#, momentum=0.9

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#def train_IR_model(IR_Model, trainloader, TOTAL_Epochs = 12,):
training_loss_epochslist = []
validation_loss_epochslist = []
training_loss_iterationlist = []
global_loss_iterationlist =[]
M1_training_loss_iterationlist = []
validation_loss_iterationlist = []
TOTAL_Epochs = 12
best_loss = 100000000000000000000


for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    loss_prnt_global= 0.0
    loss_prnt_global1= 0.0
    loss_prnt_cyclic0= 0.0
    loss_prnt_cyclic1= 0.0
    if Curriculum_learning:
        #Difficulty_factor = min(1, (EPOCH+1)/TOTAL_Epochs)
        #Difficulty_factor = min(1, np.log(0.5*EPOCH+1)/np.log(0.5*TOTAL_Epochs-7))
        if EPOCH+7<TOTAL_Epochs:
            Difficulty_factor= np.log(0.5*EPOCH+1)/np.log(0.5*TOTAL_Epochs)
        else:
            Difficulty_factor= 1
        def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx, block_min =0, block_max= Difficulty_factor): 
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
            return wrapped_img, Affine_mtrx, Affine_parameters
    if with_Difficultyadj_loss:
        weight_loss = Difficulty_factor
    else:
        weight_loss = 0.5
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for inputs, labels in loop:
        i+=1
        inputs = move_dict2device(inputs,device)
        labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IR_Model(inputs)
        loss = 0.0
        if with_global_loss:
            global_loss = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'])
            if include_loss_model1:
                global_loss1 = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx_0'])
                loss_prnt_global1 += global_loss1.detach().item()
                loss += (1-weight_loss)*global_loss + weight_loss*global_loss1
            else:
                #global_loss1 =0
                loss_prnt_global1= 0
                loss += global_loss
            loss_prnt_global += loss.detach().item()
        if with_Cyclic_losses:
            Affine_mtrx_0_gt = torch.matmul( mtrx3(labels['Affine_mtrx']), mtrx3(inv_AM(predections['pred_1'])) )[:,:2,:3]
            loss0_cyclic = cyclic_loss_factor*MSE_loss(Affine_mtrx_0_gt, predections['Affine_mtrx_0'])
            loss_prnt_cyclic0 += loss0_cyclic.detach().item()
            Affine_mtrx_1_gt = torch.matmul( mtrx3(inv_AM(predections['Affine_mtrx_0'])), mtrx3(labels['Affine_mtrx']))[:,:2,:3]
            loss1_cyclic = cyclic_loss_factor*MSE_loss(Affine_mtrx_1_gt, predections['pred_1'])
            loss_prnt_cyclic1 += loss1_cyclic.detach().item()
            loss += (loss0_cyclic + loss1_cyclic)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IR_Model, valloader,int(2000/batch_size),'Affine_mtrx').detach().item()
                if with_scheduler:
                    scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '                   
                if with_global_loss:
                    global_loss_iterationlist.append( threshold(loss_prnt_global/print_every))
                    printed_text += f'global: {loss_prnt_global/print_every:.3f},'
                    if include_loss_model1:
                        M1_training_loss_iterationlist.append( threshold(loss_prnt_global1/print_every))
                        printed_text += f'global1: {loss_prnt_global1/print_every:.3f},'
                        FreePalestine = 1948
                if with_Cyclic_losses:
                    printed_text += f'cyclic0: {loss_prnt_cyclic0/print_every:.3f},'
                    printed_text += f'cyclic1: {loss_prnt_cyclic1/print_every:.3f}'
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
                loss_prnt_global= 0.0
                loss_prnt_global1= 0.0
                loss_prnt_cyclic0= 0.0
                loss_prnt_cyclic1= 0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    if with_global_loss:
        plt.plot(global_loss_iterationlist, label = 'global loss')
    if include_loss_model1:
        plt.plot(M1_training_loss_iterationlist, label = 'model1 training loss')
        np.savetxt(savingfolder+'M1_training_loss_iterationlist.txt', M1_training_loss_iterationlist, delimiter=",", fmt="%.3f")
    
    plt.legend()
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")
    if with_global_loss:
        np.savetxt(savingfolder+'global_loss_iterationlist.txt', global_loss_iterationlist, delimiter=",", fmt="%.3f")
        


torch.save(IR_Model.state_dict(), savingfolder+'IR_Model_EndTraining.pth')
torch.save(core_model.state_dict(), savingfolder+'./core_model_EndTraining.pth')
    
if with_epch_loss:
    with open(savingfolder+'training_loss_epochslist.npy', 'wb') as f:
        np.save(f, np.array(training_loss_epochslist))
    
    with open(savingfolder+'validation_loss_epochslist.npy', 'wb') as f:
        np.save(f, np.array(validation_loss_epochslist))
    
    plt.plot(training_loss_epochslist, label = 'training loss')
    plt.plot(validation_loss_epochslist, label = 'validation loss')
    plt.legend()
    plt.savefig(savingfolder+'loss_epochs.png', bbox_inches='tight')
    plt.close()

with open(savingfolder+'training_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(training_loss_iterationlist))

with open(savingfolder+'validation_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(validation_loss_iterationlist))


plt.plot(training_loss_iterationlist, label = 'training loss')
plt.plot(validation_loss_iterationlist, label = 'validation loss')
plt.legend()
plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
plt.close()



## ----------------------------------------------------------------------------------

## -------------------------------- Fine tuning model -------------------------------
## -------------------------------- Fine tuning model -------------------------------
## -------------------------------- Fine tuning model -------------------------------
## ----------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------

file_loadingfolder = '/home/ahmadh/MIR_savedmodel/Rawblock_Mi0:12Epochs_bidirectional__uniscale_AdjustPlot_avgPool_LR_V.concat__mobileVIT_active_0.1/'
ext = '_EndTraining'#'_EndTraining' #_bestVal

FREEZE_stage1 = False
if 'VIT' in Arch:
    if 'mobileVIT' in Arch:
        from transformers import MobileViTV2ForImageClassification
        core_model_stage1 = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        if overlap=='vertical':
            original_conv1 = core_model_stage1.mobilevitv2.conv_stem.convolution
            new_conv1 = torch.nn.Conv2d(6,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # Initialize the new convolutional layer's weights
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3:, :, :] = original_conv1.weight
            core_model_stage1.mobilevitv2.conv_stem.convolution = new_conv1
            core_model_stage1.classifier = torch.nn.Linear(512, 6)
    elif Arch == 'rawVIT':
        core_model_stage1 = rawViT(img_size=128,in_channels=6,patch_size=12,embed_dim=384,hidden_dim=768,num_heads=12,num_layers=4, num_classes=6)
    IR_Model_stage1 = Build_IRmodel_ViT(core_model_stage1,registration_method = registration_method, overlap=overlap, Arch = Arch)
    core_model_stage1.to(device)
    if FREEZE_stage1:
        IR_Model_stage1.load_state_dict(torch.load(file_loadingfolder+'IR_Model'+ext+'.pth'))
    IR_Model_stage1.to(device)

if 'VIT' in Arch:
    if 'mobileVIT' in Arch:
        core_model_stage2 = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        if overlap=='vertical':
            original_conv1 = core_model_stage2.mobilevitv2.conv_stem.convolution
            new_conv1 = torch.nn.Conv2d(6,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # Initialize the new convolutional layer's weights
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3:, :, :] = original_conv1.weight
            core_model_stage2.mobilevitv2.conv_stem.convolution = new_conv1
            core_model_stage2.classifier = torch.nn.Linear(512, 6)
    elif Arch == 'rawVIT':
        core_model_stage2 = rawViT(img_size=128,in_channels=6,patch_size=12,embed_dim=384,hidden_dim=768,num_heads=12,num_layers=4, num_classes=6)
    IR_Model_stage2 = Build_IRmodel_ViT(core_model_stage2,registration_method = registration_method, overlap=overlap, Arch = Arch)
    core_model_stage2.to(device)
    if FREEZE_stage1:
        IR_Model_stage2.load_state_dict(torch.load(file_loadingfolder+'IR_Model'+ext+'.pth'))
    IR_Model_stage2.to(device)


# freeze weights of the first model
if FREEZE_stage1:
    IR_Model_stage1.eval()
    for param in IR_Model_stage1.parameters():
        param.requires_grad = False


class Build_IRmodel_finetuned_ViT(torch.nn.Module):
    def __init__(self,IR_Model_stage1, IR_Model_stage2,No_recurences=2):
        super(Build_IRmodel_finetuned_ViT, self).__init__()
        self.IR_Model_stage1 = IR_Model_stage1
        self.IR_Model_stage2 = IR_Model_stage2
        self.overlap = overlap
        self.No_recurences = No_recurences
    def forward(self, input_X_batch):
        source = input_X_batch['source']
        target = input_X_batch['target']
        source_origion = input_X_batch['source_origion'].detach()
        predections_stage1 = self.IR_Model_stage1(input_X_batch)
        Affine_matrix0 = predections_stage1['Affine_mtrx']
        predction = {'Affine_mtrx_0': Affine_matrix0}
        New_Accumilative_Affine_matrix = Affine_matrix0
        grd = torch.nn.functional.affine_grid(Affine_matrix0, size=source_origion.shape,align_corners=False)
        source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                mode='bilinear', padding_mode='zeros', align_corners=False)
        wrapped_img = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
        if FREEZE_stage1:
            wrapped_img = wrapped_img.detach()
        for j in range(1, self.No_recurences):
            inputs_j ={'source': wrapped_img.to(device),
                        'target': target.to(device),
                        'M_i': torch.zeros([int(input_X_batch['source'].shape[0]),2,3]).to(device)
                        }
            predections_stage2 = self.IR_Model_stage2(inputs_j)
            Affine_mtrx_j = predections_stage2['Affine_mtrx']
            predction['pred_{}'.format(j)] = Affine_mtrx_j
            Accumilative_Affine_matrix = New_Accumilative_Affine_matrix
            New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix), mtrx3(Affine_mtrx_j))[:,0:2,:]
            predction['Affine_mtrx_{}'.format(j)] = New_Accumilative_Affine_matrix
            grd_acc = torch.nn.functional.affine_grid(New_Accumilative_Affine_matrix, size=source_origion.shape,align_corners=False)
            source_origion_224j_acc = torch.nn.functional.grid_sample(source_origion, grid=grd_acc,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
            wrapped_img_acc = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j_acc)
            wrapped_img = wrapped_img_acc
            if FREEZE_stage1:
                wrapped_img = wrapped_img.detach()
        predction['Affine_mtrx']= New_Accumilative_Affine_matrix
        return predction


IRmodel_finetuned = Build_IRmodel_finetuned_ViT(IR_Model_stage1, IR_Model_stage2,No_recurences=2)



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of trainable paprameters: ',count_trainable_parameters(IRmodel_finetuned)/1000000, 'Millions')

pytorch_total_params = sum(p.numel() for p in IRmodel_finetuned.parameters())
print('Number of paprameters: ',pytorch_total_params/1000000, 'Millions')



Learning_rate = 0.0001
cyclic_loss_factor = 0.05

print_every = int(2000*80/batch_size)
MSE_loss = torch.nn.functional.mse_loss
optimizer = optim.AdamW(IRmodel_finetuned.parameters(), lr=Learning_rate)#, momentum=0.9
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#def train_IR_model(IR_Model, trainloader, TOTAL_Epochs = 12,):
training_loss_iterationlist = []
M1_training_loss_iterationlist = []
global_loss_iterationlist = []
validation_loss_iterationlist = []
TOTAL_Epochs = 12
if Curriculum_learning:
    TOTAL_Epochs = 45

best_loss = 100000000000000000000


for EPOCH in range(0, TOTAL_Epochs):  # loop over the dataset multiple times
    i=-1
    running_loss = 0.0
    loss_prnt_global= 0.0
    loss_prnt_global1= 0.0
    loss_prnt_cyclic0= 0.0
    loss_prnt_cyclic1= 0.0
    if Curriculum_learning:
        #Difficulty_factor = min(1, (EPOCH+1)/TOTAL_Epochs)
        #Difficulty_factor = min(1, np.log(0.5*EPOCH+1)/np.log(0.5*TOTAL_Epochs-7))
        if EPOCH+7<TOTAL_Epochs:
            Difficulty_factor= np.log(0.5*EPOCH+1)/np.log(0.5*TOTAL_Epochs)
        else:
            Difficulty_factor= 1
        def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx, block_min =0, block_max= Difficulty_factor): 
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
            return wrapped_img, Affine_mtrx, Affine_parameters
    if with_Difficultyadj_loss:
        weight_loss = Difficulty_factor
    else:
        weight_loss = 0.5
    if HPC:
        loop = trainloader
    else:
        loop = tqdm.tqdm(trainloader)
    for inputs, labels in loop:
        i+=1
        inputs = move_dict2device(inputs,device)
        labels = move_dict2device(labels,device)
        optimizer.zero_grad()
        predections = IRmodel_finetuned(inputs)
        loss = 0.0
        if with_global_loss:
            global_loss = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx'])
            if include_loss_model1:
                global_loss1 = MSE_loss(labels['Affine_mtrx'], predections['Affine_mtrx_0'])
                loss_prnt_global1 += global_loss1.detach().item()
                loss += (1-weight_loss)*global_loss + weight_loss*global_loss1
            else:
                #global_loss1 =0
                loss_prnt_global1= 0
                loss += global_loss
            loss_prnt_global += loss.detach().item()
        if with_Cyclic_losses:
            Affine_mtrx_0_gt = torch.matmul( mtrx3(labels['Affine_mtrx']), mtrx3(inv_AM(predections['pred_1'])) )[:,:2,:3]
            loss0_cyclic = cyclic_loss_factor*MSE_loss(Affine_mtrx_0_gt, predections['Affine_mtrx_0'])
            loss_prnt_cyclic0 += loss0_cyclic.detach().item()
            Affine_mtrx_1_gt = torch.matmul( mtrx3(inv_AM(predections['Affine_mtrx_0'])), mtrx3(labels['Affine_mtrx']))[:,:2,:3]
            loss1_cyclic = cyclic_loss_factor*MSE_loss(Affine_mtrx_1_gt, predections['pred_1'])
            loss_prnt_cyclic1 += loss1_cyclic.detach().item()
            loss += (loss0_cyclic + loss1_cyclic)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        if i >0:
            if i % print_every == 0:
                eval_loss_x = test_loss(IRmodel_finetuned, valloader,int(2000/batch_size),'Affine_mtrx').detach().item()
                if with_scheduler:
                    scheduler.step(eval_loss_x)
                training_loss_iterationlist.append( threshold(running_loss/print_every))
                validation_loss_iterationlist.append(threshold(eval_loss_x))
                printed_text = f'[epoch:{EPOCH}, iter:{i:5d}], training loss: {running_loss/print_every:.3f}, eval loss:{eval_loss_x:.3f}, '                   
                if with_global_loss:
                    global_loss_iterationlist.append( threshold(loss_prnt_global/print_every))
                    printed_text += f'global: {loss_prnt_global/print_every:.3f},'
                    if include_loss_model1:
                        M1_training_loss_iterationlist.append( threshold(loss_prnt_global1/print_every))
                        printed_text += f'global1: {loss_prnt_global1/print_every:.3f},'
                        FreePalestine = 1948
                if with_Cyclic_losses:
                    printed_text += f'cyclic0: {loss_prnt_cyclic0/print_every:.3f},'
                    printed_text += f'cyclic1: {loss_prnt_cyclic1/print_every:.3f}'
                print(printed_text)
                if i >1:
                    if eval_loss_x<best_loss:
                        best_loss = eval_loss_x
                running_loss = 0.0
                loss_prnt_global= 0.0
                loss_prnt_global1= 0.0
                loss_prnt_cyclic0= 0.0
                loss_prnt_cyclic1= 0.0
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    if with_global_loss:
        plt.plot(global_loss_iterationlist, label = 'training loss')
        np.savetxt(savingfolder+'global_loss_iterationlist.txt', global_loss_iterationlist, delimiter=",", fmt="%.3f")
    if include_loss_model1:
        plt.plot(M1_training_loss_iterationlist, label = 'model1 training loss')
        np.savetxt(savingfolder+'M1_training_loss_iterationlist.txt', M1_training_loss_iterationlist, delimiter=",", fmt="%.3f")
    plt.legend()
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")
    



print('Finished Training')
    #return IR_Model, training_loss_iterationlist, validation_loss_iterationlist

#IR_Model_tst, training_loss_iterationlist, validation_loss_iterationlist = train_IR_model(IRmodel_finetuned, trainloader, TOTAL_Epochs = 12,)

torch.save(IRmodel_finetuned.state_dict(), savingfolder+'IRmodel_finetuned_EndTraining.pth')
torch.save(IR_Model_stage2.state_dict(), savingfolder+'./IR_Model_stage2_EndTraining.pth')
torch.save(IR_Model_stage1.state_dict(), savingfolder+'./IR_Model_stage1_EndTraining.pth')
torch.save(core_model_stage2.state_dict(), savingfolder+'./core_model_stage2_EndTraining.pth')
torch.save(core_model_stage1.state_dict(), savingfolder+'./core_model_stage1_EndTraining.pth')





with open(savingfolder+'training_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(training_loss_iterationlist))

with open(savingfolder+'validation_loss_iterationlist.npy', 'wb') as f:
    np.save(f, np.array(validation_loss_iterationlist))


plt.plot(training_loss_iterationlist, label = 'training loss')
plt.plot(validation_loss_iterationlist, label = 'validation loss')
plt.legend()
plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
plt.close()



os.system('mkdir '+ savingfolder+ 'MIRexamples')
AM_recurrent_loss(IR_Model_stage2.eval(),testloader0 , max_iterations=100, No_recurences = 7, key ='Affine_mtrx', plot=True, plot_batach = 5, prefix = 'randomdifficulty')


def test_loss_i(model, loader, max_iterations=100, key = 'Affine_mtrx'):
    eval_loss_tot = {}
    eval_loss_avg = {}
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            if i < max_iterations:
                inputs, labels = data
                inputs = move_dict2device(inputs,device)
                labels = move_dict2device(labels,device)
                predections = model(inputs)
                try:
                    for k in predections.keys():
                        eval_loss_tot[k] += MSE_loss(labels[key], predections[k].detach())
                        eval_loss_avg[k] = eval_loss_tot[k].detach().item()/i
                except:
                    for k in predections.keys():
                        eval_loss_tot[k]=0
                        eval_loss_avg[k]=0
                    #print('initialization')
            else :
                return eval_loss_avg
    return eval_loss_avg




IRmodel_finetuned3 = Build_IRmodel_finetuned_ViT(IR_Model_stage1.eval(), IR_Model_stage2.eval(),No_recurences=4)
MSE_AffineMatrix_recurrent_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    MEASURES= Intitial_Tx
    Noise_level_testset=0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_recurrent_difficulty[Difficulty] = test_loss_i(IRmodel_finetuned3.eval(), testloader0,70, key = 'Affine_mtrx')#.detach().item()


print(MSE_AffineMatrix_recurrent_difficulty)
json.dump(MSE_AffineMatrix_recurrent_difficulty, open( savingfolder+'MSE_AffineMatrix_recurrent_difficulty_model1-3xmodel2.txt', 'w' ) )



def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =0, block_max=1): 
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
    return wrapped_img, Affine_mtrx, Affine_parameters


save_examples(IRmodel_finetuned3,testloader0, n_examples = 6, plt_markrs=True,plt_imgs=True,time=6, feed_origion=True, shadow=True, win=9)



IRmodel_finetuned
AM_recurrent_loss(IRmodel_finetuned.eval(),testloader0 , max_iterations=10, No_recurences = 2, key ='Affine_mtrx', plot=False, plot_batach = 0, prefix = 'randomdifficulty')


Measures_list = Intitial_Tx#['angle', 'scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflectionX', 'reflectionY']
BLOCK_options = {'Full':(0,1), 'Hard70-100%':(0.7,1), 'Medium30-70':(0.3,0.7), 'Easy10-30%':(0.1,0.3), 'Easier5-10%':(0.05,0.1), 'Easiest0-5%':(0,0.05)}
MSE_AffineMatrix_recurrent_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    MEASURES= Intitial_Tx
    Noise_level_testset=0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_recurrent_difficulty[Difficulty] = AM_recurrent_loss(IRmodel_finetuned.eval(),testloader0, max_iterations=70,
                                                                            No_recurences = 5, key ='Affine_mtrx', plot=True, plot_batach = 5, prefix = Difficulty)



print(MSE_AffineMatrix_recurrent_difficulty)
json.dump(MSE_AffineMatrix_recurrent_difficulty, open( savingfolder+'MSE_AffineMatrix_recurrent_difficulty_full_model.txt', 'w' ) )






## ----------------------------------------------------------------------------------

## --------------------------------  Active learning  -------------------------------
## --------------------------------  Active learning  -------------------------------
## --------------------------------  Active learning  -------------------------------
## ----------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------


file_loadingfolder = '/home/ahmadh/MIR_savedmodel/Rawblock_Mi0:12Epochs_bidirectional__uniscale_AdjustPlot_avgPool_LR_V.concat__mobileVIT_active_0.1/'
ext = '_EndTraining'#'_EndTraining' #_bestVal

if 'VIT' in Arch:
    if 'mobileVIT' in Arch:
        from transformers import MobileViTV2ForImageClassification
        core_model0 = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        if overlap=='vertical':
            original_conv1 = core_model0.mobilevitv2.conv_stem.convolution
            new_conv1 = torch.nn.Conv2d(6,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # Initialize the new convolutional layer's weights
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3:, :, :] = original_conv1.weight
            core_model0.mobilevitv2.conv_stem.convolution = new_conv1
            core_model0.classifier = torch.nn.Linear(512, 6)
    elif Arch == 'rawVIT':
        core_model0 = rawViT(img_size=128,in_channels=6,patch_size=12,embed_dim=384,hidden_dim=768,num_heads=12,num_layers=4, num_classes=6)
    IR_Model0 = Build_IRmodel_ViT(core_model0,registration_method = registration_method, overlap=overlap, Arch = Arch)
    core_model0.to(device)
    IR_Model0.load_state_dict(torch.load(file_loadingfolder+'IR_Model'+ext+'.pth'))
    IR_Model0.to(device)

# Freeze_all_except_last_n_layers(model, n):
num_layers=0
for param in IR_Model0.parameters():
    num_layers +=1

n = 300
count=0
for param in IR_Model0.parameters():
    count +=1
    if count < num_layers - n: #freezing last 3 layers
        param.requires_grad = False
    else:
        param.requires_grad = True

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of trainable paprameters: ',count_trainable_parameters(IR_Model0)/1000000, 'Millions')

pytorch_total_params = sum(p.numel() for p in IR_Model0.parameters())
print('Number of paprameters: ',pytorch_total_params/1000000, 'Millions')


dataiter = iter(valloader)
X_batch, Y_batch = next(dataiter)

X_batch = move_dict2device(X_batch,device)
Y_batch = move_dict2device(Y_batch,device)

pred = IR_Model0(X_batch)
Affine_mtrx = pred['Affine_mtrx']



Learning_rate = 0.001
print_every = int(2000*80/batch_size)
optimizer = optim.AdamW(IR_Model0.parameters(), lr=Learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
source = X_batch['source']
target = X_batch['target']
source_origion = X_batch['source_origion'].detach()
best_loss = 100000000000000000000
for i in range(300):
    optimizer.zero_grad()
    predections_j = IR_Model0(X_batch)
    Affine_matrix0 = predections_j['Affine_mtrx']
    grd = torch.nn.functional.affine_grid(Affine_matrix0, size=source_origion.shape,align_corners=False)
    source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                            mode='bilinear', padding_mode='zeros', align_corners=False)
    wrapped_img = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
    loss_item = {}
    loss = MSE_loss(target[target>0.05].detach(), wrapped_img[target>0.05])
    if loss.item()<best_loss:
        best_loss = loss
        predction = predections_j
        print('Pxl loss: ',best_loss)
    loss.backward()
    optimizer.step()
    if with_scheduler:
            scheduler.step(loss)




MSE_loss = torch.nn.functional.mse_loss
class Build_ِActive_IRmodel(torch.nn.Module):
    def __init__(self,IR_Model0, Number_updates = 12):
        super(Build_ِActive_IRmodel, self).__init__()
        self.IR_Model0 = IR_Model0
        self.Number_updates = Number_updates
    def forward(self, input_X_batch):
        Learning_rate = 0.0001
        print_every = int(2000*80/batch_size)
        optimizer = optim.AdamW(IR_Model0.parameters(), lr=Learning_rate)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        source = input_X_batch['source']
        target = input_X_batch['target']
        source_origion = input_X_batch['source_origion'].detach()
        best_loss = 100000000000000000000
        for i in range(self.Number_updates):
            optimizer.zero_grad()
            predections_j = self.IR_Model0(input_X_batch)
            Affine_matrix0 = predections_j['Affine_mtrx']
            grd = torch.nn.functional.affine_grid(Affine_matrix0, size=source_origion.shape,align_corners=False)
            source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
            wrapped_img = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
            loss_item = {}
            loss = MSE_loss(target[target>0.05].detach(), wrapped_img[target>0.05])
            if loss.item()<best_loss:
                best_loss = loss.detach().item()
                predction = predections_j
                print('Pxl loss: ',best_loss)
            loss.backward()
            optimizer.step()
            #if with_scheduler:
            #        scheduler.step(loss)
        return predction

test_loss(IR_Model0, testloader0, max_iterations=100, key = 'Affine_mtrx')

Active_IRmodel = Build_ِActive_IRmodel(IR_Model0, Number_updates = 30)

test_loss(Active_IRmodel, testloader0, max_iterations=10, key = 'Affine_mtrx')











#---------------Test--------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
    #Load best model saved on drive
ext = '_EndTraining'#'_EndTraining' #_bestVal

if Arch == 'ResNet':
    core_model_tst = resnet18(weights='ResNet18_Weights.DEFAULT')#pretrained=True)#weights='ResNet18_Weights.IMAGENET1K_V1')
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
    core_model_tst.load_state_dict(torch.load(savingfolder+'core_model'+ext+'.pth'))
    IR_Model_tst = Build_IRmodel_Resnet(core_model_tst, registration_method,Glopooling,LYR_NORM=LYR_NORM, BW_Position=False, overlap=overlap)

if 'VIT' in Arch:
    if 'mobileVIT' in Arch:
        from transformers import MobileViTV2ForImageClassification
        core_model_tst = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        if overlap=='vertical':
            original_conv1 = core_model_tst.mobilevitv2.conv_stem.convolution
            new_conv1 = torch.nn.Conv2d(6,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # Initialize the new convolutional layer's weights
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3:, :, :] = original_conv1.weight
            core_model_tst.mobilevitv2.conv_stem.convolution = new_conv1
            core_model_tst.classifier = torch.nn.Linear(512, 6)
    elif Arch == 'rawVIT':
        core_model_tst = rawViT(img_size=128,in_channels=6,patch_size=12,embed_dim=384,hidden_dim=768,num_heads=12,num_layers=4, num_classes=6)
    IR_Model_tst = Build_IRmodel_ViT(core_model_tst,registration_method = registration_method, overlap=overlap, Arch = Arch)

core_model_tst.to(device)
IR_Model_tst.load_state_dict(torch.load(savingfolder+'IR_Model'+ext+'.pth'))
IR_Model_tst.to(device)
IR_Model_tst.eval()


#----------------------------
#Load test dataset

test_set = Dataset(list_paths=routes_source, batch_size=batch_size, DATASET=DATASET_generation_split,Img_noise= True,
            registration_method =registration_method,folder_prefx = folder_prefx, mode='test')
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)


test_set0 = Dataset(list_paths=routes_source, batch_size=batch_size, DATASET=DATASET_generation_split,Img_noise= False,
            registration_method =registration_method,folder_prefx = folder_prefx, mode='test')
testloader0 = torch.utils.data.DataLoader(test_set0, batch_size=batch_size,shuffle=False)


savingfolder_test = savingfolder+'test/'
os.system('mkdir '+ savingfolder_test)
os.system('mkdir '+ savingfolder_test+ 'MIRexamples')

savingfolder = savingfolder_test


SWITCH = False
def Load_augment(image_path, dim = 128): #measures=['angle', 'scale', 'translationX','translationY','shearx','sheary'], Noise_level=0,
  #enlargement = crop_ratio
  Outputs= {}
  #image0 = load_image_from_url(image_path, int(enlargement*dim))
  #image0 = (transforms.ToTensor()(image0)).unsqueeze(0).to(torch.float32)
  image0 = load_image_pil_accelerated(image_path).unsqueeze(0)
  transformed_img0, Affine_mtrx, Affine_parameters = augment_img(image0)#, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx)
  image1 = torchvision.transforms.CenterCrop((dim, dim))(image0)
  transformed_img = torchvision.transforms.CenterCrop((dim, dim))(transformed_img0)
  if SWITCH:
    Switch_threshold = 0.5
  else:
    Switch_threshold = 10
  if np.random.rand()<=Switch_threshold:
      Outputs['Affine_mtrx'] = Affine_mtrx[0]
      Outputs['source_origion'] = image0[0]
      Outputs['source'] = image1[0]
      Outputs['target'] = transformed_img[0]
      Outputs['Affine_parameters'] = Affine_parameters[0]
  else:
      Outputs['Affine_mtrx'] = inv_AM(Affine_mtrx)[0].detach()
      Outputs['source_origion'] = transformed_img0[0].detach()
      Outputs['source'] = transformed_img[0].detach()
      Outputs['target'] = image1[0].detach()
  return Outputs
#----------------------------

##------------------------------------------------------------------
## Plot deviation graph of each augmentation parameter from the the ground-truth
##------------------------------------------------------------------
# 1- fix three parameters and sample from one
# 2- plot the distrinution of the error for the variable parameter
# fixed values s =1, angle =0, translation =0
# MSE_AffineMatrix_1measure
#test_routes = glob.glob('../../localdb/224/test/'+"**/*.JPEG", recursive = True)
Measures_list = Intitial_Tx#Intitial_Tx = ['angle', 'scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflectionX', 'reflectionY']
num2txt= {  '0': 'angle', '1': 'scaleX_inv','2': 'scaleY_inv', '3': 'translationX','4': 'translationY', '5': 'shearx','6': 'sheary', '7': 'reflectionX','8': 'reflectionY'}
txt2num ={'angle':0, 'scaleX_inv':1,'scaleY_inv':1, 'translationX':3, 'translationY':4, 'shearx':5,'sheary':6, 'reflectionX':7, 'reflectionY':8}

MSE_loss = torch.nn.functional.mse_loss


assert Noise_level_dataset>0
def augment_img(image0, NOISE_LEVEL=Noise_level_dataset, MODE='bilinear', ONE_MESURE=Intitial_Tx, block_min =0, block_max=1): #,'shearx','sheary'
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
    return wrapped_img, Affine_mtrx, Affine_parameters

MSE_affine = test_loss(IR_Model_tst, testloader0, 100, key = 'Affine_mtrx').detach().item()
print(MSE_affine)
json.dump( MSE_affine, open( savingfolder+'Test_loss_noNoise_sigma0.5.txt', 'w' ) )

if IMG_noise:
    MSE_affine = test_loss(IR_Model_tst, testloader,100, key = 'Affine_mtrx').detach().item()
    print(MSE_affine)
    json.dump( MSE_affine, open( savingfolder+'Test_loss_withNoise_sigma0.5.txt', 'w' ) )

Noise_level_testset=0
def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=Intitial_Tx, block_min =0, block_max=1): #,'shearx','sheary'
    wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
    return wrapped_img, Affine_mtrx, Affine_parameters

MSE_affine = test_loss(IR_Model_tst, testloader0,100, key = 'Affine_mtrx').detach().item()
print(MSE_affine)
json.dump( MSE_affine, open( savingfolder+'Test_loss_noNoise_sigma0.txt', 'w' ) )

save_examples(IR_Model_tst,testloader0, n_examples = 6, plt_markrs=True,plt_imgs=True,time=6, feed_origion=True, shadow=True, win=9)


#----------------------------------------------------------
#--------------- Evaluation of basic transformations --------------------------------
#----------------------------------------------------------

##------------------------------------------------------------------
## Evaluation meaures under smaller domain of basic transformation (variant test set difficulties) for all transformations
##------------------------------------------------------------------
Measures_list = Intitial_Tx#['angle', 'scaleX','scaleY','translationX','translationY','shearX','shearY', 'reflectionX', 'reflectionY']
BLOCK_options = {'Full':(0,1), 'Hard70-100%':(0.7,1), 'Medium30-70':(0.3,0.7), 'Easy10-30%':(0.1,0.3), 'Easier5-10%':(0.05,0.1), 'Easiest0-5%':(0,0.05)}

Noise_level_testset = 0
MSE_AffineMatrix_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=Measures_list, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_difficulty[Difficulty] = test_loss(IR_Model_tst,testloader0,100, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_difficulty)
json.dump(MSE_AffineMatrix_difficulty, open( savingfolder+'MSE_AffineMatrix_VariantDifficulty_AllTx_Sigma{}.txt'.format(Noise_level_testset), 'w' ) )




MSE_AffineMatrix_1measure = {}
for MEASURE in Measures_list:
    MEASURES= [MEASURE]
    Noise_level_testset = 0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =0, block_max=1): #,'shearx','sheary'
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_1measure[MEASURE] = test_loss(IR_Model_tst,testloader0,100, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_1measure)
json.dump(MSE_AffineMatrix_1measure, open( savingfolder+'Test_loss_1Transformation.txt', 'w' ) )

##------------------------------------------------------------------
## Evaluation meaures under smaller domain of basic transformation (variant test set difficulties) for basic transformations
##------------------------------------------------------------------

MSE_AffineMatrix_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    MSE_AffineMatrix_1measure = {}
    for MEASURE in Measures_list:
        MEASURES= [MEASURE]
        Noise_level_testset=0
        def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
            return wrapped_img, Affine_mtrx, Affine_parameters
        MSE_AffineMatrix_1measure[MEASURE] = test_loss(IR_Model_tst,testloader0,100, key = 'Affine_mtrx').detach().item()
    MSE_AffineMatrix_difficulty[Difficulty] =MSE_AffineMatrix_1measure

print(MSE_AffineMatrix_difficulty)
json.dump(MSE_AffineMatrix_difficulty, open( savingfolder+'MSE_AffineMatrix_VariantDifficulty_BasicTx.txt', 'w' ) )




MSE_AffineMatrix_1measure = {}
for MEASURE in Measures_list:
    MEASURES= [MEASURE]
    Noise_level_testset=0.5
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =0, block_max=1): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_1measure[MEASURE] = test_loss(IR_Model_tst,testloader0,100, key = 'Affine_mtrx').detach().item()

print(MSE_AffineMatrix_1measure)
json.dump(MSE_AffineMatrix_1measure, open( savingfolder+'MSE_AffineMatrix_Fulldataset_Sigma0.5_basicTx.txt', 'w' ) )




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

Marker_measures = evaluate_model(IR_Model, testloader0, max_iterations = 100, measures = { 'AM_MSE':True, 'Marker_RMSE':True, 'Marker_HD95':True})
json.dump( Marker_measures, open( savingfolder+'Marker_measures.txt', 'w' ) )
















##------------------------------------------------------------------
## Recurrent estimation of the Affine matrix
##------------------------------------------------------------------


def plot_examples(source,target,wrapped_img,AM_Predicted=0,AM_Previous=0, AM_GroundTruth=0, n_examples = 4, iteration = 0, prefix= 'randomdifficulty' ,plt_markrs=True,plt_imgs=True, shadow=True):
    #prepare figures
    if shadow:
        plots_per_example = 4
    else:
        plots_per_example = 3
    figure = plt.figure()
    gs = figure.add_gridspec(n_examples, plots_per_example, hspace=0.05, wspace=0)
    axis = gs.subplots(sharex='col', sharey='row')
    figure.set_figheight(5*n_examples)
    figure.set_figwidth(5*plots_per_example)
    with torch.no_grad():
        if shadow:
            Error_map = torch.abs(target.detach()-wrapped_img.detach())
        for k in range(n_examples):
            if plt_markrs:
                M_GroundTruth = workaround_matrix(AM_GroundTruth.detach(), acc = 0.5/crop_ratio)
                M_Predicted = workaround_matrix(AM_Predicted.detach(), acc = 0.5/crop_ratio)
                M_Previous = workaround_matrix(AM_Previous.detach(), acc = 0.5/crop_ratio)
                x0, y0 = generate_standard_mark()
                x0_target, y0_target = transform_standard_points(M_GroundTruth[k], x0, y0)
                x0_source, y0_source = transform_standard_points(M_Previous[k], x0, y0)
                x0_transformed, y0_transformed = transform_standard_points(M_Predicted[k], x0_source, y0_source)
                #Another way of finding the transformed marker
                #M_accumilative = torch.matmul(mtrx3(M_Predicted), mtrx3(M_Previous))[:,0:2,:]
                #x0_transformed, y0_transformed = transform_standard_points(M_accumilative[k], x0, y0)
                #--------------destandarize for plotting purposes-------------
                x_source = destandarize_point(x0_source, dim=dim, flip = False)
                x_target = destandarize_point(x0_target, dim=dim, flip = False)
                x_transformed = destandarize_point(x0_transformed, dim=dim, flip = False)
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
                axis[k, 3].imshow(torchvision.transforms.ToPILImage()(Error_map[k]),alpha=0.9)
        for ax in figure.get_axes():
            ax.label_outer()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    suffix = ''
    if plt_markrs:
            suffix += 'Elipse'
    if plt_imgs:
        suffix += 'Img'
    plt.savefig(savingfolder+'MIRexamples/{}_{}_{}.jpeg'.format(prefix, suffix,iteration), bbox_inches='tight')
    plt.close()



def AM_recurrent_loss(model,loader , max_iterations=2, No_recurences = 3, key ='Affine_mtrx', plot=True, plot_batach = 1, prefix = 'debug'):
    AM_MSE_tot = {}
    AM_MSE_avg = {}
    model = model.eval()
    for j in range(No_recurences):
        AM_MSE_tot[str(j)]=0
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
                inputs_j = inputs
                Affine_mtrx_j = predections['Affine_mtrx'].detach()
                Accumilative_Affine_matrix = torch.zeros_like(Affine_mtrx_j).detach()
                Accumilative_Affine_matrix[:,0,0]=1.
                Accumilative_Affine_matrix[:,1,1]=1.
                New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix.detach()), mtrx3(Affine_mtrx_j.detach()))[:,0:2,:]
                AM_MSE_tot[str(0)] += MSE_loss(labels[key].detach(), New_Accumilative_Affine_matrix.detach()).item()
                grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion.shape,align_corners=False)
                source_origion_224j = torch.nn.functional.grid_sample(source_origion, grid=grd,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                wrapped_img = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
                if plot:
                    if i== plot_batach:
                        '''
                        #debug:
                        source0_plt = torchvision.transforms.ToPILImage()(source0[0])
                        source0_plt.save(savingfolder+'MIRexamples/0_debug_source{}_iteration{}.jpeg'.format(i, 0))
                        wrapped_img_plt = torchvision.transforms.ToPILImage()(wrapped_img[0])
                        wrapped_img_plt.save(savingfolder+'MIRexamples/0_debug_warped{}_iteration{}.jpeg'.format(i, j))
                        #'''
                        plot_examples(source0,target,wrapped_img, AM_Predicted=Affine_mtrx_j,AM_Previous=Accumilative_Affine_matrix, AM_GroundTruth=labels[key], n_examples = 4, iteration = 0, plt_markrs=True, prefix= prefix)
                for j in range(1, No_recurences):
                    inputs_j ={'source': wrapped_img,
                                'source_origion': source_origion,
                                'target': target,
                                'M_i': torch.zeros([int(inputs['source'].shape[0]),2,3]).to(device)
                                }
                    predections = model(inputs_j)
                    Affine_mtrx_j = predections['Affine_mtrx'].detach()
                    Accumilative_Affine_matrix = New_Accumilative_Affine_matrix.detach()
                    New_Accumilative_Affine_matrix = torch.matmul(mtrx3(Accumilative_Affine_matrix.detach()), mtrx3(Affine_mtrx_j.detach()))[:,0:2,:]
                    '''
                    grd = torch.nn.functional.affine_grid(Affine_mtrx_j, size=source_origion_224j.shape,align_corners=False)
                    source_origion_224j = torch.nn.functional.grid_sample(source_origion_224j, grid=grd,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
                    wrapped_img_iteratively = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j)
                    '''
                    grd_acc = torch.nn.functional.affine_grid(New_Accumilative_Affine_matrix, size=source_origion.shape,align_corners=False)
                    source_origion_224j_acc = torch.nn.functional.grid_sample(source_origion.detach(), grid=grd_acc,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
                    wrapped_img_acc = torchvision.transforms.CenterCrop((dim, dim))(source_origion_224j_acc)
                    wrapped_img = wrapped_img_acc.detach()
                    if plot:
                        if i== plot_batach:
                            plot_examples(inputs_j['source'],target,wrapped_img,AM_Predicted=Affine_mtrx_j, AM_Previous=Accumilative_Affine_matrix, AM_GroundTruth=labels[key], n_examples = 4, iteration = j, plt_markrs=True, prefix= prefix)
                            '''#debug:
                            wrapped_img_plt = torchvision.transforms.ToPILImage()(wrapped_img[0])
                            wrapped_img_plt.save(savingfolder+'MIRexamples/0_debug_warped{}_iteration{}.jpeg'.format(i, j))
                            grd_acc = torch.nn.functional.affine_grid(New_Accumilative_Affine_matrix, size=source_origion.shape,align_corners=False)
                            Accumilative_img_224j = torch.nn.functional.grid_sample(source_origion.detach(), grid=grd_acc,
                                                    mode='bilinear', padding_mode='zeros', align_corners=False)
                            Accumilative_img = torchvision.transforms.CenterCrop((dim, dim))(Accumilative_img_224j)
                            Accumilative_img_plt = torchvision.transforms.ToPILImage()(Accumilative_img[0])
                            Accumilative_img_plt.save(savingfolder+'MIRexamples/0_debug_Accumilative_img{}_iteration{}.jpeg'.format(i, j))
                            #'''
                    AM_MSE_tot[str(j)] += MSE_loss(labels[key].detach(), New_Accumilative_Affine_matrix.detach()).item()
            else:
                for r in range(No_recurences):
                    #eval_loss_avg[str(r)] = eval_loss_tot[str(r)]/max_iterations
                    AM_MSE_avg[str(r)] = np.round(AM_MSE_tot[str(r)]/max_iterations, 4)
                return AM_MSE_avg

AM_recurrent_loss(IR_Model_tst,testloader0 , max_iterations=100, No_recurences = 3, key ='Affine_mtrx', plot=True, plot_batach = 5, prefix = 'randomdifficulty')
#debug_recurrence(IR_Model_tst,testloader0 , max_iterations=2, No_recurences = 3, key ='Affine_mtrx', plot=True, plot_batach = 1, prefix = 'debug')
#AM_recurrent_loss(IR_Model_tst,testloader0 , max_iterations=3, No_recurences = 3, key ='Affine_mtrx', plot=True, plot_batach = 1, prefix = '0Debug')



MSE_AffineMatrix_recurrent_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    MEASURES= Intitial_Tx
    Noise_level_testset=0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_recurrent_difficulty[Difficulty] = AM_recurrent_loss(IR_Model_tst,testloader0, max_iterations=70,
                                                                            No_recurences = 3, key ='Affine_mtrx', plot=True, plot_batach = 5, prefix = Difficulty)

print(MSE_AffineMatrix_recurrent_difficulty)
json.dump(MSE_AffineMatrix_recurrent_difficulty, open( savingfolder+'MSE_AffineMatrix_recurrent_difficulty.txt', 'w' ) )





MSE_AffineMatrix_recurrent_difficulty = {}
for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    MEASURES= Intitial_Tx
    Noise_level_testset=0
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE=MEASURES, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
    MSE_AffineMatrix_recurrent_difficulty[Difficulty] = AM_recurrent_loss(IR_Model_tst,testloader, max_iterations=70,
                                                                            No_recurences = 3, key ='Affine_mtrx', plot=True, plot_batach = 5, prefix = Difficulty)

print(MSE_AffineMatrix_recurrent_difficulty)
json.dump(MSE_AffineMatrix_recurrent_difficulty, open( savingfolder+'MSE_AffineMatrix_recurrent_difficulty_with_img_noise.txt', 'w' ) )







'''
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

'''
#wrapped_img = wrap_imge_uncropped(Affine_mtrx, source_origion)















#Affine_mtrx_j = torch.matmul( mtrx3(predections[key]), mtrx3(Affine_mtrx_j))[:,:2,:]
#Affine_mtrx_j = torch.matmul(mtrx3(Affine_mtrx_j), mtrx3(predections[key]))[:,:2,:]
#Affine_mtrx_j =torch.matmul(mtrx3(Affine_mtrx_j).permute(0,2,1), mtrx3(predections[key]).permute(0,2,1)).permute(0,1,2)[:,:2,:]
#Affine_mtrx_j =((torch.matmul(mtrx3(Affine_mtrx_j).T, mtrx3(predections[key]).T)).T)[:,:2,:]
#print(Affine_mtrx_j.shape)

NCC_loss = Torch_NCC(win=128,stride = 128).loss

pad=32
feed_origion = True
def recurrent_loss(model,loader , max_iterations=100, No_recurences = 3, key ='Affine_mtrx'):
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
                                'source_origion': source_origion,
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



No_recurences = 4
Noise_level_testset=0.0
Pixels_MSE_AffineMatrix_difficulty = {}
Pixels_NCC_AffineMatrix_difficulty = {}

for Difficulty in BLOCK_options.keys():
    Block_min_difficulty, Block_max_difficulty= BLOCK_options[Difficulty]
    def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx, block_min =Block_min_difficulty, block_max=Block_max_difficulty): 
            wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
            return wrapped_img, Affine_mtrx, Affine_parameters
    Pixels_MSE_AffineMatrix_difficulty[Difficulty], Pixels_NCC_AffineMatrix_difficulty[Difficulty] = recurrent_loss(model=IR_Model_tst,loader=testloader0, max_iterations=100, No_recurences = No_recurences)

json.dump( Pixels_NCC_AffineMatrix_difficulty, open( savingfolder+'Recurrent_PXL_NCC_difficulties.txt', 'w' ) )

print('Recurrent PXL_MSE_avg when sigma=0.0:', Pixels_MSE_AffineMatrix_difficulty)
print('Recurrent PXL_NCC_avg when sigma=0.0:', Pixels_NCC_AffineMatrix_difficulty)
json.dump( Pixels_MSE_AffineMatrix_difficulty, open( savingfolder+'Recurrent_PXL_MSE_difficulties.txt', 'w' ) )
json.dump( Pixels_NCC_AffineMatrix_difficulty, open( savingfolder+'Recurrent_PXL_NCC_difficulties.txt', 'w' ) )

save_examples(IR_Model_tst,testloader0, n_examples = 6, plt_markrs=True,plt_imgs=True,time=6, feed_origion=True, shadow=True, win=5)



#------------------------------------------
No_recurences = 3
Noise_level_testset=0.0
def augment_img(image0, NOISE_LEVEL=Noise_level_testset, MODE='bilinear', ONE_MESURE= Intitial_Tx, block_min =BLOCK_MIN, block_max=BLOCK_MAX): 
        wrapped_img, Affine_mtrx, Affine_parameters = pass_augment_img(image0, measure =ONE_MESURE, MODE=MODE, NOISE_LEVEL=NOISE_LEVEL, block_min =block_min, block_max=block_max)
        return wrapped_img, Affine_mtrx, Affine_parameters
PXL_MSE_avg, PXL_NCC_avg = recurrent_loss(model=IR_Model_tst,loader=testloader0, max_iterations=100, No_recurences = No_recurences)
print('Recurrent PXL_MSE_avg when Noise_level_testset=0.5:', PXL_MSE_avg)
print('Recurrent PXL_NCC_avg when Noise_level_testset=0.5:', PXL_NCC_avg)
json.dump( PXL_MSE_avg, open( savingfolder+'Recurrent_PXL_MSE_avg.txt', 'w' ) )
json.dump( PXL_NCC_avg, open( savingfolder+'Recurrent_PXL_NCC_avg.txt', 'w' ) )




import os
import json
import subprocess
import time
import psutil


def get_kernel_info():
    return {
        "kernel_version": os.uname().release,
        "system_name": os.uname().sysname,
        "node_name": os.uname().nodename,
        "machine": os.uname().machine
    }

def get_memory_info():
    return {
        "total_memory": psutil.virtual_memory().total / (1024.0 ** 3),
        "available_memory": psutil.virtual_memory().available / (1024.0 ** 3),
        "used_memory": psutil.virtual_memory().used / (1024.0 ** 3),
        "memory_percentage": psutil.virtual_memory().percent
    }

def get_cpu_info():
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "processor_speed": psutil.cpu_freq().current,
        "cpu_usage_per_core": dict(enumerate(psutil.cpu_percent(percpu=True, interval=1))),
        "total_cpu_usage": psutil.cpu_percent(interval=1)
    }

def get_disk_info():
    partitions = psutil.disk_partitions()
    disk_info = {}
    for partition in partitions:
        partition_usage = psutil.disk_usage(partition.mountpoint)
        disk_info[partition.mountpoint] = {
            "total_space": partition_usage.total / (1024.0 ** 3),
            "used_space": partition_usage.used / (1024.0 ** 3),
            "free_space": partition_usage.free / (1024.0 ** 3),
            "usage_percentage": partition_usage.percent
        }
    return disk_info

def get_network_info():
    net_io_counters = psutil.net_io_counters()
    return {
        "bytes_sent": net_io_counters.bytes_sent,
        "bytes_recv": net_io_counters.bytes_recv
    }

def get_process_info():
    process_info = []
    for process in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
        try:
            process_info.append({
                "pid": process.info['pid'],
                "name": process.info['name'],
                "memory_percent": process.info['memory_percent'],
                "cpu_percent": process.info['cpu_percent']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_info

def get_load_average():
    load_avg_1, load_avg_5, load_avg_15 = psutil.getloadavg()
    return {
        "load_average_1": load_avg_1,
        "load_average_5": load_avg_5,
        "load_average_15": load_avg_15
    }
    
def get_disk_io_counters():
    io_counters = psutil.disk_io_counters()
    return {
        "read_count": io_counters.read_count,
        "write_count": io_counters.write_count,
        "read_bytes": io_counters.read_bytes,
        "write_bytes": io_counters.write_bytes,
        "read_time": io_counters.read_time,
        "write_time": io_counters.write_time
    }
    
def get_net_io_counters():
    io_counters = psutil.net_io_counters()
    return {
        "bytes_sent": io_counters.bytes_sent,
        "bytes_recv": io_counters.bytes_recv,
        "packets_sent": io_counters.packets_sent,
        "packets_recv": io_counters.packets_recv,
        "errin": io_counters.errin,
        "errout": io_counters.errout,
        "dropin": io_counters.dropin,
        "dropout": io_counters.dropout
    }

def get_system_uptime():
    boot_time_timestamp = psutil.boot_time()
    current_time_timestamp = time.time()
    uptime_seconds = current_time_timestamp - boot_time_timestamp
    uptime_minutes = uptime_seconds // 60
    uptime_hours = uptime_minutes // 60
    uptime_days = uptime_hours // 24
    uptime_str = f"{int(uptime_days)} days, {int(uptime_hours % 24)} hours, {int(uptime_minutes % 60)} minutes, {int(uptime_seconds % 60)} seconds"
    return {"uptime": uptime_str}


if __name__ == '__main__':
    data = {
        "kernel_info": get_kernel_info(),
        "memory_info": get_memory_info(),
        "cpu_info": get_cpu_info(),
        "disk_info": get_disk_info(),
        "network_info": get_network_info(),
        "process_info": get_process_info(),
        "system_uptime": get_system_uptime(),
        "load_average": get_load_average(),
        "disk_io_counters": get_disk_io_counters(),
        "net_io_counters": get_net_io_counters(),
    }
    print("="*40)
    print("System Monitoring")
    print("="*40)
    print(json.dumps(data, indent=4))