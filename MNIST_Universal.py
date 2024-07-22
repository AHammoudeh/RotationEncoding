


#Classification
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm
import os

savingfolder = '/home/ahmadh/MIR_savedmodel/MNIST_classification/'
os.mkdir(savingfolder)


rotation_step =15
rotation_angles = list(range(0, 360, rotation_step))  # Angles from 0 to 345 degrees in 15-degree increments
N_digits = 10
N_classes = int(360/rotation_step)

# Define a function to rotate images and create labels for the rotation angles
class RotatedMNIST(Dataset):
   def __init__(self, dataset, angles):
       self.dataset = dataset
       self.angles = angles
   def __len__(self):
       return len(self.dataset)
   def __getitem__(self, idx):
       img, digit = self.dataset[idx]
       angle = np.random.choice(self.angles).astype(int)
       #angle = torch.tensor(angle, dtype=torch.float32)
       rotated_img = transforms.functional.rotate(img, int(angle))
       return rotated_img, angle/360, digit

# Define a Convolutional Neural Network for regression
class AnglePredictorCNN(nn.Module):
   def __init__(self, N_classes):
       super(AnglePredictorCNN, self).__init__()
       self.N_classes = N_classes
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       self.fc1 = nn.Linear(64 * 7 * 7, 128)
       self.fc2 = nn.Linear(128, self.N_classes)
   def forward(self, x):
       x = self.pool(torch.relu(self.conv1(x)))
       x = self.pool(torch.relu(self.conv2(x)))
       x = x.view(-1, 64 * 7 * 7)
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x#.squeeze()  # Remove unnecessary dimensions

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
   model.train()
   for epoch in range(num_epochs):
       running_loss = 0.0
       for images, angles in train_loader:
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, angles)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()
       print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Testing loop
def test(model, test_loader, criterion):
   model.eval()
   total_loss = 0.0
   with torch.no_grad():
       for images, angles in test_loader:
           outputs = model(images)
           loss = criterion(outputs, angles)
           total_loss += loss.item()
   print(f'Average test loss: {total_loss/len(test_loader):.4f}')


# Hyperparameters
batch_size = 64
# Rotation angles for training
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_rotated_dataset = RotatedMNIST(train_dataset, rotation_angles)
train_loader = DataLoader(train_rotated_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
test_rotated_dataset = RotatedMNIST(test_dataset, rotation_angles)
test_loader = DataLoader(test_rotated_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, define the loss function and the optimizer
model = AnglePredictorCNN(N_classes)
criterion = torch.nn.CrossEntropyLoss()#nn.MSELoss()
learning_rate = 0.001
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)



def one_hot(x,n_classes=24):
    x1h = torch.eye(n_classes)
    return x1h[x]

num_epochs = 50
print_every = 100
model.train()#.to(device)
for epoch in range(num_epochs):
   running_loss = 0.0
   j = 0
   for images, angles, digit in tqdm.tqdm(train_loader):
       j+=1
       optimizer.zero_grad()
       outputs = model(images)
       #pred = torch.argmax(outputs, axis=-1)
       #convert an angle into a class label
       angle_class = (N_classes*angles%N_classes).to(torch.int)
       # convert class label into 1-hot encoding
       angle_class_1h = one_hot(angle_class, N_classes)
       #loss = criterion(outputs, angles.to(torch.float32))
       loss = criterion(outputs, angle_class_1h)
       loss.backward()
       optimizer.step()
       running_loss += loss.item()
       if j%print_every == 0:
         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/j:.4f}')
   print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), savingfolder+'./model.pth')


model.eval()
conf_mtrx = np.zeros([N_digits,N_classes,N_classes])
prob_mtrx = torch.zeros([N_digits,N_classes, N_classes])
N_digits_measured = torch.zeros([N_digits,N_classes, 1])
softmx = torch.nn.Softmax(dim=0)
with torch.no_grad():
    for images, angles, digits in tqdm.tqdm(test_loader):
       outputs = model(images)
       pred_class = torch.argmax(outputs, axis=-1)
       GT_class = (N_classes*angles)%N_classes
       #angle_class = (N_classes*angles%N_classes).to(torch.int)
       #angle_class_1h = one_hot(angle_class)
       for item in range(outputs.shape[0]):
            conf_mtrx[digits[item].item(),int(GT_class[item]), int(pred_class[item])] +=1
            prob_mtrx[digits[item].item(),int(GT_class[item])] += softmx(outputs[item,:].detach())
            N_digits_measured[digits[item].item(),int(GT_class[item])]+=1

prob_mtrx /= N_digits_measured
prob_mtrx = np.around(prob_mtrx.numpy(),3)
prob_mtrx.tofile( savingfolder+f'prob_mtrx.txt', sep=', ', format='%.3f')

for digit in range(10):
    conf_mtrx[digit].tofile( savingfolder+f'cm{digit}.txt', sep=', ', format='%.03d')
    prob_mtrx.tofile( savingfolder+f'prob_matrx{digit}.txt', sep=', ', format='%.3f')

cm = np.sum(conf_mtrx, axis=0)
cm.tofile( savingfolder+f'cm_total.txt', sep=',  ', format='%.03d')

pm = np.sum(prob_mtrx/10, axis=0)
pm.tofile( savingfolder+f'prob_mtrx.txt', sep=', ', format='%.3f')



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


plot_table(prob_mtrx[], savingfolder+f'prob_mtrx.png')


prob_mtrx.tofile( savingfolder+f'prob_mtrx.txt', sep=', ', format='%.3f')

for digit in range(10):
    plot_table(prob_mtrx[digit], savingfolder+f'prob_matrx{digit}.png')
    plot_table(conf_mtrx[digit], savingfolder+f'conf_mtrx{digit}.png')


    conf_mtrx[digit].tofile( savingfolder+f'cm{digit}.txt', sep=', ', format='%.03d')
    prob_mtrx[digit].tofile( savingfolder+f'prob_matrx{digit}.txt', sep=', ', format='%.3f')

cm = np.sum(conf_mtrx, axis=0)
cm.tofile( savingfolder+f'cm_total.txt', sep=',  ', format='%.03d')
pm = np.sum(prob_mtrx/10, axis=0)
pm.tofile( savingfolder+f'prob_mtrx.txt', sep=', ', format='%.3f')

plot_table(cm, savingfolder+f'conf_mtrx_total.png')
plot_table(pm, savingfolder+f'prob_matrx_total.png')



test(model, test_loader, criterion)


# Function to visualize rotated images with their angles
def visualize_rotated_images(dataset, n_images=5):
   fig, axes = plt.subplots(1, n_images, figsize=(15, 15))
   for i in range(n_images):
       image, angle = dataset[i]
       axes[i].imshow(image.squeeze(), cmap='gray')
       axes[i].set_title(f'Angle: {angle}°')
       axes[i].axis('off')
   plt.show()

# Visualize some rotated images with their angles
visualize_rotated_images(train_rotated_dataset)







































































#Regrression
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm
import os

device_num = 0
torch.cuda.set_device(device_num)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

savingfolder = '/home/ahmadh/MIR_savedmodel/MNIST_Regression/'
os.mkdir(savingfolder)

# Define a function to rotate images and create labels for the rotation angles
class RotatedMNIST(Dataset):
    def __init__(self, dataset, rotation_step, method='regression'):
        self.dataset = dataset
        self.rotation_step = rotation_step
        self.angles = list(range(0, 360, self.rotation_step))
        self.method = method
    def __len__(self):
       return len(self.dataset)
    def __getitem__(self, idx):
       img, label = self.dataset[idx]
       angle = np.random.choice(self.angles).astype(int)
       #angle = torch.tensor(angle, dtype=torch.float32)
       rotated_img = transforms.functional.rotate(img, int(angle))
       angle_norm = angle/360
       if self.method=='classification':
        label = angle_norm//self.rotation_step
       else:
        label = angle_norm
       return rotated_img, label


# Define a Convolutional Neural Network for regression
class AnglePredictorCNN(nn.Module):
   def __init__(self, rotation_step, method = 'regression'):
       super(AnglePredictorCNN, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       self.fc1 = nn.Linear(64 * 7 * 7, 128)
       if method == 'regression':
        self.fc2 = nn.Linear(128, 1)
       else:
        self.fc2 = nn.Linear(128, int(360//rotation_step))
   def forward(self, x):
       x = self.pool(torch.relu(self.conv1(x)))
       x = self.pool(torch.relu(self.conv2(x)))
       x = x.view(-1, 64 * 7 * 7)
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x.squeeze()  # Remove unnecessary dimensions


# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
   model.train()
   for epoch in range(num_epochs):
       running_loss = 0.0
       for images, angles in train_loader:
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, angles)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()
       print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Testing loop
def test(model, test_loader, criterion):
   model.eval()
   total_loss = 0.0
   with torch.no_grad():
       for images, angles in test_loader:
           outputs = model(images)
           loss = criterion(outputs, angles)
           total_loss += loss.item()
   print(f'Average test loss: {total_loss/len(test_loader):.4f}')



# Hyperparameters
batch_size = 64
learning_rate = 0.001
rotation_step = 15
METHOD = 'regression' #'classification'

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_rotated_dataset = RotatedMNIST(train_dataset, rotation_step, method=METHOD)
train_loader = DataLoader(train_rotated_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
test_rotated_dataset = RotatedMNIST(test_dataset, rotation_step, method=METHOD)
test_loader = DataLoader(test_rotated_dataset, batch_size=100*batch_size, shuffle=True)

# Instantiate the model, define the loss function and the optimizer
model = AnglePredictorCNN(rotation_step, method = METHOD)
if METHOD == 'regression':
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# training
num_epochs = 5
print_every = 300
training_loss_iterationlist =[]
validation_loss_iterationlist =[]
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    running_val_loss =0.0
    j = 0
    for images, angles in tqdm.tqdm(train_loader):
        j+=1
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles.to(torch.float32))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if j%print_every == 0:
            training_loss_iterationlist.append(round(running_loss/j,4))
            print(f'Epoch [{j+1}/{num_epochs}], Loss: {running_loss/j:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    images_val, angles_val = next(iter(test_loader))
    '''
    outputs_val = model(images_val)
    running_val_loss = criterion(outputs_val, angles_val.to(torch.float32))
    validation_loss_iterationlist.append(round(running_val_loss.detach().item()/j,4))
    '''
    plt.plot(training_loss_iterationlist, label = 'training loss')
    plt.plot(validation_loss_iterationlist, label = 'validation loss')
    #plt.ylim(0,0.5)
    plt.legend()
    plt.savefig(savingfolder+'loss_iterations.png', bbox_inches='tight')
    plt.close()
    np.savetxt(savingfolder+'training_loss_iterationlist.txt', training_loss_iterationlist, delimiter=",", fmt="%.3f")
    np.savetxt(savingfolder+'validation_loss_iterationlist.txt', validation_loss_iterationlist, delimiter=",", fmt="%.3f")


test(model, test_loader, criterion)

# Function to visualize rotated images with their angles
def visualize_rotated_images(dataset, n_images=5):
   fig, axes = plt.subplots(1, n_images, figsize=(15, 15))
   for i in range(n_images):
       image, angle = dataset[i]
       axes[i].imshow(image.squeeze(), cmap='gray')
       axes[i].set_title(f'Angle: {angle}°')
       axes[i].axis('off')
   plt.show()


# Visualize some rotated images with their angles
visualize_rotated_images(train_rotated_dataset)






































































#Regression
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm
import os
savingfolder = '/home/ahmadh/MIR_savedmodel/MNIST_Regression/'
os.mkdir(savingfolder)


# Define a function to rotate images and create labels for the rotation angles
class RotatedMNIST(Dataset):
   def __init__(self, dataset, angles):
       self.dataset = dataset
       self.angles = angles
   def __len__(self):
       return len(self.dataset)
   def __getitem__(self, idx):
       img, digit = self.dataset[idx]
       angle = np.random.choice(self.angles).astype(int)
       #angle = torch.tensor(angle, dtype=torch.float32)
       rotated_img = transforms.functional.rotate(img, int(angle))
       return rotated_img, angle/360, digit

# Define a Convolutional Neural Network for regression
class AnglePredictorCNN(nn.Module):
   def __init__(self):
       super(AnglePredictorCNN, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       self.fc1 = nn.Linear(64 * 7 * 7, 128)
       self.fc2 = nn.Linear(128, 1)  # Predicting a single angle value
   def forward(self, x):
       x = self.pool(torch.relu(self.conv1(x)))
       x = self.pool(torch.relu(self.conv2(x)))
       x = x.view(-1, 64 * 7 * 7)
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x.squeeze()  # Remove unnecessary dimensions

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
   model.train()
   for epoch in range(num_epochs):
       running_loss = 0.0
       for images, angles in train_loader:
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, angles)
           loss.backward()
           optimizer.step()
           running_loss += loss.item()
       print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Testing loop
def test(model, test_loader, criterion):
   model.eval()
   total_loss = 0.0
   with torch.no_grad():
       for images, angles in test_loader:
           outputs = model(images)
           loss = criterion(outputs, angles)
           total_loss += loss.item()
   print(f'Average test loss: {total_loss/len(test_loader):.4f}')


# Hyperparameters
batch_size = 64
learning_rate = 0.001
# Rotation angles for training
rotation_step =15
rotation_angles = list(range(0, 360, rotation_step))  # Angles from 0 to 345 degrees in 15-degree increments
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_rotated_dataset = RotatedMNIST(train_dataset, rotation_angles)
train_loader = DataLoader(train_rotated_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
test_rotated_dataset = RotatedMNIST(test_dataset, rotation_angles)
test_loader = DataLoader(test_rotated_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, define the loss function and the optimizer
model = AnglePredictorCNN()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print_every = 100

num_epochs = 15
model.train()
for epoch in range(num_epochs):
   running_loss = 0.0
   j = 0
   for images, angles, digit in tqdm.tqdm(train_loader):
       j+=1
       optimizer.zero_grad()
       outputs = model(images)
       loss = criterion(outputs, angles.to(torch.float32))
       loss.backward()
       optimizer.step()
       running_loss += loss.item()
       if j%print_every == 0:
         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/j:.4f}')
   print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


model.eval()
j = 0
N_digits = 10
N_classes = int(360/rotation_step)

conf_mtrx = np.zeros([N_digits,N_classes,N_classes])
for images, angles, digit in tqdm.tqdm(test_loader):
   j+=1
   outputs = model(images)
   pred_class = ((360*outputs)//rotation_step)%N_classes#(N_classes*outputs).to(int)
   GT_class = ((360*angles)//rotation_step)%N_classes#.to(int)
   for item in range(outputs.shape[0]):
    conf_mtrx[digit,int(pred_class[item]),int(GT_class[item])] += 1

print(conf_mtrx[0,10:20,10:20])
for digit in range(10):
    conf_mtrx[digit].tofile( savingfolder+f'cm{digit}.csv', sep=',')

cm = np.sum(conf_mtrx, axis=0)
cm.tofile( savingfolder+f'cm_total.csv', sep=',')


test(model, test_loader, criterion)


# Function to visualize rotated images with their angles
def visualize_rotated_images(dataset, n_images=5):
   fig, axes = plt.subplots(1, n_images, figsize=(15, 15))
   for i in range(n_images):
       image, angle = dataset[i]
       axes[i].imshow(image.squeeze(), cmap='gray')
       axes[i].set_title(f'Angle: {angle}°')
       axes[i].axis('off')
   plt.show()

# Visualize some rotated images with their angles
visualize_rotated_images(train_rotated_dataset)
