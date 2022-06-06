#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 특정 gpu만 사용하도록 설정
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[ ]:


# input file format : nii.gz file
# Dataset : BraTS 2018 Dataset
# Framework : Pytorch
# Network : UNet
# Goal : Tumor segmentation from brain mri img

### 1. Build U-Net model & Loss function(dice coefficient)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import softmax

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv(in_dim, out_dim, kernel_size):
            stride = 1
            padding = 1

            model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(num_features = out_dim),
                nn.ReLU()
            )
            return model

        def deconv(in_dim, out_dim, kernel_size):
            stride = 2
            padding = 1

            model = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
            return model
        
        # Contracting path (Encoder)
        num_filter = 8
        
        self.conv1_1 = conv(in_dim=4, out_dim=num_filter, kernel_size=3)
        self.conv1_2 = conv(num_filter, num_filter, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = conv(num_filter, num_filter*2, 3)
        self.conv2_2 = conv(num_filter*2, num_filter*2, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = conv(num_filter*2, num_filter*4, 3)
        self.conv3_2 = conv(num_filter*4, num_filter*4, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = conv(num_filter*4, num_filter*8, 3)
        self.conv4_2 = conv(num_filter*8, num_filter*8, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = conv(num_filter*8, num_filter*16, 3)
        self.conv5_2 = conv(num_filter*16, num_filter*16, 3)
        
        # Expansive path (Decoder)
        self.deconv6 = deconv(in_dim = num_filter*16, out_dim = num_filter*8, kernel_size = 4)
        self.conv6_1 = conv(num_filter*16, num_filter*8, 3)
        self.conv6_2 = conv(num_filter*8, num_filter*8, 3)
        
        self.deconv7 = deconv(num_filter*8, num_filter*4, 4)
        self.conv7_1 = conv(num_filter*8, num_filter*4, 3)
        self.conv7_2 = conv(num_filter*4, num_filter*4, 3)
        
        self.deconv8 = deconv(num_filter*4, num_filter*2, 4)
        self.conv8_1 = conv(num_filter*4, num_filter*2, 3)
        self.conv8_2 = conv(num_filter*2, num_filter*2, 3)
        
        self.deconv9 = deconv(num_filter*2, num_filter, 4)
        self.conv9_1 = conv(num_filter*2, num_filter, 3)
        self.conv9_2 = conv(num_filter, num_filter, 3)
        
        self.out = nn.Conv2d(in_channels=num_filter, out_channels=2, kernel_size=1, stride=1, padding=0)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
#                 nn.init.normal_(m.weight.data, mean=0, std=0.01)
                nn.init.xavier_normal_(m.weight.data)
#                 nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, input, prt=False):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        deconv6 = self.deconv6(conv5_2)
        concat6 = torch.cat((deconv6, conv4_2), dim=1)
        conv6_1 = self.conv6_1(concat6)
        conv6_2 = self.conv6_2(conv6_1)
        
        deconv7 = self.deconv7(conv6_2)
        concat7 = torch.cat((deconv7, conv3_2), dim=1)
        conv7_1 = self.conv7_1(concat7)
        conv7_2 = self.conv7_2(conv7_1)
        
        deconv8 = self.deconv8(conv7_2)
        concat8 = torch.cat((deconv8, conv2_2), dim=1)
        conv8_1 = self.conv8_1(concat8)
        conv8_2 = self.conv8_2(conv8_1)
        
        deconv9 = self.deconv9(conv8_2)
        concat9 = torch.cat((deconv9, conv1_2), dim=1)
        conv9_1 = self.conv9_1(concat9)
        conv9_2 = self.conv9_2(conv9_1)
        
        output = self.out(conv9_2)
        output = softmax(output, dim=1)
        
        if(prt == True):
            print("input shape :", input.shape)
            print("")
            
            print("conv1_1 shape :", conv1_1.shape)
            print("conv1_2 shape :", conv1_2.shape)
            print("pool1 shape :", pool1.shape)
            print("")
            
            print("conv2_1 shape :", conv2_1.shape)
            print("conv2_2 shape :", conv2_2.shape)
            print("pool2 shape :", pool2.shape)
            print("")
            
            print("conv3_1 shape :", conv3_1.shape)
            print("conv3_2 shape :", conv3_2.shape)
            print("pool3 shape :", pool3.shape)
            print("")

            print("conv4_1 shape :", conv4_1.shape)
            print("conv4_2 shape :", conv4_2.shape)
            print("pool4 shape :", pool4.shape)
            print("")
            
            print("conv5_1 shape :", conv5_1.shape)
            print("conv5_2 shape :", conv5_2.shape)
            print("")
            
            print("deconv6_shape :", deconv6.shape)
            print("concat6_shape :", concat6.shape)
            print("conv6_1 shape :", conv6_1.shape)
            print("conv6_2 shape :", conv6_2.shape)
            print("")
            
            print("deconv7_shape :", deconv7.shape)
            print("concat7_shape :", concat7.shape)
            print("conv7_1 shape :", conv7_1.shape)
            print("conv7_2 shape :", conv7_2.shape)
            print("")
            
            print("deconv8_shape :", deconv8.shape)
            print("concat8_shape :", concat8.shape)
            print("conv8_1 shape :", conv8_1.shape)
            print("conv8_2 shape :", conv8_2.shape)
            print("")
            
            print("deconv9_shape :", deconv9.shape)
            print("concat9_shape :", concat9.shape)
            print("conv9_1 shape :", conv9_1.shape)
            print("conv9_2 shape :", conv9_2.shape)
            print("")
            
            print("\noutput shape :", output.shape)
        
        return output

# loss function : dice coefficient
def dice_coef(predict, target):
    smooth = 1

    intersection = (predict * target).sum()
    loss = 1 - ((2.0 * intersection + smooth) / (predict.sum() + target.sum() + smooth))
    
#     print('intersection :', intersection)
#     print('boonja :', (2.0 * intersection + smooth))
#     print('boonmo :', (predict.sum() + target.sum() + smooth))
#     print('loss :', loss)   
    
    return loss


# In[ ]:


### 2. Load the dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import MNI152_FILE_PATH
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
torch.set_printoptions(threshold=99999, linewidth=99999)

def convNii2np(paths, flag):
    crop_from = 24
    crop_to = 216
    depth_from = 14
    depth_to = 142
    result_list = []
    
    for path in paths:
        np_img_list = []
        
        for img in path:
            img_data = nib.load(img)
            np_img_data = np.array(img_data.get_fdata())
            np_img_data = np_img_data[crop_from:crop_to, crop_from:crop_to, depth_from:depth_to]
            np_img_data = np_img_data.transpose(2,0,1).reshape(128, 192, 192)
            
            # convert input img to gray scale.
            if(flag == 'input'):
                np_img_data = (np_img_data / np.max(np_img_data)) * 255
                np_img_data = np_img_data.astype(np.uint8)
                
                # standardization
                np_img_data = (np_img_data - np.mean(np_img_data)) / np.std(np_img_data)
                
                # normalization
#                 np_img_data = (np_img_data - np.min(np_img_data[np_img_data>0])) / (np.max(np_img_data) - np.min(np_img_data[np_img_data>0]))
                
            # convert label img to binary scale.
            elif(flag == 'label'):
                np_img_list.append(np_img_data)
                np_img_data[np_img_data > 1] = 1
                np_img_data = 1 - np_img_data
            
            np_img_list.append(np_img_data)
            
        np_input_data = np.concatenate(np_img_list, axis=0)
        result_list.append(np_input_data)
            
    return result_list

# Set up the directories.
base_dir = './data/'
train_dir = base_dir + 'train/'
valid_dir = base_dir + 'valid/'

all_dir = train_dir + 'ALL/'
hgg_dir = train_dir + 'HGG/'
lgg_dir = train_dir + 'LGG/'
train_sub_dir = []
valid_sub_dir = []

# Get train & valid's sub directories
train_dirnames = os.listdir(all_dir)
valid_dirnames = os.listdir(valid_dir)

for dirname in train_dirnames:
    train_sub_dir.append(dirname)

for dirname in valid_dirnames:
    valid_sub_dir.append(dirname)

# Get dataset paths
channels = ['flair', 't1', 't1ce', 't2']

train_paths = []
t_label_paths = []
np_train_list = []
np_t_label_list = []

valid_paths = []
v_label_paths = []
np_valid_list = []
np_v_label_list = []

for sub_dir_name in train_sub_dir:
    train_paths.append([os.path.join(all_dir + sub_dir_name + '/' + sub_dir_name + '_' + channel + '.nii.gz') for channel in channels])
    t_label_paths.append([os.path.join(all_dir + sub_dir_name + '/' + sub_dir_name + '_seg.nii.gz')])
    
for sub_dir_name in valid_sub_dir:
    valid_paths.append([os.path.join(valid_dir + sub_dir_name + '/' + sub_dir_name + '_' + channel + '.nii.gz') for channel in channels])
    v_label_paths.append([os.path.join(valid_dir + sub_dir_name + '/' + sub_dir_name + '_seg.nii.gz')])    
        
# Convert nii to tensor with reshaping (4, 128, 192, 192)
print('------ Making dataset tensor ------')
np_train_list = convNii2np(train_paths, 'input')
np_valid_list = convNii2np(valid_paths, 'input')
np_t_label_list = convNii2np(t_label_paths, 'label')   
np_v_label_list = convNii2np(v_label_paths, 'label')

train_tensor_list = []
valid_tensor_list = []
t_label_tensor_list = []
v_label_tensor_list = []

for input_, label_ in zip(np_train_list, np_t_label_list):
    temp_tensor = torch.from_numpy(input_)
    temp_tensor = temp_tensor.reshape(4, 128, 192, 192)
    temp_tensor = temp_tensor.transpose(0, 1)
    train_tensor_list.append(temp_tensor)
    
    temp_tensor = torch.from_numpy(label_)
    temp_tensor = temp_tensor.reshape(2, 128, 192, 192)
    temp_tensor = temp_tensor.transpose(0, 1)
    t_label_tensor_list.append(temp_tensor)

for input_, label_ in zip(np_valid_list, np_v_label_list):
    temp_tensor = torch.from_numpy(input_)
    temp_tensor = temp_tensor.reshape(4, 128, 192, 192)
    temp_tensor = temp_tensor.transpose(0, 1)
    valid_tensor_list.append(temp_tensor)
    
    temp_tensor = torch.from_numpy(label_)
    temp_tensor = temp_tensor.reshape(2, 128, 192, 192)
    temp_tensor = temp_tensor.transpose(0, 1)
    v_label_tensor_list.append(temp_tensor)

print("Num of train data and label data : %d, %d" %(len(train_tensor_list), len(t_label_tensor_list)))
print("Num of valid data and label data : %d, %d" %(len(valid_tensor_list), len(v_label_tensor_list)))
print("shape of the train & label : ", train_tensor_list[0].shape, t_label_tensor_list[0].shape)
print("shape of the valid & label : ", valid_tensor_list[0].shape, v_label_tensor_list[0].shape)
print('------ Done ------')

# Save the memory
del train_sub_dir
del valid_sub_dir

del train_paths
del valid_paths
del t_label_paths
del v_label_paths

del np_train_list
del np_t_label_list
del np_valid_list
del np_v_label_list

train_tensors = torch.Tensor(len(train_tensor_list), 128, 4, 192, 192)
torch.cat(train_tensor_list, dim=0, out=train_tensors)
del train_tensor_list

valid_tensors = torch.Tensor(len(valid_tensor_list), 128, 4, 192, 192)
torch.cat(valid_tensor_list, dim=0, out=valid_tensors)
del valid_tensor_list

t_label_tensors = torch.Tensor(len(t_label_tensor_list), 128, 2, 192, 192)
torch.cat(t_label_tensor_list, dim=0, out=t_label_tensors)
del t_label_tensor_list

v_label_tensors = torch.Tensor(len(v_label_tensor_list), 128, 2, 192, 192)
torch.cat(v_label_tensor_list, dim=0, out=v_label_tensors)
del v_label_tensor_list

# Set the Dataset
train_dataset = TensorDataset(train_tensors, t_label_tensors)
valid_dataset = TensorDataset(valid_tensors, v_label_tensors)


# In[ ]:


### 3. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\ndevice :', device)

# hyper parameters
lr = 1e-4
num_epoch = 100
batch_size = 32

# Load the data using DataLoader
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers=8)

net = UNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
t_loss_arr = []
v_loss_arr = []

print('\n------ Hyper Params ------')
print("learning rate: %.4e" % lr)
print("number of epoch: %d" % num_epoch)
print("batch size: %d" % batch_size)

print('\n\n------ Start Training ------')
for epoch in range(1, num_epoch+1):
    
    # train the model
    net.train()
    epoch_loss = 0
    
    ####################################################################
#     fig = plt.figure()
#     rows = 1
#     cols = 3
#     custom_depth = 0
    ####################################################################    
    
    for t_input, t_label in train_loader:

        ####################################################################
#         np_t_input_img = t_input.detach().numpy()
#         np_t_input_img = np.array(np_t_input_img)
#         print(np.max(np_t_input_img))
#         t_input_img = np_t_input_img[custom_depth, 0, :, :]
#         ax = fig.add_subplot(rows, cols, 1)
#         ax.imshow(t_input_img, vmin=0, vmax=255, cmap='gray')
#         ax.set_title('input')
        ####################################################################

        # Forward propagation
        t_input = t_input.cuda()
        t_label = t_label.cuda()
        
        t_output = net(t_input.float())
        
        ####################################################################
#         t_output_img = t_output.cpu()
#         np_t_output_img = t_output_img.detach().numpy()
#         np_t_output_img = np.array(np_t_output_img)
#         np_t_output_img = np_t_output_img.argmax(1)
        
#         t_result = np_t_output_img[0, :, :]
#         ax = fig.add_subplot(rows, cols, 2)
#         ax.imshow(t_result)#, cmap='gray')
#         ax.set_title('predict')

#         t_label_img = t_label.cpu()
#         t_label_img = t_label_img[custom_depth, 0, :, :]
#         ax = fig.add_subplot(rows, cols, 3)
#         ax.imshow(t_label_img)#, cmap='gray')
#         ax.set_title('Label')
        
#         plt.savefig('forward_result.png')
        ####################################################################
        
        # Backward propagation
        optimizer.zero_grad()    # 역전파 실행 전 grad 값 0으로 만듦
        loss = 0        

        for i in range(0, 2):
            loss_ = dice_coef(t_output[:, i, :, :], t_label[:, 1-i, :, :])
#             print(loss_)
            loss += loss_
        loss = (loss / 2)
        loss.backward()          # 매개변수에 대한 loss의 grad 값 계산
        optimizer.step()         # 매개변수 갱신
        epoch_loss += loss.item()
        
        
        
#         ''''''
#         t_output = t_output.argmax(1)
        
#         loss_ = dice_coef(t_output[:, :, :], t_label[:, 0, :, :])
#         loss += loss_
#         loss_.requires_grad = True
#         loss_.backward()
#         optimizer.step()
#         epoch_loss += loss_
#         ''''''
        
        
    print("Train: EPOCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, epoch_loss/len(train_dataset)))
    t_loss_arr.append(epoch_loss/len(train_dataset))
    
    # validate the model
    net.eval()
    epoch_loss = 0
    
    for v_input, v_label in valid_loader:
        v_input = v_input.cuda()
        v_label = v_label.cuda()
        
        v_output = net(v_input.float())
        
        loss = 0
        for i in range(0, 2):
            loss_ = dice_coef(v_output[:, i, :, :], v_label[:, 1-i, :, :])
            loss += loss_
        loss = (loss / 2)
        epoch_loss += loss.item()
    print("Valid: EPOCH %04d / %04d | LOSS %.4f\n" %(epoch, num_epoch, epoch_loss/len(valid_dataset)))
    v_loss_arr.append(epoch_loss/len(valid_dataset))
    
print("------ Finished Training ------")
epoch_list = range(1, num_epoch+1)
plt.plot(epoch_list, np.array(t_loss_arr), 'y')
plt.plot(epoch_list, np.array(v_loss_arr), 'c')
plt.title('Loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend((['train loss', 'validation loss']))
plt.savefig('loss graph2D.png')


# In[ ]:


### 4. Test the model
np.set_printoptions(threshold=np.inf, linewidth=np.inf)    # numpy 생략(...) 없애기

test_dir = base_dir + 'test/'

test_sub_dir = []
test_img_list = []
label_img_list = []
np_test_data_list = []

testdirnames = os.listdir(test_dir)
for dirname in testdirnames:
    test_sub_dir.append(dirname)
    
for test_sub_path in test_sub_dir:
    test_img_list.append([os.path.join(test_dir + test_sub_path + '/' + test_sub_path + '_' + channel + '.nii.gz') for channel in channels])
    label_img_list.append([os.path.join(test_dir + test_sub_path + '/' + test_sub_path + '_seg.nii.gz')])

# predict the output image
np_test_list = convNii2np(test_img_list, 'input')
np_test_data = np.concatenate(np_test_list, axis = 0)
test_tensor = torch.from_numpy(np_test_data)
test_tensor = test_tensor.reshape(4, 128, 192, 192)
test_tensor = test_tensor.transpose(0, 1)
test_tensor = test_tensor.cuda()

predict_img = net(test_tensor.float())

fig = plt.figure()
rows = 1
cols = 2
custom_depth = 50

## predict image
predict_img = predict_img.cpu()
np_predict_img = predict_img.detach().numpy()
np_predict_img = np.array(np_predict_img)
np_predict_img = np_predict_img.argmax(1)

test_result = np_predict_img[custom_depth, :, :]
ax = fig.add_subplot(rows, cols, 1)
ax.imshow(test_result)
ax.set_title('Predict')

# load the label image
for label_img in label_img_list[0]:
    label_nii = nib.load(label_img)
    label_img = np.array(label_nii.get_fdata()[24:216, 24:216, 14:142])
    label_img[label_img > 1] = 1
    label_img = label_img.transpose(2,0,1).reshape(128, 192, 192)
    label_img = label_img[custom_depth, :, :]
    ax = fig.add_subplot(rows, cols, 2)
    ax.imshow(label_img)
    ax.set_title('Label')

plt.savefig('result2D.png')


# In[ ]:


# Save the model
PATH = './weights/'
torch.save(net, PATH+'unet2D.pt')
torch.save(net.state_dict(), PATH+'unet_state_dict2D.pt')

