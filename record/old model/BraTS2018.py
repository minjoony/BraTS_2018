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

### 1. Load the dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import MNI152_FILE_PATH

base_dir = './data/'
train_dir = base_dir + 'train/'
hgg_dir = train_dir + 'HGG/'
lgg_dir = train_dir + 'LGG/'
sub_dir = []

# List up the train's sub directories
dirnames = os.listdir(hgg_dir)
for dirname in dirnames:
    sub_dir.append(dirname)

# sub_dir.append('Brats18_2013_0_1')    # Just for testing...

# List up dataset
channels = ['flair', 't1', 't1ce', 't2']

train_paths = []
label_paths = []
np_train_list = []
np_label_list = []

for sub_dir_name in sub_dir:
    train_paths.append([os.path.join(lgg_dir + sub_dir_name + '/' + sub_dir_name + '_' + channel + '.nii.gz') for channel in channels])
    label_paths.append([os.path.join(lgg_dir + sub_dir_name + '/' + sub_dir_name + '_seg.nii.gz')])
'''
img_paths = [   [1.flair.nii.gz, 1.t1.nii.gz, 1.t1ce.nii.gz, 1.t2.nii.gz]
                [2.flair.nii.gz, 2.t1.nii.gz, 2.t1ce.nii.gz, 2.t2.nii.gz]
                ...
            ]
            
label_paths = [ [1.seg.nii.gz]
                [2.seg.nii.gz]
                ...
              ]            
'''
# Convert nii to numpy and reshape
## Concatenate the train dataset to make tensor (4, 240, 240, 128)
print('making train tensor...')
depth_from = 14
depth_to = 142

for train_path in train_paths:
    np_img_data_list = []
    
    for img in train_path:
        img_data = nib.load(img)
        np_img_data = np.array(img_data.get_fdata())
        np_img_data = np_img_data[:, :, depth_from:depth_to]
        np_img_data = np.reshape(np_img_data, (1, 240, 240, 128))
        np_img_data_list.append(np_img_data)
        
    np_train_data = np.concatenate(np_img_data_list, axis=0)
    np_train_list.append(np_train_data)

## convert label dataset
for label_path in label_paths:
    for img in label_path:
        img_data = nib.load(img)
        np_img_data = np.array(img_data.get_fdata())
        np_img_data = np_img_data[:, :, depth_from:depth_to]
        np_img_data = np.reshape(np_img_data, (1, 240, 240, 128))
        np_label_list.append(np_img_data)
print('...done')


# In[ ]:


### 2. Build U-Net model & Loss function(dice coefficient)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import softmax

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv(in_dim, out_dim, kernel_size, channel_num):
            stride = 1
            padding = 1

            model = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm3d(num_features = out_dim),
                nn.ReLU()
            )
            return model

        def deconv(in_dim, out_dim, kernel_size, channel_num):
            stride = 2
            padding = 0

            model = nn.Sequential(
                nn.ConvTranspose3d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm3d(out_dim),
                nn.ReLU()
            )
            return model
        
        # Contracting path (Encoder)
        self.conv1_1 = conv(in_dim=4, out_dim=8, kernel_size=3, channel_num=4)
        self.conv1_2 = conv(8, 8, 3, 4)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2_1 = conv(8, 16, 3, 4)
        self.conv2_2 = conv(16, 16, 3, 4)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3_1 = conv(16, 32, 3, 4)
        self.conv3_2 = conv(32, 32, 3, 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv4_1 = conv(32, 64, 3, 4)
        self.conv4_2 = conv(64, 64, 3, 4)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv5_1 = conv(64, 128, 3, 4)
        self.conv5_2 = conv(128, 128, 3, 4)
        
        # Expansive path (Decoder)
        self.deconv6 = deconv(in_dim = 128, out_dim = 64, kernel_size = 2, channel_num = 4)
        self.conv6_1 = conv(128, 64, 3, 4)
        self.conv6_2 = conv(64, 64, 3, 4)
        
        self.deconv7 = deconv(64, 32, 2, 4)
        self.conv7_1 = conv(64, 32, 3, 4)
        self.conv7_2 = conv(32, 32, 3, 4)
        
        self.deconv8 = deconv(32, 16, 2, 4)
        self.conv8_1 = conv(32, 16, 3, 4)
        self.conv8_2 = conv(16, 16, 3, 4)
        
        self.deconv9 = deconv(16, 8, 2, 4)
        self.conv9_1 = conv(16, 8, 3, 4)
        self.conv9_2 = conv(8, 8, 3, 4)
        
        self.out = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
    
    def forward(self, input):
#         print("input shape :", input.shape)
        conv1_1 = self.conv1_1(input)
#         print("\nconv1_1 shape :", conv1_1.shape)
        conv1_2 = self.conv1_2(conv1_1)
#         print("conv1_2 shape :", conv1_2.shape)
        pool1 = self.pool1(conv1_2)
#         print("pool1 shape :", pool1.shape)
        
        conv2_1 = self.conv2_1(pool1)
#         print("\nconv2_1 shape :", conv2_1.shape)
        conv2_2 = self.conv2_2(conv2_1)
#         print("conv2_2 shape :", conv2_2.shape)
        pool2 = self.pool2(conv2_2)
#         print("pool2 shape :", pool2.shape)
        
        conv3_1 = self.conv3_1(pool2)
#         print("conv3_1 shape :", conv3_1.shape)
        conv3_2 = self.conv3_2(conv3_1)
#         print("conv3_2 shape :", conv3_2.shape)
        pool3 = self.pool3(conv3_2)
#         print("pool3 shape :", pool3.shape)

        
        conv4_1 = self.conv4_1(pool3)
#         print("\nconv4_1 shape :", conv4_1.shape)
        conv4_2 = self.conv4_2(conv4_1)
#         print("conv4_2 shape :", conv4_2.shape)
        pool4 = self.pool4(conv4_2)
#         print("pool4 shape :", pool4.shape)

        conv5_1 = self.conv5_1(pool4)
#         print("\nconv5_1 shape :", conv5_1.shape)
        conv5_2 = self.conv5_2(conv5_1)
#         print("conv5_2 shape :", conv5_2.shape)

        
        deconv6 = self.deconv6(conv5_2)
#         print("\ndeconv6 shape :", deconv6.shape)
        concat6 = torch.cat((deconv6, conv4_2), dim=1)
#         print("concat6 shape :", concat6.shape)
        conv6_1 = self.conv6_1(concat6)
#         print("conv6_1 shape :", conv6_1.shape)
        conv6_2 = self.conv6_2(conv6_1)
#         print("conv6_2 shape :", conv6_2.shape)
        
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
#         print("\nconv9_2 shape :", conv9_2.shape)

        
        output = self.out(conv9_2)
        output = softmax(output, dim=1)
        
#         if(prt == True):
#             print("input shape :", input.shape)
#             print("\n")
            
#             print("conv1_1 shape :", conv1_1.shape)
#             print("conv1_2 shape :", conv1_2.shape)
#             print("pool1 shape :", pool1.shape)
#             print("\n")
            
#             print("conv2_1 shape :", conv2_1.shape)
#             print("conv2_2 shape :", conv2_2.shape)
#             print("pool2 shape :", pool2.shape)
#             print("\n")
            
#             print("conv3_1 shape :", conv3_1.shape)
#             print("conv3_2 shape :", conv3_2.shape)
#             print("pool3 shape :", pool3.shape)
#             print("\n")

#             print("conv4_1 shape :", conv4_1.shape)
#             print("conv4_2 shape :", conv4_2.shape)
#             print("pool4 shape :", pool4.shape)
#             print("\n")

#         print("\noutput shape :", output.shape)
        return output

# loss function : dice coefficient
def dice_coef(predict, target):
    smooth = 1.0
    class_num = 4
    
    intersection = (predict * target).sum()
    loss = 1 - ((2.0 * intersection + smooth) / (predict.sum() + target.sum() + smooth))
    return loss


# In[ ]:


'''

### 3.1 Forward propagation
train_tensor = torch.from_numpy(np_train_list[0])
print(train_tensor.shape)
train_tensor = train_tensor.reshape(1, 4, 240, 240, 128)
# print(train_tensor.shape)

tempUnet = UNet()
output_tensor = tempUnet.forward(train_tensor.float())

# predict_flair = output_tensor[0, 0, :, :]
# t1_flair = output_tensor[0, 1, :, :]
# t1ce_flair = output_tensor[0, 2, :, :]
# t2_flair = output_tensor[0, 3, :, :]

# forward-propagation 결과 img로 저장.
fig = plt.figure(figsize=(15,10))
rows = 2
cols = 4
custom_depth = 65

for i in range(1,5):
    np_input_tensor = train_tensor.detach().numpy()
    predict_img = np_input_tensor[0][i-1, :, :, custom_depth]
    
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(predict_img)
    ax.set_title('Origin_' + channels[i-1])
    
# for i in range(1,5):
#     np_output_tensor = output_tensor.detach().numpy()
#     forward_img = np_output_tensor[0][i-1, :, :, custom_depth]
    
#     ax = fig.add_subplot(rows, cols, i+4)
#     ax.imshow(forward_img)
#     ax.set_title('Predict_' + channels[i-1])
    
# plt.savefig('forward_img.png')
np.set_printoptions(threshold=np.inf, linewidth=np.inf)    # numpy 생략(...) 없애기
print(output_tensor)

np_output_tensor = output_tensor.detach().numpy()
# print(np_output_tensor)

forward_img = np_output_tensor[0][0, :, :, custom_depth]
ax = fig.add_subplot(rows, cols, 5)
ax.imshow(forward_img)
ax.set_title('Predict')

label_img = np_label_list[0][0, :, :, custom_depth]
ax = fig.add_subplot(rows, cols, 6)
ax.imshow(label_img)
ax.set_title('Label')

plt.savefig('forward_img.png')

'''


# In[2]:


### 3. Train the model
# tempUnet.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\ndevice :', device)

# hyper parameters
lr = 0.01
# batch_size = 
num_epoch = 2

# input, label data setting
train_tensor = torch.from_numpy(np_train_list[0])
train_tensor = train_tensor.reshape(1, 4, 240, 240, 128)

label_tensor = torch.from_numpy(np_label_list[0])
# label_tensor = label_tensor.reshape(1, 1, 240, 240, 128)

t_inputs = [train_tensor]
t_targets = [label_tensor]

# print('\ninput shape :', inputs[0].shape)
# print('target shape :', targets[0].shape)

net = UNet().to(device)
# loss_fn = dice_coef()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
t_loss_arr = []
v_loss_arr = []

print('\n\n------ Hyper Params ------')
print("learning rate: %.4e" % lr)
# print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print('\n\n------ Start Training ------')

for epoch in range(1, num_epoch+1):
    # train the model
    net.train()
    for t_input, t_label in zip(t_inputs, t_targets):
        # Forward propagation
        t_input = t_input.cuda()
        t_label = t_label.cuda()
        
        t_output = net(t_input.float())
        
        # Backward propagation
        optimizer.zero_grad()    # 역전파 실행 전 grad 값 0으로 만듦
        loss = 0
        '''
        max를 1로 나머지 0 으로 만드는 코드 추가...         
        '''
        for i in range(len(channels)):
            loss_ = dice_coef(t_output[0, i, :, :, :], t_label[:, :, :, :])
            loss += loss_
        loss = (loss / len(channels))
        loss.backward()          # 매개변수에 대한 loss의 grad 값 계산
        optimizer.step()         # 매개변수 갱신
    
        t_loss_arr += [loss.item()]
        
        print("Train: EPOCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, np.mean(t_loss_arr)))
    
#     # validate the model
#     net.eval()
#     for v_input, v_label in zip(v_inputs, v_targets):
#         v_input = v_input.cuda()
#         v_label = v_label.cuda()
        
#         v_output = net(v_input.float())
        
#         loss = 0
#         for i in range(len(channels)):
#             loss_ = dice_coef(v_output[0, i, :, :, :], v_label[:, :, :, :])
#             loss += loss_
#         loss = (loss / len(channels))
#         v_loss_arr += [loss.item()]
#         print("Valid: EPOCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, np.mean(v_loss_arr)))

print("------ Finished Training ------")
plt.plot(np.array(t_loss_arr), 'y')
# plt.plot(np.array(v_loss_arr), 'c')
plt.title('Loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend((['train loss', 'validation loss']))
plt.savefig('loss graph.png')


# In[1]:


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
for test_img in test_img_list[0]:
    test_nii = nib.load(test_img)
    np_test_data = np.array(test_nii.get_fdata()[:, :, 14:142])
    np_test_data = np.reshape(np_test_data, (1, 240, 240, 128))
    np_test_data_list.append(np_test_data)
np_test_data = np.concatenate(np_test_data_list, axis = 0)
test_tensor = torch.from_numpy(np_test_data)
test_tensor = test_tensor.reshape(1, 4, 240, 240, 128)
test_tensor = test_tensor.cuda()

predict_img = net(test_tensor.float())
predict_img = predict_img.cpu()

fig = plt.figure()
rows = 1
cols = 2
custom_depth = 65

# load the label image
for label_img in label_img_list[0]:
    print(label_img)
    label_nii = nib.load(label_img)
    label_img = np.array(label_nii.get_fdata()[:, :, 14:142])
    label_img = label_img[:, :, custom_depth]
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(label_img)
    ax.set_title('Label')

## predict image
np_predict_img = predict_img.detach().numpy()
np_predict_img = np.array(np_predict_img)
np_predict_img = np_predict_img.argmax(1)
test_result = np_predict_img[0, :, :, custom_depth]
ax = fig.add_subplot(rows, cols, 2)
ax.imshow(test_result)
ax.set_title('Predict')

plt.savefig('result.png')


# In[ ]:




