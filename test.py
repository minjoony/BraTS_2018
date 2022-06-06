#!/usr/bin/env python
# coding: utf-8

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
        
        def conv(in_dim, out_dim, kernel_size, channel_num):
            stride = 1
            padding = 1

            model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(num_features = out_dim),
                nn.ReLU()
            )
            return model

        def deconv(in_dim, out_dim, kernel_size, channel_num):
            stride = 2
            padding = 1

            model = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
            return model
        
        # Contracting path (Encoder)
        self.conv1_1 = conv(in_dim=4, out_dim=8, kernel_size=3, channel_num=4)
        self.conv1_2 = conv(8, 8, 3, 4)
        self.pool1 = nn.MaxPool2d(kernel_size=2)#, stride=2)
        
        self.conv2_1 = conv(8, 16, 3, 4)
        self.conv2_2 = conv(16, 16, 3, 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)#, stride=2)
        
        self.conv3_1 = conv(16, 32, 3, 4)
        self.conv3_2 = conv(32, 32, 3, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)#, stride=2)
        
        self.conv4_1 = conv(32, 64, 3, 4)
        self.conv4_2 = conv(64, 64, 3, 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2)#, stride=2)
        
        self.conv5_1 = conv(64, 128, 3, 4)
        self.conv5_2 = conv(128, 128, 3, 4)
        
        # Expansive path (Decoder)
        self.deconv6 = deconv(in_dim = 128, out_dim = 64, kernel_size = 4, channel_num = 4)
        self.conv6_1 = conv(128, 64, 3, 4)
        self.conv6_2 = conv(64, 64, 3, 4)
        
        self.deconv7 = deconv(64, 32, 4, 4)
        self.conv7_1 = conv(64, 32, 3, 4)
        self.conv7_2 = conv(32, 32, 3, 4)
        
        self.deconv8 = deconv(32, 16, 4, 4)
        self.conv8_1 = conv(32, 16, 3, 4)
        self.conv8_2 = conv(16, 16, 3, 4)
        
        self.deconv9 = deconv(16, 8, 4, 4)
        self.conv9_1 = conv(16, 8, 3, 4)
        self.conv9_2 = conv(8, 8, 3, 4)
        
        self.out = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1, stride=1, padding=0)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.normal_(m.weight.data, mean=0, std=0.01)
    
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
    smooth = 1.0

    intersection = (predict * target).sum()
    loss = 1 - ((2.0 * intersection + smooth) / (predict.sum() + target.sum() + smooth))
    return loss


# In[ ]:


import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import MNI152_FILE_PATH
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

PATH = './weights/200922_lgg_NormGrayScaled/'

net = torch.load(PATH+'unet2D.pt')
net.load_state_dict(torch.load(PATH + 'unet_state_dict2D.pt'))

net.eval()

base_dir = './data/valid/'
test_dir = base_dir

channels = ['flair', 't1', 't1ce', 't2']
test_sub_dir = []
test_img_list = []
label_img_list = []
np_test_data_list = []

# testdirnames = os.listdir(test_dir)
# for dirname in testdirnames:
#     test_sub_dir.append(dirname)
    
# for test_sub_path in test_sub_dir:
#     test_img_list.append([os.path.join(test_dir + test_sub_path + '/' + test_sub_path + '_' + channel + '.nii.gz') for channel in channels])
#     label_img_list.append([os.path.join(test_dir + test_sub_path + '/' + test_sub_path + '_seg.nii.gz')])

dir_name = 'Brats18_2013_23_1'
test_img_list.append([os.path.join(base_dir + dir_name + '/' + dir_name + '_' + channel + '.nii.gz') for channel in channels])
label_img_list.append([os.path.join(base_dir + dir_name + '/' + dir_name + '_seg.nii.gz')])

print('\n\n------ Start Testing ------')

# predict the output image
for test_img in test_img_list[0]:
    test_nii = nib.load(test_img)
    np_test_data = np.array(test_nii.get_fdata()[24:216, 24:216, 14:142])
    np_test_data = np_test_data.transpose(2,0,1).reshape(128, 192, 192)
    
    # gray scale
    np_test_data = (np_test_data / np.max(np_test_data)) * 255
    np_test_data = np_test_data.astype(np.uint8)
    
    # standardization
#     np_test_data = (np_test_data - np.mean(np_test_data)) / np.std(np_test_data)
    
    # normalization
    np_test_data = (np_test_data - np.min(np_test_data[np_test_data>0])) / (np.max(np_test_data) - np.min(np_test_data[np_test_data>0]))
    
    np_test_data_list.append(np_test_data)
np_test_data = np.concatenate(np_test_data_list, axis = 0)
test_tensor = torch.from_numpy(np_test_data)
test_tensor = test_tensor.reshape(4, 128, 192, 192)
test_tensor = test_tensor.transpose(0, 1)
test_tensor = test_tensor.cuda()

predict_img = net(test_tensor.float())
predict_img = predict_img.cpu()
print("------ Finished Testing ------")

fig = plt.figure(figsize=(10, 10))
rows = 5
cols = 3

custom_depth = [40, 50, 60, 70, 80]
for idx, depth in enumerate(custom_depth):
    #input image
    input_img = test_tensor.cpu()
    input_img = input_img.detach().numpy()
    input_img = np.array(input_img)
    input_img = input_img[depth, 2, :, :]
    ax = fig.add_subplot(rows, cols, cols*idx+1)
    ax.imshow(input_img, vmin=0, vmax=1, cmap='gray')
    ax.set_title('Input')
    
    # predict image
    np_predict_img = predict_img.detach().numpy()
    np_predict_img = np.array(np_predict_img)
    np_predict_img = np_predict_img.argmax(1)
    
    test_result = np_predict_img[depth, :, :]
    ax = fig.add_subplot(rows, cols, cols*idx+2)
    ax.imshow(test_result)
    ax.set_title('Predict')
    
    # load the label image
    for label_img in label_img_list[0]:
        label_nii = nib.load(label_img)
        label_img = np.array(label_nii.get_fdata()[24:216, 24:216, 14:142])
        label_img = label_img.transpose(2,0,1).reshape(128, 192, 192)
        label_img[label_img > 1] = 1
        label_img = label_img[depth, :, :]
        ax = fig.add_subplot(rows, cols, cols*idx+3)
        ax.imshow(label_img)
        ax.set_title('Label')

plt.tight_layout()
plt.savefig('test_result.png')


# In[ ]:




