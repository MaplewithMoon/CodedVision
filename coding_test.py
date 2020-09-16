#!/usr/bin/env python2

# Writen by Haojie Liu
# Updated 2018.03.23 by Tong Chen
# CPU version of pytorch Deepcoder

import os
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as f
import torch.utils.data as Data
import h5py
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import gc
import bz2

import numpy as np
import multi_scale as AEModel
import torch_msssim
#load pretrained encoder and decoder
#AE_net = torch.load('./12000_0.94740096_0.00034681_1.54372260.pkl')
AE_net = torch.load('./28.20_4.6M_python3.pkl')#需改成相应的模型名称


print(AE_net)
AE_net = AE_net.cuda()
# test 
AE_net.eval()


ours_bpp = []
bpg_bpp = []

for filename in os.listdir('/home/maple/cifar_test/data/professional_test/test'):#读取测试图片
    print (filename)   
    src = '/home/maple/cifar_test/data/professional_test/test/'+filename

	
    #fsize1 = os.path.getsize('/home/lhj/Desktop/torch/ImageCompression/result_clic/BPG/%d.bpg'%i)
    #bpg_bpp.append(fsize1 * 8 / 768.0 / 512	
    img = cv2.imread(src)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#cv2读取图片默认是BGR格式，将其转为RGB

    print(src)
    img = img/255.0

    H_ORG, W_ORG, C = img.shape

    # divided by 8
    H_PAD = int(8.0 * np.ceil(H_ORG / 8.0))#将图像的长宽剪裁为可被8整除的最大尺寸
    W_PAD = int(8.0 * np.ceil(W_ORG / 8.0))

    print(H_PAD,W_PAD)

    im = np.zeros([H_PAD,W_PAD,C],dtype = 'float32')

    im[:H_ORG,:W_ORG,:] = img

    H, W, C = im.shape
    print(im.shape)




    im = torch.FloatTensor(im)

    im = im.permute(2,0,1).contiguous()

    im = im.view(1,C,H,W)

    im = Variable(im,volatile=True).cuda()


    #add a h5 read/write
    #output1,encoder1 = scale1_AE(im)
    #seperate
    encoder1,_ = AE_net.encoder(im)
    output1 = AE_net.decoder(encoder1)
	
	#采用算术编码方法paq对encoder的feature map进行熵编码
    h5f = h5py.File('test.h5','w')
    h5f.create_dataset('data',data=encoder1.data.cpu().numpy())
    h5f.close()

    os.system("zpaq/zpaq a test test.h5 -method 6")#这里用了linux的指令来运行zpaq的包
    fsize = os.path.getsize('/home/maple/cifar_test/test.zpaq')
    ours_bpp.append(fsize*8.0/H/W)
    os.system('rm test.zpaq')
    print ours_bpp
	
	#重建并保存图像
    output1 = output1.data[0].cpu().numpy()
    output1 = output1.transpose(1,2,0)


    output1[output1>=1] = 1
    output1[output1<=0] = 0

    recons = output1[:H_ORG,:W_ORG,:]*255.0

    rec = Image.fromarray(np.uint8(recons))
    src1 = os.path.join('/home/maple/cifar_test/test_result'+filename+'.png')
    # src1 = '"' + src1 + '"'
    rec.save(src1)
    break#可去掉

	
#计算码率（bit per pixel）
final_bpp1 = sum(ours_bpp)/24.0
print 'ours:',final_bpp1
#评价重建质量（MS-SIM）
img1 = cv2.imread(src)
H, W, C = img1.shape
img1 = torch.FloatTensor(img1)
img1 = img1.permute(2,0,1).contiguous()
img1 = img1.view(1,C,H,W)
img1 = Variable(img1).cuda()

img2 = cv2.imread(src1)
H, W, C = img2.shape
img2 = torch.FloatTensor(img2)
img2 = img2.permute(2,0,1).contiguous()
img2 = img2.view(1,C,H,W)
img2 = Variable(img2).cuda()

MS_sim =torch_msssim.MS_SSIM()
result = MS_sim.ms_ssim(img1,img2)
print(result)

