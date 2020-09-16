# -*- coding: utf-8 -*-

import os, sys, numpy as np
import argparse

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PascalLoader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
sys.path.append('/home/maple/pascal/model/')
import joint_model as JM

parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2007')
parser.add_argument('-pascal_train_path', default='/home/maple/pascal/data/VOC2012', help='Path to Pascal VOC 2007 folder')
parser.add_argument('--model', default=None, type=str, help='Pretrained model')
parser.add_argument('--epochs', default=100, type=int, help='gpu id')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--crops', default=10, type=int, help='number of random crops during testing')
args = parser.parse_args()

'''
def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)
'''

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
		transforms.RandomResizedCrop(256),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
	])

    val_transform = transforms.Compose([
		transforms.RandomResizedCrop(256),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		#normalize,
	])
    # DataLoader initialize
    train_data = DataLoader(args.pascal_train_path, 'train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch,
                                               shuffle=True)

    val_data = DataLoader(args.pascal_train_path, 'val', transform=val_transform, random_crops=args.crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
											batch_size=args.batch,
											shuffle=False)

    net = JM.CodedVision_net(64).cuda()
    pretrained_AEnet = torch.load('./model/model24 0.00065591.pkl')
    
    pretrained_AEnet_dict = pretrained_AEnet.state_dict()
    model_dict = net.state_dict()

    pretrained_AEnet_dict = {k: v for k, v in pretrained_AEnet_dict.items() if k in model_dict}
    model_dict.update(pretrained_AEnet_dict)

    net.load_state_dict(model_dict)


    net.encoder.eval()
    net.classifier.train()
    net = net.cuda()

    for param in net.encoder.parameters():
        param.requires_grad = False
    for param in net.decoder.parameters():
        param.requires_grad = False
    print(classifier_net1)

    optimizer = torch.optim.SGD(net.classifier.parameters(), lr=args.lr,momentum=0.9)
    criterion = criterion = nn.MultiLabelSoftMarginLoss()

    ############## TRAINING ###############
    print('Start training: lr %f, batch size %d' % (args.lr, args.batch))
    


    # Train the Model

    for epoch in range(args.epochs):
        mAP = []
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            
            #print(images.shape)
            #print(images)
            images = Variable(images).cuda()
            '''
            image = images.data.cpu().numpy()
            
            for k in range (10):
                inputs = image[k,:,:,:]
                print(inputs.shape)
                inputs = inputs.transpose(1,2,0)
                inputs = inputs * 255
                show_input = Image.fromarray(np.uint8(inputs))
                src2 = os.path.join('/home/maple/pascal/data/VOC2012/'+'input'+ str(k)+'.png')
                show_input.save(src2)
            '''
            labels = Variable(labels).cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            encode_features = net.encoder(images)
            #rec_images = net.decoder(encode_features)
            outputs = net.classifier(encode_features)
            '''
            #show reconstruction images
            #rec_images[rec_images>1]=1
            #rec_images[rec_images<0]=0
            
            recons = rec_images.data.cpu().numpy()
            # for all images in the batch
            for j in range (10):
                recon = recons[j,:,:,:]
                recon[recon>=1] = 1
                recon[recon<=0] = 0
                print(recon.shape)
                recon = recon.transpose(1,2,0)
                recon = recon*255
                rec = Image.fromarray(np.uint8(recon))
                src1 = os.path.join('/home/maple/pascal/data/VOC2012/'+ str(j)+'.png')
                rec.save(src1)

            #outputs = net.classifier(rec_images)
            '''
            ##mAP.append(compute_mAP(labels.data, outputs.data))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().data.numpy()
            running_loss += loss
            
            if i % 100 == 99:
                print('epoch:%d loss: %.3f lr:%.4f'%
                          (epoch + 1, running_loss/100,args.lr))
                running_loss = 0.0
        if epoch % 10==0:
            torch.save(net,'resnet_Adam_epoch_'+str(epoch)+'.pkl')
            #test(classifier_net1,val_loader)
            




def test(net, val_loader):
    mAP = []
    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 256, 256))
        images = Variable(images, volatile=True)
        images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data
        outputs = outputs.view((-1, args.crops, 21))
        outputs = outputs.mean(dim=1).view((-1, 21))

        # score = tnt.meter.mAPMeter(outputs, labels)
        # mAP.append(compute_mAP(labels, outputs))
    
    net.train()


if __name__ == "__main__":
    main()
