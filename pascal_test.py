import os, sys, numpy as np
import argparse
import time
from sklearn.metrics import average_precision_score
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PascalLoader import DataLoader
import torch.nn as nn
import torch.nn.functional as F


def main():
    
    sys.path.append('./model/')

    parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2007')
    parser.add_argument('-pascal_train_path', default='F:/datasets/VOC2012',
                        help='Path to Pascal VOC 2007 folder')
    parser.add_argument('--crops', default=10, type=int, help='number of random crops during testing')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
    ])
    val_data = DataLoader(args.pascal_train_path, 'val', transform=val_transform, random_crops=args.crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=1,
                                             shuffle=False)

    net = torch.load('./')
    print(net)
    net.cuda()
    net.eval()

    test(net, val_loader, args)


def test( net, val_loader, args):
    mAP = []
    for i, (images, labels) in enumerate(val_loader):
        print(i)
        images = images.view((-1, 3, 256, 256))
        #print(images.shape)
        images = Variable(images)
        images = images.cuda()
        # Forward + Backward + Optimiz
		
        encode_feature,_= net.encoder(images)
        rec_img = net.decoder(encode_feature)
        outputs = net.classifier(encode_feature)
        #outputs = net(images)
        #eva_start = time.clock()
        outputs = outputs.data.cpu()
        outputs = outputs.view((-1, args.crops, 21))
        outputs = outputs.mean(dim=1).view((-1, 21))
        mAP.append(compute_mAP(labels, outputs))

    print('finish test')
    print('TESTING:  mAP %.2f%%' % (100 * np.mean(mAP)))

def compute_mAP(labels, outputs):
    y_true = labels.cpu()
    y_pred = outputs.cpu()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)

if __name__ == "__main__":
    main()
