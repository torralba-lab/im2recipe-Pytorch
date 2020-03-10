import time
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.optim
# import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
# from data_loader import ImagerLoader # our data_loader
import numpy as np
from trijoint import im2recipe
import pickle
from args import get_parser
from PIL import Image
import sys
import os

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda',0))

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def main():
   
    im_path = opts.test_image_path
    ext = os.path.basename(im_path).split('.')[-1]
    if ext not in ['jpeg','jpg','png']:
        raise Exception("Wrong image format.")

    # create model
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)

    # load checkpoint
    print("=> loading checkpoint '{}'".format(opts.model_path))
    if device.type=='cpu':
        checkpoint = torch.load(opts.model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(opts.model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                normalize])

    # load image
    im = Image.open(im_path).convert('RGB')
    im = transform(im)
    im = im.view((1,)+im.shape)
    # get model output
    output = model.visionMLP(im)
    output = norm(output)
    output = output.data.cpu().numpy()
    # save output
    with open(im_path.replace(ext,'pkl'), 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()
