#!/usr/bin/env python
# coding: utf-8

"""
baseline.py: Code about the baseline models based on MiDaS to train the model and generate the depth images.
"""

__author__ = "Martin Nicolas Everaert"
__date__ = "December 2020"

from utils.data_structure import *
from utils.custom_dataset import ComicsDatasetV2
    
    
def batchnorm_trick_train():
    import glob
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from torch.utils.data import Dataset, DataLoader
    import os
    
    torch.hub.set_dir(".cache/torch/hub")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.train()
    
    def recursive_batch_norm_update(child, value=0.1/24):
        # https://discuss.pytorch.org/t/update-scheme-for-batchnorm-momentum/23755/5
        if type(child) == torch.nn.BatchNorm2d:
            child.momentum = value
            return    
        for children in child.children():
            lowest_child = recursive_batch_norm_update(children, value)    
        return
    
    recursive_batch_norm_update(midas, value=0.002)
    
    
    batch_size = 1
    dataloader = DataLoader(
        ComicsDatasetV2(train=True, val=True, test=False),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is", device)
    midas = midas.to(device)

    for n, batch in enumerate(dataloader):

        images = batch["img"]

        images = images.to(device)

        with torch.no_grad():
            prediction = midas(images)
            
    os.makedirs("models/trained/batchnorm_trick", exist_ok=True)
    torch.save({'state_dict':midas.state_dict()}, "models/trained/batchnorm_trick/batchnorm_trick.pth") 


def batchnorm_trick_generate():
    
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os

    unique_name = "depth_batchnorm_trick"

    import models.models
    import utils.custom_dataset


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")



    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    
    checkpoint_file = "models/trained/batchnorm_trick/batchnorm_trick.pth"
    checkpoint = torch.load(checkpoint_file, map_location = device)
    comics2depth.load_state_dict(checkpoint['state_dict'])
    comics2depth.eval()


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2depth = comics2depth.to(device)


    batch_size = 1

    dataloader = DataLoader(
        utils.custom_dataset.ComicsDatasetV2(train = False, val = True),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )


    comics2depth.eval()


    for n, batch in enumerate(dataloader):
        #print(n)

        images = batch["img"]

        images = images.to(device)

        with torch.no_grad():
            prediction = comics2depth(images)

        #print(images.data)
        #print(prediction.data)

        for i in range(batch["img"].size()[0]):
            maxi = torch.max(prediction[i].view(-1))
            pred = prediction[i]/maxi
            pred = pred.unsqueeze(0).unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][0][i], batch["size"][1][i]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/"+unique_name)
            print (new_name)
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))

def no_batchnorm_trick_generate():

    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os

    unique_name = "depth_no_batchnorm_trick"

    import models.models
    import utils.custom_dataset


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")



    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2depth = comics2depth.to(device)


    batch_size = 1

    dataloader = DataLoader(
        utils.custom_dataset.ComicsDatasetV2(train = False, val = True),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )


    comics2depth.eval()


    for n, batch in enumerate(dataloader):
        #print(n)

        images = batch["img"]

        images = images.to(device)

        with torch.no_grad():
            prediction = comics2depth(images)

        #print(images.data)
        #print(prediction.data)

        for i in range(batch["img"].size()[0]):
            maxi = torch.max(prediction[i].view(-1))
            pred = prediction[i]/maxi
            pred = pred.unsqueeze(0).unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][0][i], batch["size"][1][i]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/"+unique_name)
            print (new_name)
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))







def train(model):
    if model == "batchnorm_trick":
        batchnorm_trick_train()
    else:
        raise NotImplementedError

def generate_depth_images(model):
    if model == "no_batchnorm_trick":
        no_batchnorm_trick_generate()
    elif model == "batchnorm_trick":
        batchnorm_trick_generate()
    else:
        raise NotImplementedError


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        type=str,
                        help="which model to use (no_batchnorm_trick, batchnorm_trick)")
    
    group = parser.add_mutually_exclusive_group()    
    group.add_argument(
        "--train",
        help="training the model",
        action="store_true"
    )
    group.add_argument(
        "--generate_depth_images",
        help="generating depth images",
        action="store_true"
    )
    
    
    args = parser.parse_args()

    if args.train:
        train(args.model)
    if args.generate_depth_images:
        generate_depth_images(args.model)