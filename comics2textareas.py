#!/usr/bin/env python
# coding: utf-8

"""
comics2textareas.py: Code for the model about text area detection in comics images

This file still needs cleaning.
"""

__author__ = "Martin Nicolas Everaert"
__date__ = "December 2020"

from utils.data_structure import *
from utils.custom_dataset import ComicsForBalloons, ComicsDatasetV2

import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image

    
    

def train():
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
    batch_size = 16 if cuda else 2

    dataloader = DataLoader(
        ComicsForBalloons(file="train.txt", augment = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 16 if cuda else 0
    )
    
    batch_size = 16 if cuda else 2

    dataloader_val = DataLoader(
        ComicsForBalloons(file="validation.txt", augment = False),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 16 if cuda else 0
    )
    
    
    unet = unet.train().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.01)
    
    epoch = 0
    
    loss_train = []
    loss_valid = []
    iou_train = []
    iou_valid = []

    while epoch+1 <= 450:
        print("epoch", epoch+1, "training : ", end="")
        running_loss = 0
        running_iou = 0
        nb_samples = 0
        unet.train()

        for i, batch in enumerate(dataloader):

            # Set model input
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)#.long()


            # -------
            #  Train 
            # -------

            optimizer.zero_grad()

            output = unet(image).squeeze(1)
            #output = torch.cat((1-output, output), 1)

            # Loss
            #print(output.size())
            #print(mask.size())
            loss = criterion(output, mask)

            loss.backward()
            optimizer.step()


            print(i+1, end=" ")
            running_loss += loss.item()


            # IOU https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
            intersection = torch.logical_and(mask, output>0.5)
            union = torch.logical_or(mask, output>0.5)
            iou_score = (torch.sum(intersection).item()+1e-6) / (torch.sum(union).item()+1e-6)
            running_iou += iou_score * image.size()[0]

            nb_samples += image.size()[0]

            '''
            if True:
                plot_image = image[0].cpu()
                # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                plot_image[0] *= 0.229
                plot_image[0] += 0.485
                plot_image[1] *= 0.224
                plot_image[1] += 0.456
                plot_image[2] *= 0.225
                plot_image[2] += 0.406
                #print (torch.min(plot_image[0]), torch.max(plot_image[0]))
                #print (torch.min(plot_image[1]), torch.max(plot_image[1]))
                #print (torch.min(plot_image[2]), torch.max(plot_image[2]))
                m = torch.nn.Threshold(0., 0.) #resizing makes a few values outside 0,1
                plot_image = m(plot_image)
                m = torch.nn.Threshold(-1., -1.)
                plot_image = -m(-plot_image)

                plot_image = plot_image.permute([1,2,0])

                #print (torch.min(output.detach()[0]), torch.max(output.detach()[0]))
                #print (torch.min(mask[0]), torch.max(mask[0]))

                imagesplot = [plot_image, output.cpu().detach()[0], mask[0].cpu()]
                plot(imagesplot, figsize=(16.,10.), nrows_ncols=(1, 3))
            '''

        running_loss /= nb_samples
        running_iou /= nb_samples 
        print('')
        print("epoch:", epoch+1, "training loss over epoch", running_loss, "iou (at 0.5)", running_iou)
        loss_train.append(running_loss)
        iou_train.append(running_iou)

        print("epoch:", epoch+1, "evaluating : ", end="")
        running_loss = 0
        running_iou = 0
        nb_samples = 0
        unet.eval()

        for i, batch in enumerate(dataloader_val):
            with torch.no_grad():
                # Set model input
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)#    .long()    
                # -------
                output = unet(image).squeeze(1)
                #output = torch.cat((1-output, output), 1)
                # Loss
                loss = criterion(output, mask)
                print(i+1, end=" ")
                running_loss += loss.item()
                # IOU https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
                intersection = torch.logical_and(mask, output>0.5)
                union = torch.logical_or(mask, output>0.5)
                iou_score = (torch.sum(intersection).item()+1e-6) / (torch.sum(union).item()+1e-6)
                running_iou += iou_score  * image.size()[0]

                nb_samples += image.size()[0]
        running_loss /= nb_samples
        running_iou /= nb_samples
        print('')
        print("epoch", epoch+1, "validation loss", running_loss, "iou (at 0.5)", running_iou)
        loss_valid.append(running_loss)
        iou_valid.append(running_iou)

        to_save = {
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': loss_train,
                'validation_loss_history': loss_valid,
                'train_iou_history': iou_train,
                'validation_iou_history': iou_valid,
                }
        saving_folder = "models/trained/comics2textareas/"
        os.makedirs(saving_folder, exist_ok=True)
        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))

        epoch += 1

def select_epoch():
        
    epoch = 449
    saving_folder = "models/trained/comics2textareas/"
    checkpoint_file = os.path.join(saving_folder, str(epoch)+".pth")
    print(checkpoint_file)

    assert os.path.exists(checkpoint_file)

    print("FOUND CHECKPOINT")
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    loss_train = checkpoint['train_loss_history']
    loss_valid = checkpoint['validation_loss_history']
    iou_train = checkpoint['train_iou_history']
    iou_valid = checkpoint['validation_iou_history']
    print("LOADED CHECKPOINT EPOCH", epoch+1)
    epoch += 1
    
    smooth = 10
    ln = len(iou_valid)
    x = range(smooth, ln-smooth)

    epoch = max(x, key = lambda i: sum([iou_train[i+j] for j in range(-smooth, smooth+1)]))
    print(epoch+1, iou_valid[epoch])

    
    
    
def optimize_threshold():
    
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    epoch = 427
    
    saving_folder = "models/trained/comics2textareas/"
    checkpoint_file = os.path.join(saving_folder, str(epoch)+".pth")
    print(checkpoint_file)

    assert os.path.exists(checkpoint_file)

    print("FOUND CHECKPOINT")
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("LOADED CHECKPOINT EPOCH", epoch+1)
    
    
    batch_size = 16 if cuda else 2

    dataloader_val = DataLoader(
        ComicsForBalloons(file="validation.txt", augment = False),
        batch_size=batch_size,
        shuffle=True,
        num_workers= 16 if cuda else 0
    )
    
    
    t = [0.01 * i for i in range(101)]
    ious = []
    for threshold in t:
        print("threshold:", threshold, "evaluating : ", end="")
        running_iou = 0
        nb_samples = 0
        unet.eval()
        for i, batch in enumerate(dataloader_val):
            with torch.no_grad():
                # Set model input
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)

                output = unet(image).squeeze(1)

                # IOU https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
                intersection = torch.logical_and(mask, output>threshold)
                union = torch.logical_or(mask, output>threshold)
                iou_score = (torch.sum(intersection).item()+1e-6) / (torch.sum(union).item()+1e-6)
                running_iou += iou_score * image.size()[0]

                nb_samples += image.size()[0]

        running_iou /= nb_samples
        print("iou", running_iou)
        ious.append(running_iou)
    
    smooth = 15
    ln = 100
    t = range(smooth, ln-smooth)

    best = max(t, key = lambda i: sum([ious[i+j] for j in range(-smooth, smooth+1)]))
    print(best, best/100, ious[best])
    threshold =  best/100
    
    
def visualize(dataset):
        
    torch.hub.set_dir(".cache/torch/hub")
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=False)
    unet.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_file = "models/trained/comics2textareas/427.pth"
    checkpoint = torch.load(checkpoint_file, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    unet.eval()
    
    batch_size = 1

    dataloader = DataLoader(
        ComicsDatasetV2(),
        batch_size=batch_size,
        num_workers= 0#1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is", device)
    unet = unet.to(device)

    for n, batch in enumerate(dataloader):
        print(n)

        images = batch["img"]

        images = images.to(device)

        with torch.no_grad():
            prediction = unet(images)

        for i in range(batch["img"].size()[0]):
            pred = prediction[i]
            pred = pred.unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][0][i], batch["size"][1][i]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/balloons427")
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
    
    
def generate_dcm_without_text_areas():
    
    from utils.custom_dataset import Resize, get_random_crop
    import torchvision.transforms as transforms
        
    comics_file = "train.txt"
    list_comics = []
    with open(os.path.join("data/dcm_cropped", comics_file)) as file:
        content = file.read().split("\n")
        list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]       
    t0 = Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=1,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    tensorify = transforms.ToTensor()

    os.makedirs("data/dcm_cropped_noballoons", exist_ok=True)

    for index in range(len(list_comics)):
        print(index, end = " ")
        comics_name = list_comics[index % len(list_comics)]
        image_comics = cv2.imread(comics_name)       
        balloons_name = comics_name.replace("dcm_cropped/images", "dcm_cropped/balloons427").replace(".jpg", "_originalsize.png")        
        image_balloons = cv2.imread(balloons_name)
        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0
        if image_balloons.ndim != 2:
            image_balloons = cv2.cvtColor(image_balloons, cv2.COLOR_BGR2GRAY)        


        #Resize to at least 384*834
        item_comics = t0({"image": image_comics})["image"]
        item_balloons = t0({"image": image_balloons})["image"]

        maxi = 384
        n = maxi
        done = False
        while n>maxi/4 and not done:
            test = 0
            print("("+str(n)+")", end = " ")
            while test<20 and not done:
                #Random crop to exactly n*n     
                item_comics_cropped, item_balloons_cropped = get_random_crop(item_comics, item_balloons, n, n)        
                done = item_balloons_cropped.max() < 0.03*255
                test += 1
            n=n-1
        print(done)
        if done:
            save_image(tensorify(item_comics_cropped), "data/dcm_cropped_noballoons/"+str(index)+".png")




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",        
        "--train",
        action='store_true',
        help="training the Unet"
    )
    parser.add_argument(
        "-se",       
        "--select_epoch",
        action='store_true',
        help="selecting epoch with highest validation IoU at threshold 0.5"
    )
    parser.add_argument(
        "-ot",
        "--optimize_threshold",
        action='store_true',
        help="optimizing the threshold"
    )
    parser.add_argument(
        "-vev",
        "--visualize_ebdtheque_val",
        action='store_true',
        help="visualizing the results on the validation set of the eBDtheque dataset"
    )
    parser.add_argument(
        "-vd",
        "--visualize_dcm",
        action='store_true',
        help="visualizing the results on the dcm dataset"
    )
    parser.add_argument(
        "-gdwta",
        "--generate_dcm_without_text_areas",
        action='store_true',
        help="using the trained model to generate a dataset of comics without text areas"
    )
    args = parser.parse_args()

    if args.train:
        train()
    if args.select_epoch:
        select_epoch()
    if args.optimize_threshold:
        optimize_threshold()
    if args.visualize_ebdtheque_val:
        visualize("eBDtheque")
    if args.visualize_dcm:
        visualize("dcm")
    if args.generate_dcm_without_text_areas:
        generate_dcm_without_text_areas()