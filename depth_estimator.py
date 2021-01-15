#!/usr/bin/env python
# coding: utf-8

"""
depth_estimator.py: Code about the depth estimation on top of the CycleGAN between natural and comics images
"""

__author__ = "Martin Nicolas Everaert"
__date__ = "December 2020"

from utils.data_structure import *
from utils.custom_dataset import CustomDataset

def optim_simple():
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "depth_simple_lr"
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)



    from utils.custom_dataset import ImageDataset_naturalWithDepth
    from models.models import GeneratorResNet


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")




    for lr in [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:


        # Initialize generators and discriminators
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        natural2comics = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
        comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")



        # Losses 
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        criterion_depth = torch.nn.L1Loss()



        # Moving to cuda if available
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        natural2comics = natural2comics.to(device)
        comics2depth = comics2depth.to(device)
        criterion_depth = criterion_depth.to(device)



        # Optimizers
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        optimizer_comics2depth = torch.optim.Adam(comics2depth.parameters(), lr=lr)



        checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_simple", "99.pth")
        checkpoint = torch.load(checkpoint_file, map_location = device)
        natural2comics.load_state_dict(checkpoint['natural2comics_state_dict'])
        natural2comics = natural2comics.to(device)
        natural2comics.eval()


        comics2depth.train()



        batch_size=16 if cuda else 1

        dataloader = DataLoader(
            ImageDataset_naturalWithDepth(unaligned = True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=16 if cuda else 0
        )

        natural2comics.eval()
        comics2depth.train()

        # Training
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        mean = torch.Tensor(mean).view(3,1,1).to(device)
        std = torch.Tensor(std).view(3,1,1).to(device)

        tosave = ""

        for i, batch in enumerate(dataloader):

            real_natural = batch["natural"].to(device)
            real_natural_depth = batch["natural_depth"].to(device)

            with torch.no_grad():
                fake_comics = natural2comics(real_natural).detach()
                fake_comics -= mean
                fake_comics /= std


            optimizer_comics2depth.zero_grad()
            pred_depth = comics2depth(fake_comics.detach())
            loss = criterion_depth(pred_depth, real_natural_depth)

            loss.backward()
            optimizer_comics2depth.step()



            # ----------

            print("lr:", str(lr), "i:", i+1,"loss:", loss.item())
            tosave += str(loss.item())+"\n"



        with open(os.path.join(saving_folder, "lr"+str(lr)+".txt"), "w") as file:
            file.write(tosave)

    
def optim_add_text():
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "depth_add_text_lr"
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    from utils.custom_dataset import ImageDataset_naturalWithDepth_comicsWithBalloons
    from models.models import GeneratorResNet



    # In[ ]:


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")




    for lr in [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:

        # In[ ]:


        # Initialize generators and discriminators
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        natural2comicsNoB = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
        comicsNoB2comics = GeneratorResNet(input_shape=(7, 384, 384), out_channels=3, num_residual_blocks=8)
        comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")


        # In[ ]:


        # Losses 
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        criterion_depth = torch.nn.L1Loss()


        # In[ ]:


        # Moving to cuda if available
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        natural2comicsNoB = natural2comicsNoB.to(device)
        comicsNoB2comics = comicsNoB2comics.to(device)
        comics2depth = comics2depth.to(device)
        criterion_depth = criterion_depth.to(device)


        # In[ ]:


        # Optimizers
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
        optimizer_comics2depth = torch.optim.Adam(comics2depth.parameters(), lr=lr)

        # In[ ]:


        checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_without_text", "99.pth")
        checkpoint = torch.load(checkpoint_file, map_location = device)
        natural2comicsNoB.load_state_dict(checkpoint['natural2comics_state_dict'])
        natural2comicsNoB = natural2comicsNoB.to(device)
        natural2comicsNoB.eval()

        # In[ ]:


        checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_add_text", "99.pth")
        checkpoint = torch.load(checkpoint_file, map_location = device)
        comicsNoB2comics.load_state_dict(checkpoint['generator_comicsB_state_dict'])
        comicsNoB2comics = comicsNoB2comics.to(device)
        comicsNoB2comics.eval()


        # In[ ]:
        comics2depth.train()



        # In[ ]:


        batch_size=16 if cuda else 1

        dataloader = DataLoader(
            ImageDataset_naturalWithDepth_comicsWithBalloons(unaligned = True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=16 if cuda else 0
        )

        # In[ ]:
        natural2comicsNoB.eval()
        comicsNoB2comics.eval()
        comics2depth.train()

        # Training
        # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        mean = torch.Tensor(mean).view(3,1,1).to(device)
        std = torch.Tensor(std).view(3,1,1).to(device)

        tosave = ""

        for i, batch in enumerate(dataloader):

            real_natural = batch["natural"].to(device)
            real_natural_depth = batch["natural_depth"].to(device)
            real_comicsB = batch["comics"].to(device)
            real_balloons = batch["balloons"].to(device).unsqueeze(1)

            with torch.no_grad():
                comicsNoB = natural2comicsNoB(real_natural).detach()
                inp = torch.cat((real_comicsB, real_balloons, comicsNoB), dim=1)
                fake_comicsB = comicsNoB2comics(inp).detach()
                fake_comicsB -= mean
                fake_comicsB /= std


            optimizer_comics2depth.zero_grad()
            pred_depth = comics2depth(fake_comicsB.detach())
            loss = criterion_depth(pred_depth, real_natural_depth)

            loss.backward()
            optimizer_comics2depth.step()



            # ----------

            print("lr:", str(lr), "i:", i+1,"loss:", loss.item())
            tosave += str(loss.item())+"\n"



        with open(os.path.join(saving_folder, "lr"+str(lr)+".txt"), "w") as file:
            file.write(tosave)


    
def train_simple(lr):
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "depth_simple_"+str(lr)
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    from utils.custom_dataset import ImageDataset_naturalWithDepth
    from models.models import GeneratorResNet

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")


    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comics = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
    comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")

    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_depth = torch.nn.L1Loss()


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comics = natural2comics.to(device)
    comics2depth = comics2depth.to(device)
    criterion_depth = criterion_depth.to(device)


    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_comics2depth = torch.optim.Adam(comics2depth.parameters(), lr=lr)

    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_simple", "99.pth")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    natural2comics.load_state_dict(checkpoint['natural2comics_state_dict'])
    natural2comics = natural2comics.to(device)
    natural2comics.eval()

    comics2depth.train()


    batch_size=16 if cuda else 1

    dataloader = DataLoader(
        ImageDataset_naturalWithDepth(unaligned = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16 if cuda else 0
    )



    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])
        optimizer_comics2depth.load_state_dict(checkpoint['optimizer_comics2depth_state_dict'])         
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)

    natural2comics.eval()
    comics2depth.train()

    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)


    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            real_natural = batch["natural"].to(device)
            real_natural_depth = batch["natural_depth"].to(device)

            with torch.no_grad():
                fake_comics = natural2comics(real_natural).detach()
                fake_comics -= mean
                fake_comics /= std


            optimizer_comics2depth.zero_grad()
            pred_depth = comics2depth(fake_comics.detach())
            loss = criterion_depth(pred_depth, real_natural_depth)

            loss.backward()
            optimizer_comics2depth.step()


            # ----------

            print("epoch:", epoch+1, "i:", i+1,
                  "loss:", loss.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2depth_state_dict': comics2depth.state_dict(),
                'optimizer_comics2depth_state_dict': optimizer_comics2depth.state_dict(),
                }

        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1




def train_add_text(lr):
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os

    unique_name = "depth_add_text_"+str(lr)
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)
    
    from utils.custom_dataset import ImageDataset_naturalWithDepth_comicsWithBalloons
    from models.models import GeneratorResNet
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")
    
    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
    comicsNoB2comics = GeneratorResNet(input_shape=(7, 384, 384), out_channels=3, num_residual_blocks=8)
    comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")



    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_depth = torch.nn.L1Loss()



    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = natural2comicsNoB.to(device)
    comicsNoB2comics = comicsNoB2comics.to(device)
    comics2depth = comics2depth.to(device)
    criterion_depth = criterion_depth.to(device)



    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_comics2depth = torch.optim.Adam(comics2depth.parameters(), lr=lr)



    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_without_text", "99.pth")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    natural2comicsNoB.load_state_dict(checkpoint['natural2comics_state_dict'])
    natural2comicsNoB = natural2comicsNoB.to(device)
    natural2comicsNoB.eval()


    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_add_text", "99.pth")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    comicsNoB2comics.load_state_dict(checkpoint['generator_comicsB_state_dict'])
    comicsNoB2comics = comicsNoB2comics.to(device)
    comicsNoB2comics.eval()


    comics2depth.train()



    batch_size=16 if cuda else 1

    dataloader = DataLoader(
        ImageDataset_naturalWithDepth_comicsWithBalloons(unaligned = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16 if cuda else 0
    )




    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])
        optimizer_comics2depth.load_state_dict(checkpoint['optimizer_comics2depth_state_dict'])         
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)


    natural2comicsNoB.eval()
    comicsNoB2comics.eval()
    comics2depth.train()

    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)


    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            real_natural = batch["natural"].to(device)
            real_natural_depth = batch["natural_depth"].to(device)
            real_comicsB = batch["comics"].to(device)
            real_balloons = batch["balloons"].to(device).unsqueeze(1)

            with torch.no_grad():
                comicsNoB = natural2comicsNoB(real_natural).detach()
                inp = torch.cat((real_comicsB, real_balloons, comicsNoB), dim=1)
                fake_comicsB = comicsNoB2comics(inp).detach()
                fake_comicsB -= mean
                fake_comicsB /= std


            optimizer_comics2depth.zero_grad()
            pred_depth = comics2depth(fake_comicsB.detach())
            loss = criterion_depth(pred_depth, real_natural_depth)

            loss.backward()
            optimizer_comics2depth.step()


            # ----------

            print("epoch:", epoch+1, "i:", i+1,
                  "loss:", loss.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2depth_state_dict': comics2depth.state_dict(),
                'optimizer_comics2depth_state_dict': optimizer_comics2depth.state_dict(),
                }

        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1
        
        
        
def train_add_text_ignoretext(lr):
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os

    unique_name = "depth_add_text_ignoretext_"+str(lr)
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)

    from utils.custom_dataset import ImageDataset_naturalWithDepth_comicsWithBalloons
    from models.models import GeneratorResNet

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")

    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
    comicsNoB2comics = GeneratorResNet(input_shape=(7, 384, 384), out_channels=3, num_residual_blocks=8)
    comics2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS")


    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_depth = torch.nn.L1Loss()



    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = natural2comicsNoB.to(device)
    comicsNoB2comics = comicsNoB2comics.to(device)
    comics2depth = comics2depth.to(device)
    criterion_depth = criterion_depth.to(device)


    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_comics2depth = torch.optim.Adam(comics2depth.parameters(), lr=lr)


    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_without_text", "99.pth")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    natural2comicsNoB.load_state_dict(checkpoint['natural2comics_state_dict'])
    natural2comicsNoB = natural2comicsNoB.to(device)
    natural2comicsNoB.eval()



    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_add_text", "99.pth")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    comicsNoB2comics.load_state_dict(checkpoint['generator_comicsB_state_dict'])
    comicsNoB2comics = comicsNoB2comics.to(device)
    comicsNoB2comics.eval()


    comics2depth.train()



    batch_size=16 if cuda else 1

    dataloader = DataLoader(
        ImageDataset_naturalWithDepth_comicsWithBalloons(unaligned = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16 if cuda else 0
    )



    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])
        optimizer_comics2depth.load_state_dict(checkpoint['optimizer_comics2depth_state_dict'])         
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)


    natural2comicsNoB.eval()
    comicsNoB2comics.eval()
    comics2depth.train()

    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)


    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            real_natural = batch["natural"].to(device)
            real_natural_depth = batch["natural_depth"].to(device)
            real_comicsB = batch["comics"].to(device)
            real_balloons = batch["balloons"].to(device).unsqueeze(1)

            with torch.no_grad():
                comicsNoB = natural2comicsNoB(real_natural).detach()
                inp = torch.cat((real_comicsB, real_balloons, comicsNoB), dim=1)
                fake_comicsB = comicsNoB2comics(inp).detach()
                fake_comicsB -= mean
                fake_comicsB /= std


            optimizer_comics2depth.zero_grad()
            pred_depth = comics2depth(fake_comicsB.detach())
            loss = criterion_depth(pred_depth*(1-real_balloons), real_natural_depth*(1-real_balloons))

            loss.backward()
            optimizer_comics2depth.step()


            # ----------

            print("epoch:", epoch+1, "i:", i+1,
                  "loss:", loss.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2depth_state_dict': comics2depth.state_dict(),
                'optimizer_comics2depth_state_dict': optimizer_comics2depth.state_dict(),
                }

        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1
        
def depth_simple_generatedeptheval(lr):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os
    from utils.custom_dataset import ComicsDatasetV2


    unique_name = "approach2"+str(lr)

    import models.models


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
        ComicsDatasetV2(train = False, val = True),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )


    #We want to evaluate epoch 100
    epoch = 100
    epoch -= 1

    checkpoint_file = os.path.join("models/trained/depth_simple_"+str(lr), str(epoch)+".pth")
    assert os.path.exists(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])             
    epoch = checkpoint['epoch']
    print("LOADED CHECKPOINT EPOCH", epoch+1)

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

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/"+unique_name+"epoch"+str(epoch))
            print (new_name)
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))

def depth_add_text_generatedeptheval(lr):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os
    from utils.custom_dataset import ComicsDatasetV2


    unique_name = "approach2v1"+str(lr)


    import models.models



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
        ComicsDatasetV2(train = False, val = True),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )


    #We want to evaluate epoch 100
    epoch = 100
    epoch -= 1

    checkpoint_file = os.path.join("models/trained/depth_add_text_"+str(lr), str(epoch)+".pth")
    assert os.path.exists(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])             
    epoch = checkpoint['epoch']
    print("LOADED CHECKPOINT EPOCH", epoch+1)

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

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/"+unique_name+"epoch"+str(epoch))
            print (new_name)
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))

def depth_add_text_ignoretext_generatedeptheval(lr):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os
    from utils.custom_dataset import ComicsDatasetV2


    unique_name = "approach2v2"+str(lr)

    import models.models


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
        ComicsDatasetV2(train = False, val = True),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )




    #We want to evaluate epoch 100
    epoch = 100
    epoch -= 1

    checkpoint_file = os.path.join("models/trained/depth_add_text_ignoretext_"+str(lr), str(epoch)+".pth")
    assert os.path.exists(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    comics2depth.load_state_dict(checkpoint['comics2depth_state_dict'])             
    epoch = checkpoint['epoch']
    print("LOADED CHECKPOINT EPOCH", epoch+1)

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

            new_name = batch["name"][i].replace("dcm_cropped/images", "dcm_cropped/"+unique_name+"epoch"+str(epoch))
            print (new_name)
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            save_image(pred.squeeze(0).cpu(), new_name.replace(".jpg", ".png"))
            save_image(pred_resized.squeeze(0).cpu(), new_name.replace(".jpg", "_originalsize.png"))
            with open(new_name.replace(".jpg", ".txt"), "w+") as file:
                file.write(str(maxi.item()))
                
def optimize_lr(model):
    if model == "simple":
        optim_simple()
    elif model == "add_text":
        optim_add_text()
    else:
        raise NotImplementedError
        
def train(model, lr):
    if model == "simple":
        train_simple(lr)
    elif model == "add_text":
        train_add_text(lr)
    elif model == "add_text_ignoretext":
        train_add_text_ignoretext(lr)
    else:
        raise NotImplementedError
        
        
def generate_depth_images(model, lr):
    if model == "simple":
        depth_simple_generatedeptheval(lr)
    elif model == "add_text":
        depth_add_text_generatedeptheval(lr)
    elif model == "add_text_ignoretext":
        depth_add_text_ignoretext_generatedeptheval(lr)
    else:
        raise NotImplementedError
    
    
    
if __name__ == "__main__":
    
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        type=str,
                        help="which model to use (simple, add_text, add_text_ignoretext)")
    
    group = parser.add_mutually_exclusive_group()    
    group.add_argument(
        "--train",
        help="training the model",
        action="store_true"
    )
    group.add_argument(
        "--optimize_lr",
        help="optimizing the learning rate",
        action="store_true"
    )
    group.add_argument(
        "--generate_depth_images",
        help="generating depth images (trained depth estimator on comics)",
        action="store_true"
    )
        
    args = parser.parse_args()

    if args.optimize_lr:
        optimize_lr(args.model)
    if args.train:
        train(args.model, lr=1e-6)
    if args.generate_depth_images:
        generate_depth_images(args.model, lr=1e-6)