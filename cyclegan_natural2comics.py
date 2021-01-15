#!/usr/bin/env python
# coding: utf-8

"""
cyclegan_natural2comics.py: Code about the CycleGANs between natural and comics images
"""

__author__ = "Martin Nicolas Everaert"
__date__ = "December 2020"

from utils.data_structure import *
from utils.custom_dataset import CustomDataset




def train_simple():
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "cyclegan_natural_comics_simple"
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    import models.models

    from utils.custom_dataset import ImageDataset_natural_comics


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")



    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    natural2comics = models.models.GeneratorResNet()
    discrim_natural = models.models.Discriminator(input_shape = (3, 384, 384))
    discrim_comics = models.models.Discriminator(input_shape = (3, 384, 384))




    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()




    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)
    natural2comics = natural2comics.to(device)
    discrim_natural = discrim_natural.to(device)
    discrim_comics = discrim_comics.to(device)
    criterion_GAN = criterion_GAN.to(device)
    criterion_cycle = criterion_cycle.to(device)
    criterion_identity = criterion_identity.to(device)



    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_G = torch.optim.Adam(
        itertools.chain(comics2natural.parameters(), natural2comics.parameters()))
    optimizer_discrim_natural = torch.optim.Adam(discrim_natural.parameters())
    optimizer_discrim_comics = torch.optim.Adam(discrim_comics.parameters())




    dataloader = DataLoader(
        ImageDataset_natural_comics(unaligned = True),
        batch_size=4 if cuda else 1,
        shuffle=True,
        num_workers=8 if cuda else 0
    )




    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])
        natural2comics.load_state_dict(checkpoint['natural2comics_state_dict'])
        discrim_natural.load_state_dict(checkpoint['discrim_natural_state_dict'])
        discrim_comics.load_state_dict(checkpoint['discrim_comics_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_discrim_natural.load_state_dict(checkpoint['optimizer_discrim_natural_state_dict'])
        optimizer_discrim_comics.load_state_dict(checkpoint['optimizer_discrim_comics_state_dict'])             
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)


    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            # Set model input
            real_natural = batch["natural"].to(device)
            real_comics = batch["comics"].to(device)

            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_natural.size(0), *discrim_natural.output_shape))).to(device)
            fake = torch.Tensor(np.zeros((real_natural.size(0), *discrim_natural.output_shape))).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            natural2comics.train()
            comics2natural.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_natural = criterion_identity(comics2natural(real_natural), real_natural)
            loss_id_comics = criterion_identity(natural2comics(real_comics), real_comics)

            loss_identity = (loss_id_natural + loss_id_comics) / 2

            # GAN loss
            fake_comics = natural2comics(real_natural)
            loss_GAN_natural2comics = criterion_GAN(discrim_comics(fake_comics), valid)
            fake_natural = comics2natural(real_comics)
            loss_GAN_comics2natural = criterion_GAN(discrim_natural(fake_natural), valid)

            loss_GAN = (loss_GAN_natural2comics + loss_GAN_comics2natural) / 2



            # Cycle loss
            recov_natural = comics2natural(fake_comics)
            loss_cycle_natural = criterion_cycle(recov_natural, real_natural)
            recov_comics = natural2comics(fake_natural)
            loss_cycle_comics = criterion_cycle(recov_comics, real_comics)

            loss_cycle = (loss_cycle_natural + loss_cycle_comics) / 2

            # Total loss
            loss_G = loss_GAN + 10 * loss_cycle + 5 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator natural
            # -----------------------
            optimizer_discrim_natural.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_natural(real_natural), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_natural(fake_natural.detach()), fake)
            # Total loss
            loss_discrim_natural = (loss_real + loss_fake) / 2

            loss_discrim_natural.backward()
            optimizer_discrim_natural.step()

            # -----------------------
            #  Train Discriminator comics
            # -----------------------
            optimizer_discrim_comics.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_comics(real_comics), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_comics(fake_comics.detach()), fake)
            # Total loss
            loss_discrim_comics = (loss_real + loss_fake) / 2

            loss_discrim_comics.backward()
            optimizer_discrim_comics.step()

            loss_D = (loss_discrim_natural + loss_discrim_comics) / 2



            # ----------
            print("epoch:", epoch+1, "i:", i+1,
                  "GANn2c:", loss_GAN_natural2comics.item(), "GANc2n:", loss_GAN_comics2natural.item(), "cycle:", loss_cycle.item(), "identity:", loss_identity.item(),
                  "Dn:", loss_discrim_natural.item(),"Dc:", loss_discrim_comics.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2natural_state_dict': comics2natural.state_dict(),
                'natural2comics_state_dict': natural2comics.state_dict(),
                'discrim_natural_state_dict': discrim_natural.state_dict(),
                'discrim_comics_state_dict': discrim_comics.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_discrim_natural_state_dict': optimizer_discrim_natural.state_dict(),
                'optimizer_discrim_comics_state_dict': optimizer_discrim_comics.state_dict(),
                }
        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1
        
        
        
def train_depth_aware():
    
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os

    unique_name = "cyclegan_natural_comics_depth_aware"
    saving_folder = "models/trained"+unique_name
    os.makedirs(saving_folder, exist_ok=True)

    import models.models
    from utils.custom_dataset import ImageDataset_natural_comics

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")



    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    natural2comics = models.models.GeneratorResNet()
    discrim_natural = models.models.Discriminator(input_shape = (4, 384, 384))
    discrim_comics = models.models.Discriminator(input_shape = (3, 384, 384))


    natural2depth = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(device).eval()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)


    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)
    natural2comics = natural2comics.to(device)
    discrim_natural = discrim_natural.to(device)
    discrim_comics = discrim_comics.to(device)
    criterion_GAN = criterion_GAN.to(device)
    criterion_cycle = criterion_cycle.to(device)
    criterion_identity = criterion_identity.to(device)


    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_G = torch.optim.Adam(
        itertools.chain(comics2natural.parameters(), natural2comics.parameters()))
    optimizer_discrim_natural = torch.optim.Adam(discrim_natural.parameters())
    optimizer_discrim_comics = torch.optim.Adam(discrim_comics.parameters())


    dataloader = DataLoader(
        ImageDataset_natural_comics(unaligned = True),
        batch_size=4 if cuda else 1,
        shuffle=True,
        num_workers=8 if cuda else 0
    )


    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])
        natural2comics.load_state_dict(checkpoint['natural2comics_state_dict'])
        discrim_natural.load_state_dict(checkpoint['discrim_natural_state_dict'])
        discrim_comics.load_state_dict(checkpoint['discrim_comics_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_discrim_natural.load_state_dict(checkpoint['optimizer_discrim_natural_state_dict'])
        optimizer_discrim_comics.load_state_dict(checkpoint['optimizer_discrim_comics_state_dict'])             
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)



    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            # Set model input
            real_natural = batch["natural"].to(device)
            real_comics = batch["comics"].to(device)

            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_natural.size(0), *discrim_natural.output_shape))).to(device)
            fake = torch.Tensor(np.zeros((real_natural.size(0), *discrim_natural.output_shape))).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            natural2comics.train()
            comics2natural.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_natural = criterion_identity(comics2natural(real_natural), real_natural)
            loss_id_comics = criterion_identity(natural2comics(real_comics), real_comics)

            loss_identity = (loss_id_natural + loss_id_comics) / 2

            # GAN loss


            fake_comics = natural2comics(real_natural)
            loss_GAN_natural2comics = criterion_GAN(discrim_comics(fake_comics), valid)
            fake_natural = comics2natural(real_comics)
            fake_natural_with_depth = torch.cat((fake_natural, natural2depth((fake_natural-mean)/std).unsqueeze(1)), dim=1)
            loss_GAN_comics2natural = criterion_GAN(discrim_natural(fake_natural_with_depth), valid)

            loss_GAN = (loss_GAN_natural2comics + loss_GAN_comics2natural) / 2



            # Cycle loss
            recov_natural = comics2natural(fake_comics)
            loss_cycle_natural = criterion_cycle(recov_natural, real_natural)
            recov_comics = natural2comics(fake_natural)
            loss_cycle_comics = criterion_cycle(recov_comics, real_comics)

            loss_cycle = (loss_cycle_natural + loss_cycle_comics) / 2

            # Total loss
            loss_G = loss_GAN + 10 * loss_cycle + 5 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator natural
            # -----------------------
            optimizer_discrim_natural.zero_grad()

            # Real loss
            real_natural_with_depth = torch.cat((real_natural, natural2depth((real_natural-mean)/std).unsqueeze(1)), dim=1)
            loss_real = criterion_GAN(discrim_natural(real_natural_with_depth), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_natural(fake_natural_with_depth.detach()), fake)
            # Total loss
            loss_discrim_natural = (loss_real + loss_fake) / 2

            loss_discrim_natural.backward()
            optimizer_discrim_natural.step()

            # -----------------------
            #  Train Discriminator comics
            # -----------------------
            optimizer_discrim_comics.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_comics(real_comics), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_comics(fake_comics.detach()), fake)
            # Total loss
            loss_discrim_comics = (loss_real + loss_fake) / 2

            loss_discrim_comics.backward()
            optimizer_discrim_comics.step()

            loss_D = (loss_discrim_natural + loss_discrim_comics) / 2



            # ----------
            print("epoch:", epoch+1, "i:", i+1,
                  "GANn2c:", loss_GAN_natural2comics.item(), "GANc2n:", loss_GAN_comics2natural.item(), "cycle:", loss_cycle.item(), "identity:", loss_identity.item(),
                  "Dn:", loss_discrim_natural.item(),"Dc:", loss_discrim_comics.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2natural_state_dict': comics2natural.state_dict(),
                'natural2comics_state_dict': natural2comics.state_dict(),
                'discrim_natural_state_dict': discrim_natural.state_dict(),
                'discrim_comics_state_dict': discrim_comics.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_discrim_natural_state_dict': optimizer_discrim_natural.state_dict(),
                'optimizer_discrim_comics_state_dict': optimizer_discrim_comics.state_dict(),
                }
        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1

    
    
def train_without_text():
    
    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "cyclegan_natural_comics_without_text"
    saving_folder = "models/trained/"+unique_name
    os.makedirs(saving_folder, exist_ok=True)


    import models.models

    from utils.custom_dataset import ImageDataset_natural_comicsnoballoons


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")


    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    natural2comics = models.models.GeneratorResNet()
    discrim_natural = models.models.Discriminator(input_shape = (3, 384, 384))
    discrim_comics = models.models.Discriminator(input_shape = (3, 384, 384))

    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)
    natural2comics = natural2comics.to(device)
    discrim_natural = discrim_natural.to(device)
    discrim_comics = discrim_comics.to(device)
    criterion_GAN = criterion_GAN.to(device)
    criterion_cycle = criterion_cycle.to(device)
    criterion_identity = criterion_identity.to(device)


    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_G = torch.optim.Adam(
        itertools.chain(comics2natural.parameters(), natural2comics.parameters()))
    optimizer_discrim_natural = torch.optim.Adam(discrim_natural.parameters())
    optimizer_discrim_comics = torch.optim.Adam(discrim_comics.parameters())



    dataloader = DataLoader(
        ImageDataset_natural_comicsnoballoons(unaligned = True),
        batch_size=4 if cuda else 1,
        shuffle=True,
        num_workers=8 if cuda else 0
    )


    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])
        natural2comics.load_state_dict(checkpoint['natural2comics_state_dict'])
        discrim_natural.load_state_dict(checkpoint['discrim_natural_state_dict'])
        discrim_comics.load_state_dict(checkpoint['discrim_comics_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_discrim_natural.load_state_dict(checkpoint['optimizer_discrim_natural_state_dict'])
        optimizer_discrim_comics.load_state_dict(checkpoint['optimizer_discrim_comics_state_dict'])             
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)


    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            # Set model input
            real_natural = batch["natural"].to(device)
            real_comics = batch["comics"].to(device)

            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_natural.size(0), *discrim_natural.output_shape))).to(device)
            fake = torch.Tensor(np.zeros((real_natural.size(0), *discrim_natural.output_shape))).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            natural2comics.train()
            comics2natural.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_natural = criterion_identity(comics2natural(real_natural), real_natural)
            loss_id_comics = criterion_identity(natural2comics(real_comics), real_comics)

            loss_identity = (loss_id_natural + loss_id_comics) / 2

            # GAN loss
            fake_comics = natural2comics(real_natural)
            loss_GAN_natural2comics = criterion_GAN(discrim_comics(fake_comics), valid)
            fake_natural = comics2natural(real_comics)
            loss_GAN_comics2natural = criterion_GAN(discrim_natural(fake_natural), valid)

            loss_GAN = (loss_GAN_natural2comics + loss_GAN_comics2natural) / 2



            # Cycle loss
            recov_natural = comics2natural(fake_comics)
            loss_cycle_natural = criterion_cycle(recov_natural, real_natural)
            recov_comics = natural2comics(fake_natural)
            loss_cycle_comics = criterion_cycle(recov_comics, real_comics)

            loss_cycle = (loss_cycle_natural + loss_cycle_comics) / 2

            # Total loss
            loss_G = loss_GAN + 10 * loss_cycle + 5 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator natural
            # -----------------------
            optimizer_discrim_natural.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_natural(real_natural), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_natural(fake_natural.detach()), fake)
            # Total loss
            loss_discrim_natural = (loss_real + loss_fake) / 2

            loss_discrim_natural.backward()
            optimizer_discrim_natural.step()

            # -----------------------
            #  Train Discriminator comics
            # -----------------------
            optimizer_discrim_comics.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_comics(real_comics), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_comics(fake_comics.detach()), fake)
            # Total loss
            loss_discrim_comics = (loss_real + loss_fake) / 2

            loss_discrim_comics.backward()
            optimizer_discrim_comics.step()

            loss_D = (loss_discrim_natural + loss_discrim_comics) / 2



            # ----------
            print("epoch:", epoch+1, "i:", i+1,
                  "GANn2c:", loss_GAN_natural2comics.item(), "GANc2n:", loss_GAN_comics2natural.item(), "cycle:", loss_cycle.item(), "identity:", loss_identity.item(),
                  "Dn:", loss_discrim_natural.item(),"Dc:", loss_discrim_comics.item())

            
        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'comics2natural_state_dict': comics2natural.state_dict(),
                'natural2comics_state_dict': natural2comics.state_dict(),
                'discrim_natural_state_dict': discrim_natural.state_dict(),
                'discrim_comics_state_dict': discrim_comics.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_discrim_natural_state_dict': optimizer_discrim_natural.state_dict(),
                'optimizer_discrim_comics_state_dict': optimizer_discrim_comics.state_dict(),
                }
        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1

    
    
    
def train_add_text():

    import torch
    import itertools
    import numpy as np
    from torch.utils.data import DataLoader
    import os


    unique_name = "cyclegan_natural_comics_add_text"
    saving_folder = "models/trained"+unique_name
    os.makedirs(saving_folder, exist_ok=True)
    
    from utils.custom_dataset import ImageDataset_natural_comicsWithBalloons
    from models.models import GeneratorResNet, Discriminator
    


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")

    
    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = GeneratorResNet(input_shape=(3, 384, 384), out_channels=3, num_residual_blocks=8)
    generator_comicsB = GeneratorResNet(input_shape=(7, 384, 384), out_channels=3, num_residual_blocks=8)
    discrim_comicsB = Discriminator(input_shape = (3, 384, 384))



    # Losses 
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    criterion_GAN = torch.nn.MSELoss()
    criterion_balloons = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    natural2comicsNoB = natural2comicsNoB.to(device)
    generator_comicsB = generator_comicsB.to(device)
    discrim_comicsB = discrim_comicsB.to(device)
    criterion_GAN = criterion_GAN.to(device)
    criterion_balloons = criterion_balloons.to(device)
    criterion_identity = criterion_identity.to(device)



    # Optimizers
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    optimizer_G = torch.optim.Adam(generator_comicsB.parameters())
    optimizer_discrim_comicsB = torch.optim.Adam(discrim_comicsB.parameters())



    batch_size=16 if cuda else 1

    dataloader = DataLoader(
        ImageDataset_natural_comicsWithBalloons(unaligned = True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8 if cuda else 0
    )



    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_without_text", "99.pth")
    checkpoint = torch.load(checkpoint_file, map_location = device)
    natural2comicsNoB.load_state_dict(checkpoint['natural2comics_state_dict'])
    natural2comicsNoB = natural2comicsNoB.to(device)
    natural2comicsNoB.eval()



    epoch = 0

    checkpoint_file = os.path.join(saving_folder, "latest.pth")
    if os.path.exists(checkpoint_file):
        print("FOUND CHECKPOINT")
        checkpoint = torch.load(checkpoint_file)
        generator_comicsB.load_state_dict(checkpoint['generator_comicsB_state_dict'])
        discrim_comicsB.load_state_dict(checkpoint['discrim_comicsB_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_discrim_comicsB.load_state_dict(checkpoint['optimizer_discrim_comicsB_state_dict'])           
        epoch = checkpoint['epoch']
        print("LOADED CHECKPOINT EPOCH", epoch+1)
        epoch += 1
        print("RESUMING AT EPOCH", epoch+1)

    # Training
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

    while epoch+1 <= 100:

        for i, batch in enumerate(dataloader):

            # Set model input
            real_natural = batch["natural"].to(device)
            real_comicsB = batch["comics"].to(device)
            real_balloons = batch["balloons"].to(device).unsqueeze(1)

            natural2comicsNoB.eval()
            real_comicsNoB = natural2comicsNoB(real_natural).detach()


            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_natural.size(0), *discrim_comicsB.output_shape))).to(device)
            fake = torch.Tensor(np.zeros((real_natural.size(0), *discrim_comicsB.output_shape))).to(device)

            # ------------------
            #  Train Generator
            # ------------------
            generator_comicsB.train()
            optimizer_G.zero_grad()


            # Identity loss
            withItself = torch.cat((real_comicsB, real_balloons, real_comicsB), dim=1)
            loss_identity = criterion_identity(generator_comicsB(withItself), real_comicsB)

            # GAN loss
            inp = torch.cat((real_comicsB, real_balloons, real_comicsNoB), dim=1)
            fake_comicsB = generator_comicsB(inp)

            loss_GAN = criterion_GAN(discrim_comicsB(fake_comicsB), valid)

            # No change where no balloons loss
            loss_noB = criterion_balloons(real_comicsNoB*(1-real_balloons), fake_comicsB*(1-real_balloons))
            # No change where balloons loss
            loss_B = criterion_balloons(real_comicsB*real_balloons, fake_comicsB*real_balloons)

            # Total loss
            loss_G = loss_GAN + 3 * loss_noB + 3 * loss_B  + 5 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator
            # -----------------------
            discrim_comicsB.train()
            optimizer_discrim_comicsB.zero_grad()

            # Real loss
            loss_real = criterion_GAN(discrim_comicsB(real_comicsB), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(discrim_comicsB(fake_comicsB.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_discrim_comicsB.step()



            # ----------

            print("epoch:", epoch+1, "i:", i+1,
                  "GAN:", loss_GAN.item(), "noB:", loss_noB.item(), "B:", loss_B.item(), "identity:", loss_identity.item(),
                  "D:", loss_D.item())

        # END OF EPOCH https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("epoch:", epoch+1, "saving")
        to_save = {
                'epoch': epoch,
                'generator_comicsB_state_dict': generator_comicsB.state_dict(),
                'discrim_comicsB_state_dict': discrim_comicsB.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_discrim_comicsB_state_dict': optimizer_discrim_comicsB.state_dict(),
                }

        torch.save(to_save, os.path.join(saving_folder, str(epoch)+".pth"))    
        torch.save(to_save, checkpoint_file)
        print("epoch:", epoch+1, "saved")
        epoch += 1


def generate_depth_images_simple():
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os
    

    unique_name = "approach1"

    import models.models


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")


    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")



    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    midas = midas.to(device)


    from utils.custom_dataset import ImageDataset_comics

    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)


    #We want to evaluate epoch 100
    epoch = 100
    epoch -= 1

    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_simple", str(epoch)+".pth")
    assert os.path.exists(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])             
    epoch = checkpoint['epoch']
    print("LOADED CHECKPOINT EPOCH", epoch+1)

    comics2natural.eval()
    midas.eval()



    batch_size = 1

    dataloader = DataLoader(
        ImageDataset_comics(),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )



    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)

    for n, batch in enumerate(dataloader):
        #print(n)

        images = batch["comics"]

        images = images.to(device)

        with torch.no_grad():
            fake_natural = comics2natural(images)

            fake_natural -= mean
            fake_natural /= std
            prediction = midas(fake_natural)

        #print(images.data)
        #print(prediction.data)

        for i in range(batch["comics"].size()[0]):
            maxi = torch.max(prediction[i].view(-1))
            pred = prediction[i]/maxi
            pred = pred.unsqueeze(0).unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][1][i], batch["size"][0][i]),
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


    
    
def generate_depth_images_depth_aware():
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from PIL import Image
    import os


    unique_name = "apporoach1v1"

    import models.models
    from utils.custom_dataset import ImageDataset_comics


    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print (device)
    torch.hub.set_dir(".cache/torch/hub")


    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")



    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    midas = midas.to(device)


    # Initialize generators and discriminators
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = models.models.GeneratorResNet()
    # Moving to cuda if available
    # Inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
    comics2natural = comics2natural.to(device)


    #We want to evaluate epoch 100
    epoch = 100
    epoch -= 1

    checkpoint_file = os.path.join("models/trained/cyclegan_natural_comics_depth_aware", str(epoch)+".pth")
    assert os.path.exists(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    comics2natural.load_state_dict(checkpoint['comics2natural_state_dict'])             
    epoch = checkpoint['epoch']
    print("LOADED CHECKPOINT EPOCH", epoch+1)

    comics2natural.eval()
    midas.eval()


    batch_size = 1

    dataloader = DataLoader(
        ImageDataset_comics(),
        batch_size=batch_size,
        #shuffle=True,
        num_workers= 8 if cuda else 0
    )


    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).view(3,1,1).to(device)
    std = torch.Tensor(std).view(3,1,1).to(device)

    for n, batch in enumerate(dataloader):
        #print(n)

        images = batch["comics"]

        images = images.to(device)

        with torch.no_grad():
            fake_natural = comics2natural(images)

            fake_natural -= mean
            fake_natural /= std
            prediction = midas(fake_natural)

        #print(images.data)
        #print(prediction.data)

        for i in range(batch["comics"].size()[0]):
            maxi = torch.max(prediction[i].view(-1))
            pred = prediction[i]/maxi
            pred = pred.unsqueeze(0).unsqueeze(0)

            # Resize to original resolution
            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(batch["size"][1][i], batch["size"][0][i]),
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



    
    
    
    
    
def train(model):
    if model == "simple":
        train_simple()
    elif model == "depth_aware":
        train_depth_aware()
    elif model == "without_text":
        train_without_text()
    elif model == "add_text":
        train_add_text()     
    else:
        raise NotImplementedError
         
def generate_depth_images(model):
    if model == "simple":
        generate_depth_images_simple()
    elif model == "depth_aware":
        generate_depth_images_depth_aware()
    else:
        raise NotImplementedError
        



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        type=str,
                        help="which model to use (simple, depth_aware, without_text, add_text)")
    
    group = parser.add_mutually_exclusive_group()    
    group.add_argument(
        "--train",
        help="training the model",
        action="store_true"
    )
    group.add_argument(
        "--generate_depth_images",
        help="generating depth images (comics2natural2depth)",
        action="store_true"
    )
    
    args = parser.parse_args()

    if args.train:
        train(args.model)
    if args.generate_depth_images:
        generate_depth_images(args.model)