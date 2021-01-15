#!/usr/bin/env python
# coding: utf-8

"""
custom_dataset.py: Code for the custom PyTorch Dataset
"""

import numpy as np
import cv2
__author__ = "Martin Nicolas Everaert"
__date__ = "December 2020"

from typing import List, Tuple

# Partially based on https://github.com/eriklindernoren/PyTorch-GAN/blob/80e7702c25266925774d020e047fdff8d44f7a74/implementations/cyclegan/datasets.py
# Partially based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py
# Partially based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/utils.py
# Partially based on https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
# Partially based on https://pytorch.org/hub/intelisl_midas_v2/

from utils.data_structure import *
import torch
import warnings
import glob
import random
import os
import torch.utils.data
import PIL
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import albumentations as A

      
def to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """Converts the image to RGB mode.

    Args:
        image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Output image in RGB mode
    """
    rgb_image = PIL.Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def to_l(image: PIL.Image.Image) -> PIL.Image.Image:
    """Converts the image to L mode.

    Args:
        image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Output image in L mode
    """
    l_image = PIL.Image.new("L", image.size)
    l_image.paste(image)
    return l_image


# Based on https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
def get_random_crop(image, image2, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = 0 if max_x == 0 else np.random.randint(0, max_x)
    y = 0 if max_y == 0 else np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    if image2 is not None:
        crop2 = image2[y: y + crop_height, x: x + crop_width]
    else:
        crop2 = None
    return crop, crop2

# Based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py
class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

# Based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py
class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample
    
# Based on https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py
class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width: int,
        height: int,
        resize_target: bool = True,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resize_method: str = "lower_bound",
        image_interpolation_method: int = cv2.INTER_AREA,
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=1, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width,
                                      height), interpolation=cv2.INTER_NEAREST
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"].astype(bool)

        return sample


class CustomDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset to train our models
    """

    def __init__(self,
                 dcm: List[str] = ["train", "validation", "test"],
                 coco17: bool = True,
                 eBDtheque: bool = False,
                 eBDtheque_cropped: bool = False,
                 eBDtheque_cropped_mask: bool = False,
                 natural_depth: bool = True,
                 unaligned: bool = False,
                 resize: Tuple[int, int] = (384, 384),
                 interpolation: int = PIL.Image.BICUBIC,
                 resize_mode: str = "train",
                 max_len: int = 0,
                 ):
        """Init.

        Args:
            dcm (List[str], optional):
                Subsets of the DCM dataset to use.
                Defaults to ["train", "validation", "test"].
            coco17 (bool, optional):
                True: Use the COCO 2017 validation dataset.
                False: Do not use.
                Defaults to True.
            eBDtheque (bool, optional):
                True: Use the eBDtheque dataset.
                False: Do not use.
                Defaults to False.
            eBDtheque_cropped (bool, optional): 
                True: Use the "cropped" eBDtheque dataset.
                False: Do not use.
                Defaults to False.
            eBDtheque_cropped_mask (bool, optional):
                True: Use the "cropped" eBDtheque dataset with text masks.
                False: Do not use.
                Defaults to False.
            coco17_depth (bool, optional): 
                True: Use the COCO 2017 validation dataset with MiDas depth.
                False: Do not use.
                Defaults to True.
            unaligned (bool, optional):
                Whether to unalign the "natural" and the "comics" domains.
                True: Different indexes are used for both domains.
                False: Same index is used for both domains.
                Defaults to True.
            resize (Tuple[int, int], optional):
                Desired size.
                Defaults to (384, 384).
            interpolation (int, optional):
                Interpolation method.
                Defaults to PIL.Image.BICUBIC.
            resize_mode (str, optional): 
                "train": 'lower_bound' resize to at least the desired size,
                    and then random crop to exactely the desired size.
                "inference": 'upper_bound' resize to at most the desired size.
                Defaults to "train".
            max_len (int, optional): 
                0: No restriction on the length of the Dataset.
                max_len>=1: Restricts the length of the Dataset to max_len.
                Defaults to 0.
        """
        # Checking the given parameter
        use_dcm = len(dcm) > 0
        use_eBDtheque = eBDtheque or eBDtheque_cropped or eBDtheque_cropped_mask
        if use_dcm and use_eBDtheque:
            raise Exception("Please use only one comics dataset.")
        if eBDtheque_cropped_mask and not eBDtheque_cropped:
            warnings.warn(
                "eBDtheque_cropped is False but eBDtheque_cropped_mask is True.")
        if eBDtheque and (eBDtheque_cropped or eBDtheque_cropped_mask):
            warnings.warn(
                "eBDtheque (not cropped) and (eBDtheque_cropped or eBDtheque_cropped_mask) are both True.")
        if resize != (384, 384):
            warnings.warn("resize != (384, 384).")
        if resize_mode not in ["train", "inference"]:
            raise Exception("resize_mode not in [\"train\", \"inference\"].")
        if max_len >= 1:
            warnings.warn(
                "Length of the Dataset is restricted to "+str(max_len)+".")

        self.list_comics = []
        for subset in dcm:
            self.list_comics += [DCM_IMAGE_PATH_FROM_NAME(
                x) for x in DCM_GET_FILES_LIST(DCM_FILENAMES[subset])]

        self.list_natural = []
        if coco17:
            self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
            self.list_natural = [x for x in self.list_natural if x[-3:]!=".md"]

        self.resize_mode = resize_mode

        resize_method = "lower_bound" if resize_mode == "train" else "upper_bound"
        self.resize = Resize(
            resize[0],
            resize[1],
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_method,
            image_interpolation_method=interpolation
        )

        self.unaligned = unaligned
        
        self.prepare = PrepareForNet()
        
        self.max_len = max_len


    def __getitem__(self, index):
        index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
        
        
        #https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/utils.py#L98
               
        img_coco17 = cv2.imread(self.list_natural[index % len(self.list_natural)])
        if img_coco17.ndim == 2:
            img_coco17 = cv2.cvtColor(img_coco17, cv2.COLOR_GRAY2BGR)
        img_coco17 = cv2.cvtColor(img_coco17, cv2.COLOR_BGR2RGB) / 255.0
            
        size = img_coco17.shape
        
        img_coco17 = self.resize({"image": img_coco17})["image"]
        img_coco17 = self.prepare({"image": img_coco17})["image"]
        
        return {"img_coco17":img_coco17,
                "name_coco17":self.list_natural[index % len(self.list_natural)],
                "size_coco17":size
               }    
        
    def __len__(self):
        length = max(len(self.list_comics), len(self.list_natural))
        if self.max_len>=1:
            length = min(length, self.max_len)
        return length
        
class ComicsDatasetV2(Dataset):
    
    def __init__(self, train=True, val=True, test=True):
        self.list_comics = []
        if train: 
            with open(os.path.join("data/dcm_cropped", "train.txt")) as file:
                content = file.read().split("\n")
                self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
        if val: 
            with open(os.path.join("data/dcm_cropped", "validation.txt")) as file:
                content = file.read().split("\n")
                self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
        if test: 
            with open(os.path.join("data/dcm_cropped", "test.txt")) as file:
                content = file.read().split("\n")
                self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
                
        self.transform = transforms.Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )#from https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/midas/transforms.py#L48

    def __getitem__(self, index):
        #https://github.com/intel-isl/MiDaS/blob/b00bf61f846d73fadc1f287293648db9f88d3615/utils.py#L98
        img = cv2.imread(self.list_comics[index % len(self.list_comics)])
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            
        size = image.shape
        item = self.transform({"image": image})["image"]
        
        return {"img":item,
                "name":self.list_comics[index % len(self.list_comics)],
                "size":size
               } 
    
    def __len__(self):
        return len(self.list_comics)  
    

class ComicsForBalloons(Dataset):

    def __init__(self, file="train.txt", augment=False):

        with open("data/eBDtheque_cropped/"+file) as f:
            self.file_list = f.readlines()
        self.file_list = [x[:-1] for x in self.file_list]

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()

        self.augment = augment
        if self.augment :
            random.seed(42)
            self.aug = A.Compose([  
                              A.ToGray(p=.1), 
                              A.RandomBrightnessContrast(p=.8),    
                              A.RandomGamma(p=.8),    
                              #A.CLAHE(p=.8),    
                              #A.JpegCompression(p=.5),   
                              A.HorizontalFlip(p=.5), 
                              #A.GridDistortion(p=.8), 
                              #A.ElasticTransform(p=.8) 
                          ], p=1)


    def __getitem__(self, index):

        file = "data/eBDtheque_cropped/" + self.file_list[index % len(self.file_list)]
        file_mask = "data/eBDtheque_cropped/" +self.file_list[index % len(self.file_list)].replace(".bmp", "_mask.bmp")

        image = cv2.imread(file)
        mask = cv2.imread(file_mask)

        if self.augment :
            augmented = self.aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.
        if mask.ndim != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  
        mask = mask.astype(bool)



        #print("before resize :", image.shape[1], image.shape[0])
        #print("before resize :", mask.shape[1], mask.shape[0])
        #print("before resize :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #resize
        if image.shape[0]<384 or image.shape[1]<384 :
            r = self.t0({"image": image, "mask":mask})
            image = r["image"]
            mask = r["mask"]

        #print("after resize :", image.shape[1], image.shape[0])
        #print("after resize :", mask.shape[1], mask.shape[0])
        #print("after resize :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #NormalizeImage
        image = self.t1({"image": image})["image"]


        #print("after NormalizeImage :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #Random crop to exactly 384*834     
        image, mask = get_random_crop(image, mask, 384, 384)

        #print("after crop :")
        #print (np.min(image[:, :, 0]), np.max(image[:, :, 0]))
        #print (np.min(image[:, :, 1]), np.max(image[:, :, 1]))
        #print (np.min(image[:, :, 2]), np.max(image[:, :, 2]))

        #PrepareForNet
        r = self.t2({"image": image, "mask": mask})
        image = r["image"]
        mask = r["mask"]

        #print("after prepare :")
        #print (np.min(image[0]), np.max(image[0]))
        #print (np.min(image[1]), np.max(image[1]))
        #print (np.min(image[2]), np.max(image[2]))


        return {"image": image, "mask": mask}

    def __len__(self):
        return len(self.file_list)
    
class ImageDataset_natural_comicsWithBalloons(Dataset):
    def __init__(self,
                 unaligned = False):

        self.list_comics = []
        with open(os.path.join("data/dcm_cropped", "train.txt")) as file:
            content = file.read().split("\n")
            self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
        self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
        self.unaligned = unaligned

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()
        self.tensorify = transforms.ToTensor()

    def __getitem__(self, index):
        index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
        natural_name = self.list_natural[index % len(self.list_natural)]
        comics_name = self.list_comics[index2 % len(self.list_comics)]     
        balloons_name = comics_name.replace("dcm_cropped/images", "dcm_cropped/balloons427").replace(".jpg", "_originalsize.png")      


        image_natural = cv2.imread(natural_name)
        image_comics = cv2.imread(comics_name)   
        image_balloons = cv2.imread(balloons_name)    


        if image_natural.ndim == 2:
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
        image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0 
        if image_balloons.ndim != 2:
            image_balloons = cv2.cvtColor(image_balloons, cv2.COLOR_BGR2GRAY)      
        image_balloons = image_balloons / 255.0


        #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
        #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
        #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


        #Resize to at least 384*834
        item_natural = self.t0({"image": image_natural})["image"]
        r = self.t0({"image": image_comics, "disparity":image_balloons})
        item_comics = r["image"]
        item_balloons = r["disparity"]


        #NormalizeImage
        #item_natural = self.t1({"image": item_natural})["image"]
        #item_comics = self.t1({"image": item_comics})["image"] 

        #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

        #Random crop to exactly 384*834     
        item_natural, _ = get_random_crop(item_natural, None, 384, 384)
        item_comics, item_balloons = get_random_crop(item_comics, item_balloons, 384, 384)

        #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

        #PrepareForNet
        item_natural = self.t2({"image": item_natural})["image"]
        r = self.t2({"image": item_comics, "disparity":item_balloons})
        item_comics = r["image"]
        item_balloons = r["disparity"]


        return {"natural": item_natural,
                "comics": item_comics,
                "balloons": item_balloons,
               }

    def __len__(self):
        return max(len(self.list_comics), len(self.list_natural))
        
class ImageDataset_comics(Dataset):

    def __init__(self):

        comics_file = "validation.txt"
        list_comics = []
        with open(os.path.join("data/dcm_cropped", comics_file)) as file:
            content = file.read().split("\n")
            list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]    
        self.list_comics = list_comics

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()
        self.tensorify = transforms.ToTensor()

    def __getitem__(self, index):
        comics_name = self.list_comics[index % len(self.list_comics)]

        image_comics = cv2.imread(comics_name)       


        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0   


        #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
        #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
        #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


        #Resize to at least 384*834
        x, y = image_comics.shape[1], image_comics.shape[0]
        item_comics = self.t0({"image": image_comics})["image"]


        #NormalizeImage
        #item_natural = self.t1({"image": item_natural})["image"]
        #item_comics = self.t1({"image": item_comics})["image"] 

        #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

        #Random crop to exactly 384*834     
        #item_comics, _ = get_random_crop(item_comics, None, 384, 384)

        #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

        #PrepareForNet
        item_comics = self.t2({"image": item_comics})["image"]


        return {
                "comics": item_comics,
                "size": (x,y),
                "name":comics_name
               }

    def __len__(self):
        return len(self.list_comics)
    
class ImageDataset_natural_comics(Dataset):

    def __init__(self,
                 unaligned = False):

        comics_file = "train.txt"
        list_comics = []
        with open(os.path.join("data/dcm_cropped", comics_file)) as file:
            content = file.read().split("\n")
            list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]    
        self.list_comics = list_comics
        self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
        self.unaligned = unaligned

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()
        self.tensorify = transforms.ToTensor()

    def __getitem__(self, index):
        index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
        natural_name = self.list_natural[index % len(self.list_natural)]
        comics_name = self.list_comics[index2 % len(self.list_comics)]

        image_natural = cv2.imread(natural_name)
        image_comics = cv2.imread(comics_name)       


        if image_natural.ndim == 2:
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
        image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0   


        #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
        #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
        #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


        #Resize to at least 384*834
        item_natural = self.t0({"image": image_natural})["image"]
        item_comics = self.t0({"image": image_comics})["image"]


        #NormalizeImage
        #item_natural = self.t1({"image": item_natural})["image"]
        #item_comics = self.t1({"image": item_comics})["image"] 

        #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

        #Random crop to exactly 384*834     
        item_natural, _ = get_random_crop(item_natural, None, 384, 384)
        item_comics, _ = get_random_crop(item_comics, None, 384, 384)

        #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

        #PrepareForNet
        item_natural = self.t2({"image": item_natural})["image"]
        item_comics = self.t2({"image": item_comics})["image"]


        return {"natural": item_natural,
                "comics": item_comics,
               }

    def __len__(self):
        return max(len(self.list_comics), len(self.list_natural))

class ImageDataset_natural_comicsnoballoons(Dataset):

    def __init__(self,
                 unaligned = False):

        self.list_comics = sorted(glob.glob("data/dcm_cropped_noballoons_human/*.*"))
        self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
        self.unaligned = unaligned

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()
        self.tensorify = transforms.ToTensor()

    def __getitem__(self, index):
        index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
        natural_name = self.list_natural[index % len(self.list_natural)]
        comics_name = self.list_comics[index2 % len(self.list_comics)]

        image_natural = cv2.imread(natural_name)
        image_comics = cv2.imread(comics_name)       


        if image_natural.ndim == 2:
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
        image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0   


        #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
        #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
        #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


        #Resize to at least 384*834
        item_natural = self.t0({"image": image_natural})["image"]
        item_comics = self.t0({"image": image_comics})["image"]


        #NormalizeImage
        #item_natural = self.t1({"image": item_natural})["image"]
        #item_comics = self.t1({"image": item_comics})["image"] 

        #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

        #Random crop to exactly 384*834     
        item_natural, _ = get_random_crop(item_natural, None, 384, 384)
        item_comics, _ = get_random_crop(item_comics, None, 384, 384)

        #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

        #PrepareForNet
        item_natural = self.t2({"image": item_natural})["image"]
        item_comics = self.t2({"image": item_comics})["image"]


        return {"natural": item_natural,
                "comics": item_comics,
               }

    def __len__(self):
        return max(len(self.list_comics), len(self.list_natural))

class ImageDataset_naturalWithDepth_comicsWithBalloons(Dataset):

    def __init__(self,
                 unaligned = False):

        self.list_comics = []
        with open(os.path.join("data/dcm_cropped", "train.txt")) as file:
            content = file.read().split("\n")
            self.list_comics += [os.path.join("data/dcm_cropped/images", x+'.jpg') for x in content if x != ""]
        self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
        self.unaligned = unaligned

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()
        self.tensorify = transforms.ToTensor()

    def __getitem__(self, index):
        index2 = random.randrange(0, len(self.list_comics)) if self.unaligned else index
        natural_name = self.list_natural[index % len(self.list_natural)]
        comics_name = self.list_comics[index2 % len(self.list_comics)]     
        balloons_name = comics_name.replace("dcm_cropped/images", "dcm_cropped/balloons427").replace(".jpg", "_originalsize.png")      

        depth_name = natural_name.replace("coco_val2017", "coco_val2017_depth").replace(".jpg", ".png")
        with open(depth_name.replace(".png", ".txt")) as file:
            scale = float(file.read())        

        image_natural = cv2.imread(natural_name)
        image_comics = cv2.imread(comics_name)   
        image_balloons = cv2.imread(balloons_name)    
        image_natural_depth = cv2.imread(depth_name)


        if image_natural.ndim == 2:
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
        image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
        if image_comics.ndim == 2:
            image_comics = cv2.cvtColor(image_comics, cv2.COLOR_GRAY2BGR)
        image_comics = cv2.cvtColor(image_comics, cv2.COLOR_BGR2RGB) / 255.0 
        if image_balloons.ndim != 2:
            image_balloons = cv2.cvtColor(image_balloons, cv2.COLOR_BGR2GRAY)      
        image_balloons = image_balloons / 255.0
        if image_natural_depth.ndim != 2:
            image_natural_depth = cv2.cvtColor(image_natural_depth, cv2.COLOR_BGR2GRAY)      
        image_natural_depth = image_natural_depth / 255.0


        #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
        #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
        #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


        #Resize to at least 384*834
        r = self.t0({"image": image_natural, "disparity":image_natural_depth})
        item_natural = r["image"]
        item_natural_depth = r["disparity"]

        r = self.t0({"image": image_comics, "disparity":image_balloons})
        item_comics = r["image"]
        item_balloons = r["disparity"]


        #NormalizeImage
        #item_natural = self.t1({"image": item_natural})["image"]
        #item_comics = self.t1({"image": item_comics})["image"] 

        #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

        #Random crop to exactly 384*834     
        item_natural, item_natural_depth = get_random_crop(item_natural, item_natural_depth, 384, 384)
        item_comics, item_balloons = get_random_crop(item_comics, item_balloons, 384, 384)

        #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

        #PrepareForNet
        r = self.t2({"image": item_natural, "disparity":item_natural_depth})
        item_natural = r["image"]
        item_natural_depth = r["disparity"]
        r = self.t2({"image": item_comics, "disparity":item_balloons})
        item_comics = r["image"]
        item_balloons = r["disparity"]

        item_natural_depth *= scale


        return {"natural": item_natural,
                "natural_depth": item_natural_depth,
                "scale": scale,
                "comics": item_comics,
                "balloons": item_balloons,
               }

    def __len__(self):
        return max(len(self.list_comics), len(self.list_natural))
    
class ImageDataset_naturalWithDepth(Dataset):

    def __init__(self,
                 unaligned = False):

        self.list_natural = sorted(glob.glob("data/coco_val2017/*.*"))
        self.unaligned = unaligned

        self.t0 = Resize(
                    384,
                    384,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
        #self.t1 = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.t2 = PrepareForNet()
        self.tensorify = transforms.ToTensor()

    def __getitem__(self, index):
        natural_name = self.list_natural[index % len(self.list_natural)]

        depth_name = natural_name.replace("coco_val2017", "coco_val2017_depth").replace(".jpg", ".png")
        with open(depth_name.replace(".png", ".txt")) as file:
            scale = float(file.read())        

        image_natural = cv2.imread(natural_name)
        image_natural_depth = cv2.imread(depth_name)


        if image_natural.ndim == 2:
            image_natural = cv2.cvtColor(image_natural, cv2.COLOR_GRAY2BGR)
        image_natural = cv2.cvtColor(image_natural, cv2.COLOR_BGR2RGB) / 255.0
        if image_natural_depth.ndim != 2:
            image_natural_depth = cv2.cvtColor(image_natural_depth, cv2.COLOR_BGR2GRAY)      
        image_natural_depth = image_natural_depth / 255.0


        #print("before resize :", natural_name, image_natural.shape[1], image_natural.shape[0])
        #if self.natural_depth: print("before resize :",depth_name, image_natural_depth.shape[1], image_natural_depth.shape[0])
        #print("before resize :",comics_name, image_comics.shape[1], image_comics.shape[0])


        #Resize to at least 384*834
        r = self.t0({"image": image_natural, "disparity":image_natural_depth})
        item_natural = r["image"]
        item_natural_depth = r["disparity"]


        #NormalizeImage
        #item_natural = self.t1({"image": item_natural})["image"]
        #item_comics = self.t1({"image": item_comics})["image"] 

        #print("after resize :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after resize :",depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after resize :",comics_name, item_comics.shape[1], item_comics.shape[0])

        #Random crop to exactly 384*834     
        item_natural, item_natural_depth = get_random_crop(item_natural, item_natural_depth, 384, 384)

        #print("after crop :", natural_name, item_natural.shape[1], item_natural.shape[0])
        #if self.natural_depth: print("after crop :", depth_name, item_natural_depth.shape[1], item_natural_depth.shape[0])
        #print("after crop :", comics_name, item_comics.shape[1], item_comics.shape[0])

        #PrepareForNet
        r = self.t2({"image": item_natural, "disparity":item_natural_depth})
        item_natural = r["image"]
        item_natural_depth = r["disparity"]

        item_natural_depth *= scale


        return {"natural": item_natural,
                "natural_depth": item_natural_depth,
               }

    def __len__(self):
        return len(self.list_natural)
