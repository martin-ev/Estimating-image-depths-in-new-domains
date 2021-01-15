Estimating image depths in new domains

This is the source code for the project on estimating image depths in new domains.

Trained models and hand-made depth ordering for evaluation are downloadable in this [Drive](https://drive.google.com/drive/folders/1R9xZcKAP5yae5w0Xr1AfGXkeRqWTf7Za?usp=sharing) (requires EFPL account), in the folder *models/trained* and *data/dcm_cropped/depth* (*data/Images Experiment cropped depth*).


# Table of Contents

- [Table of Contents](#table-of-contents)
- [A. Setting the datasets](#a-setting-the-datasets)
  - [A.I. Getting the datasets](#ai-getting-the-datasets)
    - [A.I.a. Comics domain](#aia-comics-domain)
      - [A.I.a.1. DCM772 dataset](#aia1-dcm772-dataset)
      - [A.I.a.2. eBDtheque dataset](#aia2-ebdtheque-dataset)
    - [A.I.b. Natural domain](#aib-natural-domain)
      - [A.I.b.1. COCO 2017 dataset](#aib1-coco-2017-dataset)
  - [A.II. Preparing the datasets](#aii-preparing-the-datasets)
    - [A.II.a. Generating cropped images from the DCM772 dataset](#aiia-generating-cropped-images-from-the-dcm772-dataset)
    - [A.II.b. Generating comics text areas mask from the eBDtheque dataset](#aiib-generating-comics-text-areas-mask-from-the-ebdtheque-dataset)
    - [A.II.c. Generating depth ordering for evaluation](#aiic-generating-depth-ordering-for-evaluation)
    - [A.II.d. Generating "ground-truth" depth of natural images](#aiid-generating-ground-truth-depth-of-natural-images)
- [B. About the models](#b-about-the-models)
  - [B.I. Existing depth estimator on natural image](#bi-existing-depth-estimator-on-natural-image)
  - [B.II. CycleGAN between comics and natural domains](#bii-cyclegan-between-comics-and-natural-domains)
  - [B.III. Use trained CycleGAN to do comics2natural2depth](#biii-use-trained-cyclegan-to-do-comics2natural2depth)
  - [B.IV. Use trained CycleGAN to do natural2depth, natural2comics -> train comics2depth](#biv-use-trained-cyclegan-to-do-natural2depth-natural2comics---train-comics2depth)
 - [C. Training the models](#c-training-the-models)
   - [C.I. Comics text area detector](#ci-comics-text-area-detector)
     - [C.I.1. Train U-net](#ci1-train-u-net)
       - [C.I.1.a. Train](#ci1a-train)
       - [C.I.1.b. Select epoch with highest validation IoU at threshold 0.5](#ci1b-select-epoch-with-highest-validation-iou-at-threshold-05)
       - [C.I.1.c. Optimize the threshold](#ci1c-optimize-the-threshold)
       - [C.I.1.d. Visualize the results](#ci1d-visualize-the-results)
     - [C.I.2. Generate comics without text areas](#ci2-generate-comics-without-text-areas)
       - [C.I.2.a. With the trained U-net](#ci2a-with-the-trained-u-net)
       - [C.I.2.b. By hand](#ci2b-by-hand)
   - [C.II. CycleGANs between natural and comics images](#cii-cyclegans-between-natural-and-comics-images)
     - [C.II.a. CycleGAN between natural and comics images](#ciia-cyclegan-between-natural-and-comics-images)
     - [C.II.b. "Depth-aware" CycleGAN between natural and comics images](#ciib-depth-aware-cyclegan-between-natural-and-comics-images)
     - [C.II.c. CycleGAN between natural and comics images without text](#ciic-cyclegan-between-natural-and-comics-images-without-text)
   - [C.III. Adding text areas in comics with GAN](#ciii-adding-text-areas-in-comics-with-gan)
   - [C.IV. Depth estimator](#civ-depth-estimator)
     - [C.IV.a. No more training with trained cycleGAN from comics to natural images](#civa-no-more-training-with-trained-cyclegan-from-comics-to-natural-images)
     - [C.IV.b. No more training with trained  "Depth-aware" cycleGAN from comics to natural images](#civb-no-more-training-with-trained--depth-aware-cyclegan-from-comics-to-natural-images)
     - [C.IV.c. Training with trained cycleGAN from natural to comics images](#civc-training-with-trained-cyclegan-from-natural-to-comics-images)
       - [C.IV.c.1. Optimizing lr](#civc1-optimizing-lr)
       - [C.IV.c.2. Training](#civc2-training)
     - [C.IV.d. Training with trained cycleGAN/GAN from natural to comics without text to comics with text images](#civd-training-with-trained-cyclegangan-from-natural-to-comics-without-text-to-comics-with-text-images)
       - [C.IV.d.1. Optimizing lr](#civd1-optimizing-lr)
         - [C.IV.d.1.i. Without ignore text loss](#civd1i-without-ignore-text-loss)
       - [C.IV.d.2. Training](#civd2-training)
         - [C.IV.d.2.i. Without ignore text loss](#civd2i-without-ignore-text-loss)
         - [C.IV.d.2.ii. With ignore text loss](#civd2ii-with-ignore-text-loss)
     - [C.IV.e. The baseline: batchnorm trick on existing depth estimator](#cive-the-baseline-batchnorm-trick-on-existing-depth-estimator)
- [D. Evaluating the models](#d-evaluating-the-models)
  - [D.1. Generate depth images](#d1-generate-depth-images)
  - [D.2. Scoring the results](#d2-scoring-the-results)
- [E. References](#e-references)

<!-- toc -->
# A. Setting the datasets

## A.I. Getting the datasets

### A.I.a. Comics domain

#### A.I.a.1. DCM772 dataset

Follow the instructions at [https://git.univ-lr.fr/crigau02/dcm_dataset/tree/master](https://git.univ-lr.fr/crigau02/dcm_dataset/tree/master) to get the images and annotations and place the downloaded folder *dcm_dataset.git* to *REPO_NAME/data*.

##### A.I.a.2. eBDtheque dataset

Follow the instructions at [http://ebdtheque.univ-lr.fr/registration/](http://ebdtheque.univ-lr.fr/registration/) to get the images and annotations and place the downloaded and unzipped folder *eBDtheque_database_v3* to *REPO_NAME/data*.

#### A.I.b. Natural domain

##### A.I.b.1. COCO 2017 dataset

Download the *2017 Val images [5K/1GB]* folder at [https://cocodataset.org/#download](https://cocodataset.org/#download), unzip it, rename it to *coco_val2017* and place it to *REPO_NAME/data*.

## A.II. Preparing the datasets

### A.II.a. Generating cropped images from the DCM772 dataset

In the DCM dataset, one image corresponds to one page. In order to train our models, we want to have one image for each frame. To generate those cropped images in the *REPO_NAME/data/dcm_cropped* folder, use:
```
python datasets_preparation.py --crop_frames
```

### A.II.b. Generating comics text areas mask from the eBDtheque dataset

Perform the same operation on the eBDtheque dataset with the following command to obtain the *REPO_NAME/data/eBDtheque_cropped* folder. It also randomly split the obtained set of signle frane image into train, validation and test sets.
```
python datasets_preparation.py --generate_text_masks
```


### A.II.c. Generating depth ordering for evaluation

In order to evaluate and compare the models, we generated by hand some ground-truth depth ordering. Those ground-truth depth ordering must be place under *REPO_NAME/data/dcm_cropped/depth*. To generate them, use:
```
python datasets_preparation.py --generate_evaluation
```

### A.II.d. Generating "ground-truth" depth of natural images

In order to train our models, we want to have the depth of natural images, which we compute using [MiDaS](https://github.com/intel-isl/MiDaS). We do this once now instead of doing it at run-time. Those ground-truth depth maps of natural images must be place under *REPO_NAME/data/coco_val2017_depth*. Generate them with:
```
python datasets_preparation.py --coco17_depth
```

# B. About the models
# B.I. Existing depth estimator on natural image
We use the work from [MiDaS](https://github.com/intel-isl/MiDaS) both as existing depth estimation on natural images and as a baseline for depth estimation on comics images (after a small "batch-norm trick" to adapt it to comics domain).
# B.II. CycleGAN between comics and natural domains
In both of proposed approaches, we train a CycleGAN to translate natural images to comics images and comics images to natural images. Then we used the trained CycleGAN in 2 different ways to get the depth of a comics image.
# B.III. Use trained CycleGAN to do comics2natural2depth
The idea in the first approach is to translate (with a CycleGAN) a given real comics image into a generated natural image and then to use (without any supplementary training) the existing depth estimation on natural images on this generated natural image. The prediction of the existing depth estimation on natural images is considered for this model as the depth prediction for the real comics image.

In a "depth-aware" variant, we give more information to the discriminator on natural images by giving it also the depth obtained with the existing depth estimation on natural images. Indeed, we aim to have a realistic depth and not so much a realistic generated natural image.

# B.IV. Use trained CycleGAN to do natural2depth, natural2comics -> train comics2depth
The idea in this second approach is to use the trained CycleGAN not to translate a given real comics image into a generated natural image but rather to translate a given real natural image into a generated comics image. This is done so that we obtain a "(comics image, ground-truth depth)" pair using the generated comics image and [MiDaS](https://github.com/intel-isl/MiDaS) (to estimate the depth of the real natural image and consider it as pseudo-ground-truth depth for the generated comics image). We then train a depth estimation model on the comics domain using the generated comics image as input and a reconstruction loss between the prediction and the pseudo-ground-truth depth.

The main idea of the 2 variants of this second approach is to control the position of the text areas in the generated comics. For this, we trained a U-net for text area detection (using the eBDtheque dataset) and generated a "comics without text" dataset (from the DCM772 dataset). A trained CycleGAN between the natural and the "comics without text" domains is used to generate a "(fake comics image without text, pseudo-ground-truth depth)" pair. We then add text areas from a real comics image with a GAN approach and finally train the comics-image-depth-estimation using the generated comics image with added text as input and a reconstruction loss between the prediction and the pseudo-ground-truth depth.

  
# C. Training the models

## C.I. Comics text area detector

### C.I.1. Train U-net

### C.I.1.a. Train

Using the eBDtheque dataset and its annotations, we trained for 450 epochs a U-net model for text areas detection in comics (). We trained this U-net with a loss between the text areas prediction (output of the model) and the ground-truth text mask:

```
python comics2textareas.py --train
```

### C.I.1.b. Select epoch with highest validation IoU at threshold 0.5
We kept epoch 427 that showed better validation IoU (Intersection over Union score) at threshold 0.5:
```
python comics2textareas.py --select_epoch
```

### C.I.1.c. Optimize the threshold
For the selected epoch, we optimized the threshold:
```
python comics2textareas.py --optimize_threshold
```

### C.I.1.d. Visualize the results 
We generate the estimated text areas for the validation set of eBDtheque and for the whole (single-frame images extracted from) DCM772 dataset:
```
python comics2textareas.py --visualize_ebdtheque_val
python comics2textareas.py --visualize_dcm
```


### C.I.2. Generate comics without text areas

#### C.I.2.a. With the trained U-net
To generate the "comics without text" dataset, we randomly cropped to 384×384 the single-frame images extracted from DCM772, along with their text areas prediction from the trained U-net. If the maximum value in the cropped text areas prediction exceed some value (3%), we continue until it is not the case anymore, decreasing by 1 the cropping size (384×384, 383×383, 382×382, ...) every 20 unsuccessful trials:
```
python comics2textareas.py --generate_dcm_without_text_areas
```

#### C.I.2.b. By hand

Generated "comics without text" images are then checked by hand and the ones that still contain text areas are manually deleted.


## C.II. CycleGANs between natural and comics images

For the purpose explained previously, we trained 3 different CycleGANs ("simple", "depth-aware" and "without text") to translate natural images to comics images and comics images to natural images.

### C.II.a. CycleGAN between natural and comics images
```
python cyclegan_natural2comics.py simple --train
```

### C.II.b. "Depth-aware" CycleGAN between natural and comics images
```
python cyclegan_natural2comics.py depth_aware --train
```

### C.II.c. CycleGAN between natural and comics images without text
```
python cyclegan_natural2comics.py without_text --train
```


## C.III. Adding text areas in comics with GAN
The GAN technique used in approach 2 variants (to add text in the generated comics images without text) has been trained on top of the trained "without text" CycleGAN:
```
python cyclegan_natural2comics.py add_text --train
```



## C.IV. Depth estimator

### C.IV.a. No more training with trained CycleGAN from comics to natural images
In the first approach, we only need to train the CycleGAN part. Then, for inference, we use the trained "Comics To Natural" sub-model and the existing depth estimation network on natural images [MiDaS](https://github.com/intel-isl/MiDaS). There is no need to train a depth estimation model for this approach.

### C.IV.b. No more training with trained  "Depth-aware" CycleGAN from comics to natural images
There is also no need to train a depth estimation model for this variant of the first approach for the same reasons.

### C.IV.c. Training with trained cycleGAN from natural to comics images
#### C.IV.c.1. Optimizing lr
The learning rate to train the depth estimation model for approach 2 is optimized with the following command:
```
python depth_estimator.py simple --optimize_lr
```
#### C.IV.c.2. Training
And then the depth estimation model is really trained:
```
python depth_estimator.py simple --train
```

### C.IV.d. Training with trained cycleGAN/GAN from natural to comics without text to comics with text images
#### C.IV.d.1. Optimizing lr
##### C.IV.d.1.i. Without ignore text loss
The learning rate to train the depth estimation model for approach 2 variant 1 is optimized with the following command:
```
python depth_estimator.py add_text --optimize_lr
```


#### C.IV.d.2. Training
And then the depth estimation models for approach 2 variant 1 and variant 2 are trained:
##### C.IV.d.2.i. Without ignore text loss
```
python depth_estimator.py add_text --train
```
##### C.IV.d.2.ii. With ignore text loss
```
python depth_estimator.py add_text_ignoretext --train
```

### C.IV.e. The baseline: batchnorm trick on existing depth estimator
The model we used from [MiDaS](https://github.com/intel-isl/MiDaS) contains batch normalization layers which performs differently in the training and inference mode. This can be explained by the fact the statistics learned by the batch normalization layers during training on natural images do not correspond at all to the statistics of the comics domain.
Therefore, a simple "batch-norm trick" to get better results was to change the momentum of the batch normalization layers to a very small value (0.002) and do a complete epoch on the comics domain (without any loss, only so that the batch normalization layers learn the statistics of the comics domain) and then switch to the inference mode for real inference:
```
python baseline.py batchnorm_trick --train
```



# D. Evaluating the models

## D.1. Generate depth images

For each model, we computed the predicted depth map of all images in the evaluation set of the DCM772 dataset. This is done in the inference mode after an "upper-bound" resizing. Predictions are then re-resized to the original image size.

For the first approach (comics2natural2depth) and its "depth-aware" variant:
```
python cyclegan_natural2comics.py simple --generate_depth_images
python cyclegan_natural2comics.py depth_aware --generate_depth_images
```

For the first approach (natural2depth, natural2comics --> trained comics2depth) and its text-handling variants:
```
python depth_estimator.py simple --generate_depth_images
python depth_estimator.py add_text --generate_depth_images
python depth_estimator.py add_text_ignoretext --generate_depth_images
```

For the [MiDaS](https://github.com/intel-isl/MiDaS) model and its improvement with the "batch-norm trick":
```
python baseline.py no_batchnorm_trick --generate_depth_images
python baseline.py batchnorm_trick --generate_depth_images
```


## D.2. Scoring the results
We then compute for each of the predicted depth map an "inter-objects accuracy" and an "intra-object accuracy". 

"Inter-objects accuracy"  corresponds to the ratio of pairs of points (in the hand-made depth-ordering) with different object numbers that are correctly ordered in the depth prediction (smallest object number of the pair should be assigned a closer depth). Taking the mean of all inter-objects accuracies give an inter-objects accuracy score for the model.

"Intra-object accuracy" corresponds to the ratio of pairs of points (in the hand-made depth-ordering) with same object number and different intra-object numbers that are correctly ordered in the depth prediction (smallest intra-object number should be assigned a closer depth). Taking the mean of all intra-object accuracies give an intra-object accuracy score for the method.

The following commands are used to compute those 2 scores, which are in addition normalized with regard to the baseline and put in the log scale:

```
python evaluation.py natural2comics simple
python evaluation.py natural2comics depth_aware
python evaluation.py depth_estimator simple
python evaluation.py depth_estimator add_text
python evaluation.py depth_estimator add_text_ignoreloss
python evaluation.py baseline no_batchnorm_trick
python evaluation.py baseline batchnorm_trick
```

# E. References 
The following works were our references for this project:

- Kiyoharu Aizawa, Azuma Fujimoto, Atsushi Otsubo, Toru Ogawa, Yusuke Matsui, Koki Tsubota, and Hikaru Ikuta. Building a manga dataset “manga109” with annotations for multimedia applications. IEEE MultiMedia, 27(2):8–18, 2020.
- Karteek Alahari, Guillaume Seguin, Josef Sivic, and Ivan Laptev. Pose estimation and segmentation of people in 3d movies. In Proceedings of the IEEE International Conference on Computer Vision, pages 2112–2119, 2013.
- Ibraheem Alhashim and Peter Wonka. High quality monocular depth estimation via transfer learning. arXiv preprint arXiv:1812.11941, 2018.
- Amir Atapour-Abarghouei and Toby P Breckon. Real-time monocular depth estimation using synthetic datawith domain adaptation via image style transfer. In Proceedings of the IEEE Conference on Computer Visionand Pattern Recognition, pages 2800–2810, 2018.
- Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013.
- Andreas Geiger, Philip Lenz, and Raquel Urtasun.  Are we ready for autonomous driving?  the kitti visionbenchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
- Clément Guérin, Christophe Rigaud, Antoine Mercier, Farid Ammar-Boudjelal, Karell Bertet, Alain Bouju, Jean-Christophe Burie, Georges Louis, Jean-Marc Ogier, and Arnaud Revel. ebdtheque: a representative database of comics. In Proceedings of the 12th International Conference on Document Analysis and Recognition (ICDAR), 2013.
- Heiko Hirschmuller. Accurate and efficient stereo processing by semi-global matching and mutual information. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 2, pages 807–814. IEEE, 2005.
- Naoto Inoue, Ryosuke Furuta, Toshihiko Yamasaki, and Kiyoharu Aizawa. Cross-domain weakly-supervised object detection through progressive domain adaptation. In Proceedings of the IEEE conference on computervision and pattern recognition, pages 5001–5009, 2018.
- Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.
- Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.
- Hsin-Ying Lee, Hung-Yu Tseng, Jia-Bin Huang, Maneesh Singh, and Ming-Hsuan Yang. Diverse image-to-image translation via disentangled representations. In Proceedings of the European conference on computervision (ECCV), pages 35–51, 2018.
- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.  Microsoft coco:  Common objects in context.  In European conference on computervision, pages 740–755. Springer, 2014.
- Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf.  Consistent video depth estimation. arXiv preprint arXiv:2004.15021, 2020.
- Yusuke Matsui, Kota Ito, Yuji Aramaki, Azuma Fujimoto, Toru Ogawa, Toshihiko Yamasaki, and Kiyo-haru Aizawa.  Sketch-based manga retrieval using manga109 dataset. Multimedia Tools and Applications, 76(20):21811–21838, 2017.
- Youssef Alami Mejjati, Christian Richardt, James Tompkin, Darren Cosker, and Kwang In Kim. Unsupervised attention-guided image-to-image translation. In Advances in Neural Information Processing Systems, pages 3693–3703, 2018.
- Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784,2014.
- Nhu-Van Nguyen, Christophe Rigaud, and Jean-Christophe Burie. Digital comics image indexing based on deep learning. Journal of Imaging, 4(7), 2018.72
- Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d ken burns effect from a single image. ACM Transactions on Graphics (TOG), 38(6):1–15, 2019.
- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems, pages 8026–8037, 2019.
- Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, and Nicu Sebe. Unsupervised adversarial depth estimation using cycled generative networks.  In 2018 International Conference on 3D Vision (3DV), pages 587–595.IEEE, 2018.
- René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysisand Machine Intelligence, 2020.
- Anita Rau, PJ Eddie Edwards, Omer F Ahmad, Paul Riordan, Mirek Janatka, Laurence B Lovat, and Danail Stoyanov. Implicit domain adaptation with conditional generative adversarial networks for depth prediction inendoscopy. International journal of computer assisted radiology and surgery, 14(7):1167–1176, 2019.
- Lawrence G Roberts. Machine perception of three-dimensional solids. PhD thesis, Massachusetts Institute of Technology, 1963.
- Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical imagesegmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
- Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.
- Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmarksuite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015.
- Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5505–5514, 2018.
- Amir R Zamir, Alexander Sax,  William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer visionand pattern recognition, pages 3712–3722, 2018.
- Ziyu Zhang, Alexander G Schwing, Sanja Fidler, and Raquel Urtasun. Monocular object instance segmentationand depth ordering with cnns. In Proceedings of the IEEE International Conference on Computer Vision,pages 2614–2622, 2015.
- Chaoqiang Zhao, Qiyu Sun, Chongzhen Zhang, Yang Tang, and Feng Qian. Monocular depth estimationbased on deep learning: An overview. Science China Technological Sciences, pages 1–16, 2020.
- Shanshan Zhao, Huan Fu, Mingming Gong, and Dacheng Tao. Geometry-aware symmetric domain adaptationfor monocular depth estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9788–9798, 2019.
- Tinghui Zhou, Matthew Brown, Noah Snavely, and David G Lowe.  Unsupervised learning of depth andego-motion from video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1851–1858, 2017.
- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros.  Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computervision, pages 2223–2232, 2017.
