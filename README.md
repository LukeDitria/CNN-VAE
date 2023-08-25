
# CNN-VAE
A Res-Net Style VAE with an adjustable perceptual loss using a pre-trained vgg19. <br>
Based off of ![Deep Feature Consistent Variational Autoencoder](https://arxiv.org/pdf/1610.00291.pdf)
 <br>
<br>
<b> Latent space interpolation </b> <br>
![Latent space interpolation](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE.gif)
<br>
# NEW!
1.2) Added Dynamic Depth Architecture, define the "blocks" parameter, a list of channel scales. Each scale will create a new Res up/down block with each block scaling up/down by a factor of 2. 
Default parameters will downsample a 3x64x64 image to a 512x4x4 latent space although any square image will work.
<br>
<br>
1.1) Added training script with loss logging etc. Dataset uses Pytorch "ImageFolder" dataset, code assumes there is no pre-defined train/test split and creates
one if w fixed random seed so it will be the same every time the code is run.<br>
<br>
<b> Basic training command: </b>
```
python train_vae.py -mn test_run --dataset_root #path to dataset root#
```

<br>
<b> Starting from an existing checkpoint: </b><br>

```
python train_vae.py -mn test_run --load_checkpoint --dataset_root #path to dataset root#
```

<br>
<b> Train without a feature loss: </b><br>

```
python train_vae.py -mn test_run --feature_scale 0 --dataset_root #path to dataset root#
```

<br>
<b> Define a Custom Architecture: </b><br>

```
python train_vae.py -mn test_run --latent_channels 512 --block_widths 1 2 4 8 --ch_multi 64 --dataset_root #path to dataset root#
```

<br>

# Results

<br>
Results on validation images of the STL10 dataset at 64x64 with a latent vector size of 512 (images on top are the reconstruction)
NOTE: RES_VAE_64_old.py was used to generate the results below<br>

**With Perception loss**
<br>
![VAE Trained with perception/feature loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_STL10_64.png)


**Without Perception loss**
<br>
![VAE Trained without perception/feature loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_STL10_no_perception_64.png)

## Additional Results - celeba
The images in the STL10 have a lot of variation meaning more "features" need to be encoded in the latent space to achieve a good reconstruction. Using a data-set with less variation (and the same latent vector size) should results in a higher quality reconstructed image.

![Celeba trained with perception loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_celeba_64.png)

**New Model** Test images from VAE trained on CelebA at 128x128 resolution (latent space is therefore 512x2x2) using all layers of the VGG model for the perception loss
![Celeba 128x128 test images trained with perception loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_CelebA_all_Feat_new_model_128.png)


