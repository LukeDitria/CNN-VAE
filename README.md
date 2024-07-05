
# CNN-VAE
A Res-Net Style VAE with an adjustable perceptual loss using a pre-trained vgg19. <br>
Based off of ![Deep Feature Consistent Variational Autoencoder](https://arxiv.org/pdf/1610.00291.pdf)
 <br>
<br>
<b> Latent space interpolation </b> <br>
![Latent space interpolation](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE.gif)
<br>
<br>
# My Pytorch Deep Learning Series on Youtube
[Whole Playlist](https://youtube.com/playlist?list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa&si=EVHfovKS-vQ_VZ5a)<br>
[Pytorch VAE Basics](https://youtu.be/dDJv6DiuqEk)<br>
<br>
# If you found this code useful
[Buy me a Coffee](https://www.buymeacoffee.com/lukeditria)

# NEW!
**Let me know if any other features would be useful!**<br><br>
<b>1.3)</b> Default model is now much larger, but still has a similar memory usage plus much better performance. Added some additional arguments for greater customization!<br>
--norm_type arg to change the layer norm type between BatchNorm (bn) and GroupNorm (gn), use GroupNorm if you can only train with a small batch size.<br>
--num_res_blocks arg defines how many Res-Identity blocks are at the BottleNeck for both the Encoder and Decoder, increase this for a deeper model while maintaining low memory footprint.<br>
--deep_model arg will add a Res-Identity block to each of the up/down-sampling stages for both the Encoder and Decoder, use this to increase depth, but will result in a larger memory footprint + slower training.
<br>
<br>
<b>1.2)</b> Added Dynamic Depth Architecture, define the "blocks" parameter, a list of channel scales. Each scale will create a new Res up/down block with each block scaling up/down by a factor of 2. 
Default parameters will downsample a 3x64x64 image to a 256x4x4 latent space although any square image will work.
<br>
<br>
<b>1.1)</b> Added training script with loss logging etc. Dataset uses Pytorch "ImageFolder" dataset, code assumes there is no pre-defined train/test split and creates
one if w fixed random seed so it will be the same every time the code is run.<br>

# Training Examples
<b>Notes:</b><br>
Avoid using a Bottle-Neck feature map size of less than 4x4 as all conv kernels are 3x3, if you do set --num_res_blocks to 0 to avoid adding a lot of model parameters that won't do much <br>
If you can only train with a very small batch size consider using GroupNorm instead of BatchNorm, aka set --norm_type to gn.<br>

<br>
<b> Basic training command: </b><br>
This will create a 51 Million Parameter VAE for a 64x64 sized image and will create a 256x4x4 latent representation.

```
python train_vae.py -mn test_run --dataset_root #path to dataset root#
```

<br>
<b> Starting from an existing checkpoint: </b><br>
The code will attempt to load a checkpoint with the name provided in the "save_dir" specified.

```
python train_vae.py -mn test_run --load_checkpoint --dataset_root #path to dataset root#
```

<br>
<b> Train without a feature loss: </b><br>
This will also stop the VGG19 model from being created and will result in faster training but lower quality image features.

```
python train_vae.py -mn test_run --feature_scale 0 --dataset_root #path to dataset root#
```

<br>
<b> Define a Custom Architecture: </b><br>
Example showing how to define each of the main parameters of the VAE Architecture.

```
python train_vae.py -mn test_run --latent_channels 128 --block_widths 1 2 4 8 --ch_multi 64 --dataset_root #path to dataset root#
```

<br>
<b> Define Deeper Architecture for a larger image: </b><br>
Example showing how to change the image size (128x128) used while keeping the same latent representation (256x4x4) by changing the number of blocks.

```
python train_vae.py -mn test_run --image_size 128 --block_widths 1 2 4 4 8 --dataset_root #path to dataset root#
```

Train with a 128x128 image with a deeper model by adding Res-Identity Blocks to each down/up-sample stage without additional downsampling.

```
python train_vae.py -mn test_run --image_size 128 --deep_model  --latent_channels 64 --dataset_root #path to dataset root#
```

Latent space representation will be 64x8x8, same number of latent variables as before, but a different shape!

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
<br>
<br>
![Celeba trained with perception loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_celeba_64.png)
<br>
<br>
**New Model** Test images from VAE trained on CelebA at 128x128 resolution (latent space is therefore 512x2x2) using all layers of the VGG model for the perception loss
![Celeba 128x128 test images trained with perception loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_CelebA_all_Feat_new_model_128.png)
<br>
<br>

# As Used in:
```
@article{ditria2023long,
  title={Long-Term Prediction of Natural Video Sequences with Robust Video Predictors},
  author={Ditria, Luke and Drummond, Tom},
  journal={arXiv preprint arXiv:2308.11079},
  year={2023}
}
```

