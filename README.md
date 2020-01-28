
# CNN-VAE
A Res-Net Style VAE with an adjustable perception loss using a pre-trained vgg19

## Results

Results on validation images of the STL10 dataset at 64x64 with a latent vector size of 512 (images on top are the reconstruction)

**With Perception loss**
<br>
![VAE Trained with perception/feature loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_STL10_64.png)


**Without Perception loss**
<br>
![VAE Trained without perception/feature loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_STL10_no_perception_64.png)

## Additional Results - celeba
The images in the STL10 have a lot of variation meaning more "features" need to be encoded in the latent space to achieve a good reconstruction. Using a data-set with less variation (and the same latent vector size) should results in a higher quality reconstructed image.

![Celeba trained with perception loss](https://github.com/LukeDitria/CNN-VAE/blob/master/Results/VAE_celeba_64.png)

