# CNN-VAE
A Res-Net Style VAE with a adjustable perception loss using a pre-trained vgg19

## Results

Results on validation images of the STL10 dataset at 64x64 with a latent vector size of 512 (images on top are the reconstruction)

**With Perception loss**
![VAE Trained with perception/feature loss](https://drive.google.com/open?id=1zP64Ku1GSFM7rqm_N-zL7S68gRFcoGc4)
**Without Perception loss**
![VAE Trained without perception/feature loss](https://drive.google.com/open?id=1uRcK7A2KDyvXTxrxqCpalTd--7PujgHk)
## Additional Results - celeba
The images in the STL10 have a lot of variation meaning more "features" need to be encoded in the latent space to achieve a good reconstruction. Using a data-set with less variation (and the same latent vector size) should results in higher image quality.

![Celeba trained with perception loss](https://drive.google.com/open?id=1m-s1PUJ5SeBLtRQUvQjaYBgMBALVQd3l)
