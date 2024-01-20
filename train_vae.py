import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

import Helpers as hf
from vgg19 import VGG19
from RES_VAE_Dynamic import VAE

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--norm_type", "-nt",
                    help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=256)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))
# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
print("")

# Create dataloaders
# This code assumes there is no pre-defined test/train split and will create one for you
print("-Target Image Size %d" % args.image_size)
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)

# Randomly split the dataset with a fixed random seed for reproducibility
test_split = 0.9
n_train_examples = int(len(data_set) * test_split)
n_test_examples = len(data_set) - n_train_examples
train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
test_images, _ = next(dataiter)

# Create AE network.
vae_net = VAE(channel_in=test_images.shape[1],
              ch=args.ch_multi,
              blocks=args.block_widths,
              latent_channels=args.latent_channels,
              num_res_blocks=args.num_res_blocks,
              norm_type=args.norm_type,
              deep_model=args.deep_model).to(device)

# Setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

if args.norm_type == "bn":
    print("-Using BatchNorm")
elif args.norm_type == "gn":
    print("-Using GroupNorm")
else:
    ValueError("norm_type must be bn or gn")

# Create the feature loss module if required
if args.feature_scale > 0:
    feature_extractor = VGG19().to(device)
    print("-VGG19 Feature Loss ON")
else:
    feature_extractor = None
    print("-VGG19 Feature Loss OFF")

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in vae_net.parameters():
    num_model_params += param.flatten().shape[0]

print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
fm_size = args.image_size//(2 ** len(args.block_widths))
print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

# Create the save directory if it does not exist
if not os.path.isdir(args.save_dir + "/Models"):
    os.makedirs(args.save_dir + "/Models")
if not os.path.isdir(args.save_dir + "/Results"):
    os.makedirs(args.save_dir + "/Results")

# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)
if args.load_checkpoint:
    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                                map_location="cpu")
        print("-Checkpoint loaded!")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        vae_net.load_state_dict(checkpoint['model_state_dict'])

        if not optimizer.param_groups[0]["lr"] == args.lr:
            print("Updating lr!")
            optimizer.param_groups[0]["lr"] = args.lr

        start_epoch = checkpoint["epoch"]
        data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
    else:
        raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
else:
    # If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        raise ValueError("Warning Checkpoint exists -> add -cp flag to use this checkpoint")
    else:
        print("Starting from scratch")
        start_epoch = 0
        # Loss and metrics logger
        data_logger = defaultdict(lambda: [])
print("")

# Start training loop
for epoch in trange(start_epoch, args.nepoch, leave=False):
    vae_net.train()
    for i, (images, _) in enumerate(tqdm(train_loader, leave=False)):
        current_iter = i + epoch * len(train_loader)
        images = images.to(device)
        bs, c, h, w = images.shape

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            recon_img, mu, log_var = vae_net(images)

            kl_loss = hf.kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, images)
            loss = args.kl_scale * kl_loss + mse_loss

            # Perception loss
            if feature_extractor is not None:
                feat_in = torch.cat((recon_img, images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += args.feature_scale * feature_loss
                data_logger["feature_loss"].append(feature_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        # Log losses and other metrics for evaluation!
        data_logger["mu"].append(mu.mean().item())
        data_logger["mu_var"].append(mu.var().item())
        data_logger["log_var"].append(log_var.mean().item())
        data_logger["log_var_var"].append(log_var.var().item())

        data_logger["kl_loss"].append(kl_loss.item())
        data_logger["img_mse"].append(mse_loss.item())

        # Save results and a checkpoint at regular intervals
        if (current_iter + 1) % args.save_interval == 0:
            # In eval mode the model will use mu as the encoding instead of sampling from the distribution
            vae_net.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Save an example from testing and log a test loss
                    recon_img, mu, log_var = vae_net(test_images.to(device))
                    data_logger['test_mse_loss'].append(F.mse_loss(recon_img,
                                                                   test_images.to(device)).item())

                    img_cat = torch.cat((recon_img.cpu(), test_images), 2).float()
                    vutils.save_image(img_cat,
                                      "%s/%s/%s_%d_test.png" % (args.save_dir,
                                                                "Results",
                                                                args.model_name,
                                                                args.image_size),
                                      normalize=True)

                # Keep a copy of the previous save in case we accidentally save a model that has exploded...
                if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                    shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
                                    dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

                # Save a checkpoint
                torch.save({
                            'epoch': epoch + 1,
                            'data_logger': dict(data_logger),
                            'model_state_dict': vae_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                             }, args.save_dir + "/Models/" + save_file_name + ".pt")

                # Set the model back into training mode!!
                vae_net.train()
