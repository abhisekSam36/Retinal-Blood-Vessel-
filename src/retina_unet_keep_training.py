###################################################
#
#   Script to:
#   - Load images & extract patches
#   - Load pretrained Attention U-Net (CBAM)
#   - Resume training & save new weights
#
##################################################

import numpy as np
import configparser
import argparse
import sys
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

sys.path.insert(0, './lib/')
from help_functions import *
from extract_patches import get_data_training_rotate
# >>> Import new model builder
from retina_unet_training import build_attention_unet_channels_first

# ---------------- ARGUMENT PARSER ----------------
parser = argparse.ArgumentParser(description='Resume U-Net training')
parser.add_argument('--dataset', '-d', action='store', default='DRIVE', help='Dataset name (DRIVE, STARE, CHASE)')
parser.add_argument('--config', '-c', action='store', default='../configuration_drive.txt',
                    help='Path of Configuration file, default: ../configuration_drive.txt')
args = parser.parse_args()
config_name = args.config

# ---------------- CONFIG ----------------
config = configparser.RawConfigParser()
config.read('./' + config_name)

path_data = config.get('data paths', 'path_local')
experiment_name = config.get('experiment name', 'name')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
best_last = config.get('training settings', 'best_last')

datasets = {'DRIVE', 'STARE', 'CHASE'}
dataset_name = args.dataset  # <-- FIXED
if dataset_name not in datasets:
    print("Dataset NOT supported!")
    exit(1)
print("Dataset:", dataset_name)

# ---------------- LOAD PATCHES ----------------
patches_imgs_train, patches_masks_train = get_data_training_rotate(
    train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV'),
    dataset=dataset_name
)

# save samples for debug
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
          '../' + experiment_name + '/' + "sample_input_imgs")
visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5),
          '../' + experiment_name + '/' + "sample_input_masks")

# ---------------- BUILD ATTENTION U-NET (CBAM) ----------------
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

model = build_attention_unet_channels_first(n_ch, patch_height, patch_width)

# ---------------- LOAD PRETRAINED WEIGHTS ----------------
path_pretrained = './' + experiment_name + '/'
model.load_weights(path_pretrained + experiment_name + '_' + best_last + '_weights.h5')
print("✅ Successfully loaded pretrained Attention U-Net weights")

# ---------------- TRAINING ----------------
checkpointer = ModelCheckpoint(
    filepath='./' + experiment_name + '/' + experiment_name + '_best_weights.h5',
    verbose=1,
    monitor='val_loss',
    mode='auto',
    save_best_only=True
)

patches_masks_train = masks_Unet(patches_masks_train)  # memory optimized
model.fit(
    patches_imgs_train,
    patches_masks_train,
    epochs=N_epochs,                  # <-- FIXED
    batch_size=batch_size,
    verbose=2,
    shuffle=True,
    validation_split=0.1,
    callbacks=[checkpointer]
)

# ---------------- SAVE LAST ----------------
model.save_weights('./' + experiment_name + '/' + experiment_name + '_last_weights.h5', overwrite=True)
print("✅ Training complete. Saved best & last weights.")
