#!/usr/bin/env python3
"""
train_seg_enhanced.py

Upgraded segmentation training script:
- Config-driven model selection (res_att_unet, baseline_unet, gnet)
- Residual + Attention U-Net (recommended)
- Dice + BCE loss for class imbalance
- Callbacks: ModelCheckpoint, ReduceLROnPlateau, CSVLogger
- Saves architecture JSON and model plot (if pydot+graphviz installed)

Depends on your existing helpers in ./lib/:
 - get_data_training (from extract_patches)
 - masks_Unet, visualize, group_images (from help_functions)
"""

import os
import sys
import configparser
import numpy as np

# Use tensorflow.keras for compatibility
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                     concatenate, Dropout, Reshape, Permute, Activation,
                                     Add, Multiply, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.utils import plot_model

# local helpers (your project)
sys.path.insert(0, './lib/')
from help_functions import visualize, group_images, masks_Unet
from extract_patches import get_data_training

# ----------------------------
# Losses and metrics
# ----------------------------
def dice_coef(y_true, y_pred, smooth=1.):
    """
    y_true: one-hot last dim (or second dim depending reshape)
    y_pred: probabilities
    We'll compute dice over the class 1 (foreground) vs background if needed.
    """
    # Reshape to (batch, H*W, classes) if necessary
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1, tf.shape(y_true)[-1]))
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))
    # sum over spatial dims, compute dice per class, then mean over classes
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    dice_per_class = (2. * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice_per_class)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred, bce_weight=0.5):
    bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    return bce_weight * bce + (1.0 - bce_weight) * dloss

# ----------------------------
# Blocks: residual conv + attention
# ----------------------------
def res_conv_block(x, filters, kernel_size=3, dropout=0.2, data_format='channels_first'):
    """Residual convolutional block with two convs and a 1x1 shortcut."""
    shortcut = Conv2D(filters, (1, 1), padding="same", data_format=data_format)(x)
    out = Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu', data_format=data_format)(x)
    out = Dropout(dropout)(out)
    out = Conv2D(filters, (kernel_size, kernel_size), padding='same', data_format=data_format)(out)
    out = Add()([out, shortcut])
    out = Activation('relu')(out)
    return out

def attention_gate(x, g, inter_channels, data_format='channels_first'):
    """
    Attention gate as used in Attention U-Net.
    x: skip connection feature map (encoder)
    g: gating signal (decoder feature)
    returns: filtered skip connection
    """
    # 1x1 conv to reduce channels
    theta_x = Conv2D(inter_channels, (1,1), padding='same', data_format=data_format)(x)
    phi_g = Conv2D(inter_channels, (1,1), padding='same', data_format=data_format)(g)
    add = Activation('relu')(Add()([theta_x, phi_g]))
    psi = Conv2D(1, (1,1), padding='same', activation='sigmoid', data_format=data_format)(add)
    # multiply attention coefficients with the skip connection (broadcasted)
    attn = Multiply()([x, psi])
    return attn

# ----------------------------
# Model builders
# ----------------------------
def baseline_unet(n_ch, patch_height, patch_width, data_format='channels_first'):
    """A slightly cleaned-up baseline U-Net that preserves your original output reshape."""
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same', data_format=data_format)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same', data_format=data_format)(conv1)
    pool1 = MaxPooling2D((2,2), data_format=data_format)(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', data_format=data_format)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', data_format=data_format)(conv2)
    pool2 = MaxPooling2D((2,2), data_format=data_format)(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same', data_format=data_format)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same', data_format=data_format)(conv3)

    up1 = UpSampling2D((2,2), data_format=data_format)(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3,3), activation='relu', padding='same', data_format=data_format)(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3,3), activation='relu', padding='same', data_format=data_format)(conv4)

    up2 = UpSampling2D((2,2), data_format=data_format)(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3,3), activation='relu', padding='same', data_format=data_format)(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3,3), activation='relu', padding='same', data_format=data_format)(conv5)

    conv6 = Conv2D(2, (1,1), activation='relu', padding='same', data_format=data_format)(conv5)
    conv6 = Reshape((2, patch_height*patch_width))(conv6)
    conv6 = Permute((2,1))(conv6)
    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    return model

def res_att_unet(n_ch, patch_height, patch_width, data_format='channels_first'):
    """Residual + Attention U-Net. Output reshaped to the same format as before."""
    inputs = Input(shape=(n_ch, patch_height, patch_width))

    # Encoder
    c1 = res_conv_block(inputs, 32, data_format=data_format)
    p1 = MaxPooling2D((2,2), data_format=data_format)(c1)

    c2 = res_conv_block(p1, 64, data_format=data_format)
    p2 = MaxPooling2D((2,2), data_format=data_format)(c2)

    c3 = res_conv_block(p2, 128, data_format=data_format)
    p3 = MaxPooling2D((2,2), data_format=data_format)(c3)

    c4 = res_conv_block(p3, 256, data_format=data_format)
    p4 = MaxPooling2D((2,2), data_format=data_format)(c4)

    # Bottleneck
    bn = res_conv_block(p4, 512, data_format=data_format)

    # Decoder with attention gates on skip connections
    u1 = UpSampling2D((2,2), data_format=data_format)(bn)
    att1 = attention_gate(c4, u1, inter_channels=128, data_format=data_format)
    m1 = concatenate([u1, att1], axis=1)
    c5 = res_conv_block(m1, 256, data_format=data_format)

    u2 = UpSampling2D((2,2), data_format=data_format)(c5)
    att2 = attention_gate(c3, u2, inter_channels=64, data_format=data_format)
    m2 = concatenate([u2, att2], axis=1)
    c6 = res_conv_block(m2, 128, data_format=data_format)

    u3 = UpSampling2D((2,2), data_format=data_format)(c6)
    att3 = attention_gate(c2, u3, inter_channels=32, data_format=data_format)
    m3 = concatenate([u3, att3], axis=1)
    c7 = res_conv_block(m3, 64, data_format=data_format)

    u4 = UpSampling2D((2,2), data_format=data_format)(c7)
    att4 = attention_gate(c1, u4, inter_channels=16, data_format=data_format)
    m4 = concatenate([u4, att4], axis=1)
    c8 = res_conv_block(m4, 32, data_format=data_format)

    # Final conv, reshape to (batch, H*W, classes) and softmax (to match your masks_Unet expectation)
    conv_final = Conv2D(2, (1,1), padding='same', data_format=data_format)(c8)
    conv_final = Reshape((2, patch_height*patch_width))(conv_final)
    conv_final = Permute((2,1))(conv_final)
    out = Activation('softmax')(conv_final)

    model = Model(inputs=inputs, outputs=out)
    return model

# Keep your original g-net builder for compatibility (slightly cleaned)
def gnet(n_ch, patch_height, patch_width, data_format='channels_first'):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(conv1)
    up1 = UpSampling2D(size=(2, 2), data_format=data_format)(conv1)
    #
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=data_format)(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=data_format)(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv2)
    #
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv3)
    #
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format)(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format)(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv4)
    #
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format)(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format)(conv5)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2), data_format=data_format)(conv5), conv4], axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format)(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format)(conv6)
    #
    up3 = concatenate([UpSampling2D(size=(2, 2), data_format=data_format)(conv6), conv3], axis=1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(conv7)
    #
    up4 = concatenate([UpSampling2D(size=(2, 2), data_format=data_format)(conv7), conv2], axis=1)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=data_format)(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=data_format)(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv8)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format)(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format=data_format)(conv9)
    conv10 = Reshape((2,patch_height*patch_width))(conv10)
    conv10 = Permute((2,1))(conv10)
    conv10 = Activation('softmax')(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    return model

# ----------------------------
# Main
# ----------------------------
def main():
    # Load config
    config = configparser.RawConfigParser()
    if len(sys.argv) < 2:
        print("Usage: python train_seg_enhanced.py configuration_drive.txt")
        sys.exit(1)
    config_name = sys.argv[1]
    config.read(config_name)

    # Paths and basic settings
    path_data = config.get('data paths', 'path_local')
    name_experiment = config.get('experiment name', 'name')
    os.makedirs(name_experiment, exist_ok=True)

    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    model_choice = config.get('model', 'architecture') if config.has_section('model') else 'res_att_unet'
    data_format = 'channels_first'  # keep compatibility with your original code

    # Load patches (keeps your original call)
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
        DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
        N_subimgs = int(config.get('training settings', 'N_subimgs')),
        inside_FOV = config.getboolean('training settings', 'inside_FOV')
    )

    # Preview a sample
    N_sample = min(patches_imgs_train.shape[0], 40)
    visualize(group_images(patches_imgs_train[0:N_sample,:,:,:], 5), './' + name_experiment + '/' + "sample_input_imgs")
    visualize(group_images(patches_masks_train[0:N_sample,:,:,:], 5), './' + name_experiment + '/' + "sample_input_masks")

    # Build model
    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]

    if model_choice.lower() == 'res_att_unet':
        model = res_att_unet(n_ch, patch_height, patch_width, data_format=data_format)
    elif model_choice.lower() == 'baseline_unet':
        model = baseline_unet(n_ch, patch_height, patch_width, data_format=data_format)
    elif model_choice.lower() == 'gnet':
        model = gnet(n_ch, patch_height, patch_width, data_format=data_format)
    else:
        print("Unknown model in config. Falling back to res_att_unet.")
        model = res_att_unet(n_ch, patch_height, patch_width, data_format=data_format)

    print("Model output shape (should be [None, H*W, 2]):", model.output_shape)

    # Save architecture
    try:
        plot_model(model, to_file=os.path.join(name_experiment, name_experiment + '_model.png'), show_shapes=True)
    except Exception as e:
        print("Warning: could not plot model (pydot/graphviz may be missing):", e)

    # Write JSON architecture
    json_string = model.to_json()
    with open(os.path.join(name_experiment, name_experiment + '_architecture.json'), 'w') as f:
        f.write(json_string)

    # Compile with combined loss
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=bce_dice_loss,
                  metrics=['accuracy', dice_coef])

    # Callbacks
    ckpt_path = os.path.join(name_experiment, name_experiment + '_best_weights.h5')
    checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)
    csv_logger = CSVLogger(os.path.join(name_experiment, name_experiment + '_training_log.csv'))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

    # Preprocess masks to memory-efficient format (your helper)
    patches_masks_train = masks_Unet(patches_masks_train)

    # Fit
    model.fit(
        patches_imgs_train, patches_masks_train,
        epochs=N_epochs,
        batch_size=batch_size,
        verbose=2,
        shuffle=True,
        validation_split=0.1,
        callbacks=[checkpointer, reduce_lr, csv_logger, early_stop]
    )

    # Save final weights
    model.save_weights(os.path.join(name_experiment, name_experiment + '_last.weights.h5'), overwrite=True)
    print("Training completed. Best weights saved to:", ckpt_path)

if __name__ == '__main__':
    main()
