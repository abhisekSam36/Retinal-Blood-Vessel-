###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import sys

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, Permute, Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.utils import plot_model as plot

sys.path.insert(0, './lib/')
from help_functions import *

# function to obtain data for training/testing (validation)
from extract_patches import get_data_training


# Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = Reshape((2,patch_height*patch_width))(conv6)
    conv6 = Permute((2,1))(conv6)
    ############
    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def get_gnet(n_ch, patch_height, patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    #
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    #
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv10 = Reshape((2,patch_height*patch_width))(conv10)
    conv10 = Permute((2,1))(conv10)
    ############
    conv10 = Activation('softmax')(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
    return model
if __name__ == "__main__":
    #========= Load settings from Config file
    config = configparser.RawConfigParser()
    config_name = sys.argv[1]
    config.read(config_name)
    #patch to the datasets
    path_data = config.get('data paths', 'path_local')
    #Experiment name
    name_experiment = config.get('experiment name', 'name')
    #training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))



    #============ Load the data and divided in patches
    # Note: the parameter names below are hardcoded for the DRIVE dataset.
    # If you use this with STARE or CHASE, you might need to change them.
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
        DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
        N_subimgs = int(config.get('training settings', 'N_subimgs')),
        inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
    )


    #========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0],40)
    visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
    visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


    #=========== Construct and save the model arcitecture =====
    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]
    model = get_gnet(n_ch, patch_height, patch_width)  # Using the g-net/LadderNet model
    print("Check: final output of the network:")
    print(model.output_shape)
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    with open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w') as f:
        f.write(json_string)


    #============  Training ==================================
    checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

    patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
    model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


    #========== Save and test the last model ===================
    model.save_weights('./'+name_experiment+'/'+name_experiment +'_last.weights.h5', overwrite=True)