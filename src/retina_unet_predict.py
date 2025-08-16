###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

# imports
import sys
import time
import math
import numpy as np
import configparser
from matplotlib import pyplot as plt
# Keras
from keras.models import Model
# scikit learn
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, jaccard_score, f1_score, accuracy_score

sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone, recompone_overlap, paint_border, kill_border, pred_only_FOV
from extract_patches import get_data_testing, get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc
# >>> NEW import for Attention U-Net
from retina_unet_training import build_attention_unet_channels_first

# ======== CONFIG FILE ========
config_name = None
if len(sys.argv) == 2:
    config_name = sys.argv[1]
else:
    print("Wrong Argument!")
    exit(1)

config = configparser.RawConfigParser()
config.read('./' + config_name)

# ======== PATHS & SETTINGS ========
path_data = config.get('data paths', 'path_local')

# original test images
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

# DRIVE border masks
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(DRIVE_test_border_masks)

# patch dimensions
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

# stride (for overlap mode)
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

# model experiment name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' + name_experiment + '/'

# N full images to predict
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
# visualization grouping
N_visual = int(config.get('testing settings', 'N_group_visual'))
# average mode
average_mode = config.getboolean('testing settings', 'average_mode')

datasets = {'DRIVE', 'STARE', 'CHASE'}
dataset_name = config.get('data paths', 'dataset_name')
if dataset_name not in datasets:
    print("Dataset NOT supported!")
    exit(1)
if dataset_name == 'DRIVE':
    width, height = 565, 584
elif dataset_name == 'STARE':
    width, height = 700, 605
else:
    width, height = 999, 960
print("Dataset:", dataset_name)

# ============ Load test data ============
patches_imgs_test, new_height, new_width, masks_test, patches_masks_test = None, None, None, None, None
if average_mode:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original=test_imgs_original,
        DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),
        Imgs_to_test=Imgs_to_test,
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        DRIVE_test_imgs_original=test_imgs_original,
        DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),
        Imgs_to_test=Imgs_to_test,
        patch_height=patch_height,
        patch_width=patch_width
    )

# ============ Build & Load Model ============
best_last = config.get('testing settings', 'best_last')
n_ch = 1  # grayscale images

# >>> Use Attention U-Net with CBAM (channels_first wrapper)
model = build_attention_unet_channels_first(n_ch, patch_height, patch_width)

model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')

start = time.time()
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
end = time.time()
print("Inference time (s): ", end - start)
print("Predicted patches size:", predictions.shape)

# ============ Convert back to images ============
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")

if average_mode:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)
    orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])
    gtruth_masks = masks_test
else:
    N_w = math.ceil(width / patch_width)
    N_h = math.ceil(height / patch_height)
    pred_imgs = recompone(pred_patches, N_h, N_w)
    orig_imgs = recompone(patches_imgs_test, N_h, N_w)
    gtruth_masks = recompone(patches_masks_test, N_h, N_w)

kill_border(pred_imgs, test_border_masks)  # apply DRIVE mask

# crop back to original dims
orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]

print("Orig imgs:", orig_imgs.shape)
print("Pred imgs:", pred_imgs.shape)
print("GT imgs:", gtruth_masks.shape)

# ============ Visualize ============
visualize(group_images(orig_imgs, N_visual), path_experiment + "all_originals")
visualize(group_images(pred_imgs, N_visual), path_experiment + "all_predictions")
visualize(group_images(gtruth_masks, N_visual), path_experiment + "all_groundTruths")

# ============ Evaluation ============
print("\n\n========  Evaluate the results =======================")

# Only inside FOV
y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks)
print("y_scores pixels:", y_scores.shape[0])
print("y_true pixels:", y_true.shape[0])

# ROC / AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print("\nArea under the ROC curve:", AUC_ROC)
plt.figure()
plt.plot(fpr, tpr, '-', label='AUC = %0.4f' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig(path_experiment + "ROC.png")

# Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision, recall)
print("Area under PR curve:", AUC_prec_rec)
plt.figure()
plt.plot(recall, precision, '-', label='AUC = %0.4f' % AUC_prec_rec)
plt.title('Precision-Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment + "Precision_recall.png")

# Confusion Matrix
threshold_confusion = 0.5
print("\nConfusion matrix with threshold", threshold_confusion)
y_pred = (y_scores >= threshold_confusion).astype(int)
confusion = confusion_matrix(y_true, y_pred)
print(confusion)

accuracy = accuracy_score(y_true, y_pred)
specificity = confusion[0,0] / (confusion[0,0] + confusion[0,1])
sensitivity = confusion[1,1] / (confusion[1,1] + confusion[1,0])
precision_val = confusion[1,1] / (confusion[1,1] + confusion[0,1])

print("Accuracy:", accuracy)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Precision:", precision_val)

# Jaccard
jaccard_index = jaccard_score(y_true, y_pred)
print("Jaccard score:", jaccard_index)

# F1
F1_score = f1_score(y_true, y_pred)
print("F1 score:", F1_score)

# Save results
with open(path_experiment + 'performances.txt', 'w') as f:
    f.write("AUC ROC: " + str(AUC_ROC) +
            "\nAUC PR: " + str(AUC_prec_rec) +
            "\nJaccard: " + str(jaccard_index) +
            "\nF1 score: " + str(F1_score) +
            "\n\nConfusion Matrix:\n" + str(confusion) +
            "\nAccuracy: " + str(accuracy) +
            "\nSensitivity: " + str(sensitivity) +
            "\nSpecificity: " + str(specificity) +
            "\nPrecision: " + str(precision_val))
