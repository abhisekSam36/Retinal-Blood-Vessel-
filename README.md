Retinal Blood Vessel Segmentation
This project provides a deep learning pipeline for segmenting blood vessels in retinal fundus images using TensorFlow and Keras. It includes implementations of the U-Net and LadderNet architectures, along with preprocessing scripts and a framework for training and testing on public datasets.

Visualizations
Model Architectures

U-Net Architecture Diagram (U-Net.png)
LadderNet Architecture Diagram (LadderNet.png)

Segmentation Results

The figure below shows sample segmentation results from the original paper on the DRIVE, STARE, and CHASE_DB1 datasets.
Example results from the paper (a) Original Image (b) Ground Truth (c) U-Net Prediction (d) LadderNet Prediction.

Key Features
Two Architectures: Implements both U-Net and LadderNet for vessel segmentation.

Multiple Datasets: Supports the DRIVE, STARE, and CHASE_DB1 public datasets.

Preprocessing Pipeline: Includes a pipeline for converting images to grayscale, performing CLAHE enhancement, and gamma correction to improve vessel visibility.

Patch-Based Training: Uses a patch extraction strategy to handle high-resolution images efficiently.

Modern Environment: Updated to run on modern Python with TensorFlow 2, including support for Apple Silicon GPUs.

Setup and Installation
Prerequisites

macOS or Linux

Python 3.9+

Homebrew (for macOS users, to install Graphviz)

Git and Git Large File Storage (LFS)

Step 1: Clone the Repository

Clone the repository from GitHub. The git lfs pull command is essential to download the large .hdf5 dataset files.

Bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
git lfs pull
Step 2: Create Virtual Environment

Create and activate a Python virtual environment to manage dependencies.

Bash
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies

Install the required packages. This command is tailored for macOS with Apple Silicon.

Bash
# Install Python packages
pip install tensorflow-macos tensorflow-metal opencv-python pillow h5py scikit-learn matplotlib pydot

# Install Graphviz system dependency (for plotting model architecture)
brew install graphviz
Usage Workflow
Step 1: Download Raw Datasets

Manually download the raw images for the desired datasets (e.g., DRIVE) and place them in the project folder according to the structure expected by the preparation scripts (e.g., ./DRIVE/training/images/).

Step 2: Prepare Datasets

Run the corresponding script to convert the raw images into the .hdf5 format required for training.

Bash
# Example for the DRIVE dataset
python prepare_datasets_DRIVE.py
Step 3: Configure Experiment

Before training, edit the appropriate configuration file (e.g., configuration_drive.txt). It is highly recommended to set a unique name for each experiment to save results in a separate folder.

Ini, TOML
[experiment name]
name = my_first_gnet_experiment
Step 4: Run Training

Launch the training process using the run_training.py script and your chosen configuration file.

Bash
python run_training.py configuration_drive.txt
Step 5: Run Testing

After training is complete, evaluate the model on the test set.

Bash
python run_testing.py configuration_drive.txt
The results, including predicted images and a performances.txt file with metrics, will be saved in your experiment folder.

Experimenting
You can easily experiment with different models and settings:

Switching Models: To switch between U-Net and LadderNet, simply change the function call from model = get_gnet(...) to model = get_unet(...) in the src/retina_unet_training.py file.

Changing Hyperparameters: Adjust settings like the number of epochs (N_epochs), batch size (batch_size), and number of patches (N_subimgs) in the configuration files to see how they affect performance.

Benchmark Results
The following are the benchmark results from the original research paper for comparison.

Dataset	Method	F1-score	Accuracy	Sn	Sp
DRIVE	U-Net	0.8169	0.9533	0.7675	0.9814
LadderNet	0.8219	0.9555	0.7899	0.9808
STARE	U-Net	0.8393	0.9698	0.8159	0.9863
LadderNet	0.8203	0.9678	0.7621	0.9880
My Experimental Results
The following results were achieved by running the LadderNet architecture on the DRIVE dataset using an Apple M2 GPU.

Metric	Score
F1-score (F-measure)	0.8101
Global Accuracy	0.9548
Sensitivity (Recall)	0.7565
Specificity	0.9838
Area under the ROC curve	0.9770
Area under Precision-Recall curve	0.9054
Jaccard similarity score	0.6808
File Descriptions
run_*.py: Launcher scripts for training and testing.

prepare_datasets_*.py: Scripts to process raw datasets.

configuration_*.txt: Configuration files for experiment parameters.

src/: Contains the core logic, including model definitions (retina_unet_training.py) and prediction code (retina_unet_predict.py).

lib/: Contains helper functions for data handling and processing.

Images/: Contains architecture and result diagrams for this README.

.gitignore: Specifies which files and folders (like venv and datasets) for Git to ignore.

.gitattributes: Configures Git LFS to handle large .hdf5 files.

Bibliography
[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
[2] Islam, M. M., & S. M. D. (2017). A novel method for retinal blood vessel segmentation. MICCAI.
[3] Alom, M. Z., Hasan, M., Yakopcic, C., Taha, T. M., & Asari, V. K. (2018). Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation. arXiv.
[4] Owen, C. G., Rudnicka, A. R., Nightingale, C. M., & Mullen, R. (2012). Retinal vessel tortuosity: a valid and reliable biometric? Current eye research.
[5] Staal, J., Abràmoff, M. D., Niemeijer, M., Viergever, M. A., & van Ginneken, B. (2004). Ridge-based vessel segmentation in fundus images. IEEE Transactions on Medical Imaging.
[6] Fraz, M. M., Remagnino, P., Hoppe, A., Uyyanonvara, B., Rudnicka,A. R., Owen, C. G., & Barman, S. A. (2012). An ensemble classification-based approach to retinal vessel segmentation in funduscopic images. IEEE Transactions on Biomedical Engineering.
[7] Ricci, E., & Perfetti, R. (2007). Retinal blood vessel segmentation using line operators and support vector classification. IEEE Transactions on Medical Imaging.
[8] Zana, F., & Klein, J. C. (2001). Segmentation of vessel-like patterns using mathematical morphology and curvature evaluation. IEEE transactions on image processing.