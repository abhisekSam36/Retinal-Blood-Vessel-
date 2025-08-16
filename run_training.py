###################################################
#
#   Script to launch the training (and optionally prediction)
#
##################################################

import os, sys
import configparser
import time

start = time.time()

if len(sys.argv) != 2:
    print("Usage: python run_training.py configuration_drive.txt")
    exit(1)

config_name = sys.argv[1]

# read config
config = configparser.RawConfigParser()
config.read(config_name)

# experiment name
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')   # std output on log file?

# create results folder
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
os.makedirs(result_dir, exist_ok=True)

# copy config into experiment folder
print("2. Copy the configuration file into the results folder")
if sys.platform == 'win32':
    os.system(f'copy {config_name} .\\{name_experiment}\\{name_experiment}_configuration.txt')
else:
    os.system(f'cp {config_name} ./{name_experiment}/{name_experiment}_configuration.txt')

# run training
print("\n3. Run the training")
if nohup:
    os.system(f'nohup python -u retina_unet_training.py --config {config_name} > {name_experiment}/{name_experiment}_training.nohup &')
else:
    os.system(f'python retina_unet_training.py --config {config_name}')

# optional: run prediction after training
print("\n4. Run prediction & evaluation")
os.system(f'python retina_unet_predict.py {config_name}')

end = time.time()
print(f"\n✅ Finished in {(end - start)/60:.2f} minutes. Results saved in: {name_experiment}/")
