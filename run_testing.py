###################################################
#
#   Script to execute the prediction & evaluation
#
##################################################

import os, sys, time, configparser

start = time.time()

if len(sys.argv) != 2:
    print("Usage: python run_testing.py configuration_drive.txt")
    exit(1)

config_name = sys.argv[1]

# read config
config = configparser.RawConfigParser()
config.read(config_name)

# experiment name
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')   # log to file?

# create result folder if not existing
result_dir = name_experiment
os.makedirs(result_dir, exist_ok=True)

# run prediction
print("\n1. Running prediction...")
if nohup:
    cmd = f'nohup python -u retina_unet_predict.py {config_name} > {name_experiment}/{name_experiment}_prediction.nohup &'
else:
    cmd = f'python retina_unet_predict.py {config_name}'

print(f"Executing: {cmd}")
os.system(cmd)

end = time.time()
print(f"\n✅ Prediction completed in {(end-start):.2f} sec. Results in: {name_experiment}/")
