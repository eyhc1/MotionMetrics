import shutil
import os
import requests
import numpy as np
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
from glob import glob
from rich import print
from build_data import *

# root = "../../data"
root = "."
npy_dir = "data/npy"

labels = ["dws", "ups", "sit", "std", "wlk", "jog"]

# download the zip file
url = "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/A_DeviceMotion_data.zip"  # contains all the data files
# url = "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/B_Accelerometer_data.zip"  # contains only accelerometer data
response = requests.get(url)
if response.status_code == 200:
    for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading", unit="KB"):
        with open(f"{root}/data/A_DeviceMotion_data.zip", "ab") as f:
            f.write(chunk)

# unzip
zip_file = ZipFile(glob(f"{root}/**/*.zip", recursive=True)[0])
zip_file.extractall(f"{root}/data")
zip_file.close()

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
trial_codes = [TRIAL_CODES[act] for act in ACT_LABELS]
dataset = creat_time_series(set_data_types(["userAcceleration"]), ACT_LABELS, trial_codes, mode="raw", labeled=True, data_dir=glob("**/A_DeviceMotion_data", recursive=True)[0])
print("Shape of time-Series dataset:"+str(dataset.shape))    

test_trail = [11,12,13,14,15,16]  
print("Test Trials: "+str(test_trail))
test_ts = dataset.loc[(dataset['trial'].isin(test_trail))]
train_ts = dataset.loc[~(dataset['trial'].isin(test_trail))]

## This Variable Defines the Size of Sliding Window
## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor) 
w = 128 # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
s = 10

train_data, act_train, id_train, train_mean, train_std = ts_to_secs(train_ts.copy(),
                                                                   w,
                                                                   s,
                                                                   standardize = True)
test_data, act_test, id_test, test_mean, test_std = ts_to_secs(test_ts.copy(),
                                                              w,
                                                              s,
                                                              standardize = True,
                                                              mean = train_mean, 
                                                              std = train_std)
