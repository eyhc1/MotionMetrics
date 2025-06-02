import numpy as np
import pandas as pd
# import rich.traceback
from glob import glob
from tqdm import tqdm
# from rich import print

# rich.traceback.install(show_locals=True)

TOTAL_SUBJECTS = 24

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True, data_dir='A_DeviceMotion_data'):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.
        data_dir: The directory where the data files are stored.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list) * 3

    if labeled:
        dataset = np.zeros((0, num_data_cols + 3))  # "3" --> [act, id, trial] 
    else:
        dataset = np.zeros((0, num_data_cols))
    
    for sub_id in tqdm(range(1, TOTAL_SUBJECTS+1), desc="Creating Time-Series for Subjects", unit="sub"):
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                # fname = 'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                fname = f'{data_dir}/{act}_{trial}/sub_{int(sub_id)}.csv'
                try:
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(['Unnamed: 0'], axis=1, errors='ignore')
                    
                    vals = np.zeros((len(raw_data), num_data_cols))
                    for x_id, axes in enumerate(dt_list):
                        if mode == "mag":
                            vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                        else:
                            vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    
                    if labeled:
                        lbls = np.array([[act_id,
                                sub_id-1,
                                trial          
                               ]]*len(raw_data))
                        vals = np.concatenate((vals, lbls), axis=1)
                    
                    dataset = np.append(dataset, vals, axis=0)
                    
                except FileNotFoundError:
                    print(f"[WARNING] -- File not found: {fname}")
                    continue
                except Exception as e:
                    print(f"[ERROR] -- Error processing {fname}: {e}")
                    continue
    
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset

def ts_to_secs(dataset, w, s, standardize = False, features = 3, **options):
    
    data = dataset[dataset.columns[:-features]].values    
    act_labels = dataset["act"].values
    id_labels = dataset["id"].values
    trial_labels = dataset["trial"].values

    mean = 0
    std = 1
    if standardize:
        ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
        ## As usual, we normalize test dataset by training dataset's parameters 
        if options:
            mean = options.get("mean")
            std = options.get("std")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
        data -= mean
        data /= std


    ## We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    m = data.shape[0]   # Data Dimension 
    ttp = data.shape[1] # Total Time Points
    number_of_secs = int(round(((ttp - w)/s)))

    ##  Create a 3D matrix for Storing Sections  
    secs_data = np.zeros((number_of_secs , m , w ))
    act_secs_labels = np.zeros(number_of_secs)
    id_secs_labels = np.zeros(number_of_secs)

    k=0
    for i in range(0 , ttp-w, s):
        j = i // s
        if j >= number_of_secs:
            break
        if id_labels[i] != id_labels[i+w-1]: 
            continue
        if act_labels[i] != act_labels[i+w-1]: 
            continue
        if trial_labels[i] != trial_labels[i+w-1]:
            continue
            
        secs_data[k] = data[:, i:i+w]
        act_secs_labels[k] = act_labels[i].astype(int)
        id_secs_labels[k] = id_labels[i].astype(int)
        k = k+1
        
    secs_data = secs_data[0:k]
    act_secs_labels = act_secs_labels[0:k]
    id_secs_labels = id_secs_labels[0:k]
    return secs_data, act_secs_labels, id_secs_labels, mean, std
