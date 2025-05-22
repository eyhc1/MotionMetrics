import shutil
import os
import numpy as np
import pandas as pd
from zipfile import ZipFile
from glob import glob
from rich import print

# root = "../../data"
root = "."
npy_dir = "data/npy"

labels = ["dws", "ups", "sit", "std", "wlk", "jog"]

# unzip
zip_file = ZipFile(glob(f"{root}/**/*.zip", recursive=True)[0])
zip_file.extractall(f"{root}/Unzipped")
zip_file.close()

# files = glob(f"{root}/**/*dws*/*.csv", recursive=True)

os.makedirs(npy_dir, exist_ok=True)

# # combine all files into one
# frames = [pd.read_csv(file) for file in files]
# df = pd.concat(frames, ignore_index=True, sort=False).to_csv("test.csv", index=False)
    
for label in labels:
    # get all files for the label
    files = glob(f"{root}/Unzipped/**/*{label}*/*.csv", recursive=True)
    print(f"Processing {len(files)} files for label: {label}")

    # combine all files into one
    frames = [pd.read_csv(file) for file in files]
    ds = pd.concat(frames, ignore_index=True, sort=False).to_numpy()
    np.save(f"{npy_dir}/{label}.npy", ds, allow_pickle=False)
    # df = pd.concat(frames, ignore_index=True, sort=False)

    # # save to csv
    # df.to_csv(f"{label}.csv", index=False)

# remove the unzipped files
shutil.rmtree(f"{root}/Unzipped")
