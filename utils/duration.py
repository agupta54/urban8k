import sox
import glob
import numpy as np
import sys
from tqdm import tqdm
import os
from joblib import Parallel, delayed


def audio_duration(local_file):
    return sox.file_info.duration(local_file)


def get_duration(path_local):
    if not os.path.exists(path_local):
        raise Exception("Sorry this path doesn't exists")
    files = glob.glob(path_local + '/**/*.wav', recursive=True)
    print("Number of files present: ", len(files))

    durations = [Parallel(n_jobs=24)(delayed(audio_duration)(local_file) for local_file in tqdm(files))]

    return durations


if __name__ == "__main__":
    path = sys.argv[1]
    audio_durations = get_duration(path)
    print(np.sum(audio_durations) / 3600)
