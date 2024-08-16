import os
import gdown
import numpy as np
import pandas as pd

from src.config import DatasetConfig
from sklearn.model_selection import train_test_split

def download_dataset():
    file_id = '1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    DATASET_DIR = 'src/data'
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    gdown.download(url,
                   output=DatasetConfig.DATASET_PATH,
                   quiet=True,
                   fuzzy=True)
    
def split_dataset(x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=DatasetConfig.VAL_SIZE,
                                                      shuffle=DatasetConfig.IS_SHUFFLE,
                                                      random_state=DatasetConfig.RANDOM_SEED)
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                      test_size=DatasetConfig.TEST_SIZE,
                                                      shuffle=DatasetConfig.IS_SHUFFLE,
                                                      random_state=DatasetConfig.RANDOM_SEED)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def load_df(csv_path):
    if not os.path.exists(csv_path):
        try:
            download_dataset()
        except Exception as e:
            ERROR_MSG = 'Failed when attempting download the dataset. Please check the download process.'
            raise e(ERROR_MSG)
    df = pd.read_csv(csv_path)

    return df

    
    
    