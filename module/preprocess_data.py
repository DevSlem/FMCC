import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from module.get_mfcc import get_mfcc

def preprocess_data(data_list):
    feature_list = []
    for i in tqdm(range(len(data_list))):
        feature_list.append(get_mfcc(data_list[i]))
    return pd.DataFrame(feature_list)