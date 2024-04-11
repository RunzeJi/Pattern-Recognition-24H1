import os
import pandas as pd
from tqdm import tqdm
from itertools import islice
from matplotlib import pyplot as plt

def get_trace_img(file):
    file_df = pd.read_csv(file)
    file_df.drop(['渔船ID','time','速度','方向'], axis=1, inplace=True)
    file_df.head()
        
    return plt.scatter(file_df['lat'], file_df['lon'], s=1)