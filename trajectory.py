import os
import pandas as pd
from tqdm import tqdm
from itertools import islice
from matplotlib import pyplot as plt


dict = {
    '围网':0,
    '刺网':1,
    '拖网':2
    }

path_dict = {
    0:f'/Users/macosexternal/Desktop/PR/classification/custom_dataset/cat0/',
    1:f'/Users/macosexternal/Desktop/PR/classification/custom_dataset/cat1/',
    2:f'/Users/macosexternal/Desktop/PR/classification/custom_dataset/cat2/'
    }

BASE_PATH = '/Users/macosexternal/Desktop/PR/train/'

def get_trace_img(count):
    file_df = pd.read_csv(f'{BASE_PATH}{count}.csv')
    file_df.drop(['渔船ID','time','速度','方向'], axis=1, inplace=True)
    file_df.head()

    cat = dict[file_df['type'].iloc[0]]
    #print(cat)

    plt.scatter(file_df['lon'], file_df['lat'], s=1)
    #plt.show()


    plt.savefig(f'{path_dict[cat]}{count}.jpg', dpi=250)
    plt.clf()

for count in tqdm(range(1,18329), 'plotting and saving'):
    get_trace_img(count)