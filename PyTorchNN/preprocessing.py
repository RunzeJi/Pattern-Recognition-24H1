import os
import pandas
from tqdm import tqdm
from itertools import islice

TRAIN_FILES_COUNT = 18000
TRAIN_FILES_OFFSET = 0
EPOCH = 20

TRAIN_FILES_PATH = '../../PR/train'
TEST_DATASET_PATH = '../../PR/test_dataset'
MODEL_PATH = '../../PR/model/'
MODEL_NAME = 'model.ptm'
LOGS_PATH = '../../PR/logs'

