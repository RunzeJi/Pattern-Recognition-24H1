import os, threading

from torch.utils.data import Dataset, DataLoader
import torch

def newThread():

    print(f'[{os.getpid()}] New Thread Created')

new_tread = threading.Thread(target=newThread)

new_tread.start()

new_tread.join()

