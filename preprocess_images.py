# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import random
import numpy as np
from sten import Sten
import os.path
from tqdm.auto import tqdm, trange

NUM_FILES = 100

st = Sten(3)

# +
path = os.getcwd() + '/data/set1'
num_files_set1 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

path = os.getcwd() + '/data/set2'
num_files_set2 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
# -

for i in trange(NUM_FILES, desc='File', leave=True):
    set1File = random.randint(1,num_files_set1)
    set2File = random.randint(1,num_files_set2)
    encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/{}.npy".format(i))
    decImg = st.decode("./decodedArray/{}.jpg".format(i), "./encodedArray/{}.npy".format(i))
