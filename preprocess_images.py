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
import sys
import numpy as np
from sten import Sten
import os.path
from tqdm.auto import tqdm, trange

NUM_FILES = int(sys.argv[1])

st = Sten(7)
done = []

# +
path = os.getcwd() + '/data/set1'
num_files_set1 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

path = os.getcwd() + '/data/set2'
num_files_set2 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
# -

i = 0

for set1File in trange(1,11):
    for set2File in trange(1,6):
        
#     set1File = random.randint(1,15)
#     set2File = random.randint(1,5)
        name = str(set1File) + '.' + str(set2File)
        if name not in done:
            encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/{}_{}.npy".format(set2File,i))
            decImg = st.decode("./encodedArray/{}_{}.npy".format(set2File,i), "./decodedArray/{}.npy".format(i))
            done.append(name)
            i += 1
