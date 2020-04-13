#!/mnt/c/Users/nnibr/Miniconda3/envs/tf_gpu_mingo/python.exe
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
import pathlib


# NUM_FILES = int(sys.argv[1])

#done = []

# +
#path = os.getcwd() + '/data/set1'
#num_files_set1 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

#path = os.getcwd() + '/data/set2'
#num_files_set2 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
# -

# i = 0
# for _ in trange(NUM_FILES, desc='File', leave=True):
#    set1File = random.randint(1,int(sys.argv[2]))
#    set2File = random.randint(1,int(sys.argv[3]))
#    name = str(set1File) + '_' + str(set2File)
#    if name not in done:
#        encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/{}.npy".format(name))
#        decImg = st.decode("./encodedArray/{}.npy".format(name), "./decodedArray/{}.npy".format(name))
#        done.append(name)
#        i += 1
for x in trange(7, 2, -1):
    st = Sten(x)
    pathlib.Path("./encodedArray/bit_{0}".format(x)).mkdir(parents=True, exist_ok=True)
    pathlib.Path("./decodedArray/bit_{0}".format(x)).mkdir(parents=True, exist_ok=True)
    for set1File in trange(1, int(sys.argv[1])+1):
        for set2File in trange(1, int(sys.argv[2])+1):
            name = str(set1File) + '_' + str(set2File)
            encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/bit_{0}/{1}.npy".format(x, name))
            decImg = st.decode("./encodedArray/bit_{0}/{1}.npy".format(x, name), "./decodedArray/bit_{0}/{1}.npy".format(x, name))
