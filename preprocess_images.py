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

st = Sten(7)
done = []

# +
#path = os.getcwd() + '/data/set1'
#num_files_set1 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

#path = os.getcwd() + '/data/set2'
#num_files_set2 = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
# -

for set1File in trange(1,11):
    for set2File in trange(1,6):
        name = str(set1File) + '_' + str(set2File)
        encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/{}.npy".format(name))
        decImg = st.decode("./encodedArray/{}.npy".format(name), "./decodedArray/{}.npy".format(name))
