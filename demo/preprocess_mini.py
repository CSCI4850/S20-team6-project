# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pathlib
import os.path
import random
import sys
import numpy as np

from sten import Sten
from tqdm.auto import tqdm, trange
from multiprocessing.pool import ThreadPool as Pool


# -


def create(x):
    set1File = x+11
    set2File = x+6
    name = str(set1File) + '_' + str(set2File)
    encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/bit_{0}/{1}.npy".format(x, name))
    decImg = st.decode("./encodedArray/bit_{0}/{1}.npy".format(x, name), "./decodedArray/bit_{0}/{1}.npy".format(x, name))


pool_size = 5

pool = Pool(pool_size)

pathlib.Path("./encodedArray/bit_7").mkdir(parents=True, exist_ok=True)
pathlib.Path("./decodedArray/bit_7").mkdir(parents=True, exist_ok=True)
st = Sten(7)
for x in trange(5):
    pool.apply_async(create, (x,))

pool.close()
pool.join()
