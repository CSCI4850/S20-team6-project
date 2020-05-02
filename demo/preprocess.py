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
    st = Sten(x)
    pathlib.Path("./encodedArray/bit_{0}".format(x)).mkdir(parents=True, exist_ok=True)
    pathlib.Path("./decodedArray/bit_{0}".format(x)).mkdir(parents=True, exist_ok=True)
    for set1File in trange(1, int(sys.argv[1])+1):
        for set2File in trange(1, int(sys.argv[2])+1):
            name = str(set1File) + '_' + str(set2File)
            encImg = st.encode("./data/set1/{}.jpg".format(set1File), "./data/set2/{}.jpg".format(set2File), "./encodedArray/bit_{0}/{1}.npy".format(x, name))
            decImg = st.decode("./encodedArray/bit_{0}/{1}.npy".format(x, name), "./decodedArray/bit_{0}/{1}.npy".format(x, name))


pool_size = 9

pool = Pool(pool_size)

for x in trange(0, 9):
    pool.apply_async(create, (x,))

pool.close()
pool.join()
