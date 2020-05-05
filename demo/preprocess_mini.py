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


# -


def run():
    for x in trange(5):
        set1File = x+11
        set2File = x+6
        name = str(set1File) + '_' + str(set2File)
        encImg = st.encode("./data/set1/{0}.jpg".format(set1File), "./data/set2/{0}.jpg".format(set2File), "./encodedArray/bit_7/{0}.npy".format(name))
        decImg = st.decode("./encodedArray/bit_7/{0}.npy".format(name), "./decodedArray/bit_7/{0}.npy".format(name))


pathlib.Path("./encodedArray/bit_7").mkdir(parents=True, exist_ok=True)
pathlib.Path("./decodedArray/bit_7").mkdir(parents=True, exist_ok=True)
st = Sten(7)
run()
