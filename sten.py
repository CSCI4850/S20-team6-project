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

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from matplotlib import cm
import imageio

class Sten():

    def __init__(self):
        self.HEIGHT = 256
        self.WIDTH = 256
        self.BITS = 3


    def encode(self, cover_name, hidden_name):
        cover_pix = mpimg.imread(cover_name)
        hidden_pix = mpimg.imread(hidden_name)
        output_image = np.empty((self.HEIGHT, self.WIDTH, 3))
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                cover_R = '{0:08b}'.format(cover_pix[x][y][0])
                cover_G = '{0:08b}'.format(cover_pix[x][y][1])
                cover_B = '{0:08b}'.format(cover_pix[x][y][2])
                hidden_R = '{0:08b}'.format(hidden_pix[x][y][0])
                hidden_G = '{0:08b}'.format(hidden_pix[x][y][1])
                hidden_B = '{0:08b}'.format(hidden_pix[x][y][2])
                output_R = int(cover_R[:8-self.BITS] + hidden_R[:self.BITS], 2)
                output_G = int(cover_G[:8-self.BITS] + hidden_G[:self.BITS], 2)
                output_B = int(cover_B[:8-self.BITS] + hidden_B[:self.BITS], 2)
                output_image[x][y] = [output_R, output_G, output_B]
        return output_image


    def decode(self, hidden_pix):
        hidden_pix = hidden_pix
        output_image = np.empty((self.HEIGHT, self.WIDTH, 3))
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                hidden_R = '{0:08b}'.format(int(hidden_pix[x][y][0]))
                recoverd_R = hidden_R[-self.BITS:] +  ''.join(['0' for a in range(8-self.BITS)])
                hidden_G = '{0:08b}'.format(int(hidden_pix[x][y][1]))
                recoverd_G = hidden_G[-self.BITS:] + ''.join(['0' for a in range(8-self.BITS)])
                hidden_B = '{0:08b}'.format(int(hidden_pix[x][y][2]))
                recoverd_B = hidden_B[-self.BITS:] + ''.join(['0' for a in range(8-self.BITS)])
                output_R = int(recoverd_R, 2) / 255.0
                output_G = int(recoverd_G, 2) / 255.0
                output_B = int(recoverd_B, 2) / 255.0
                output_image[x][y] = [output_R, output_G, output_B]
        return output_image


# hidden_img = encode("cover.jpg", "hidden.jpg")

# decoded_img = decode(hidden_img)

# plt.imshow(mpimg.imread("cover.jpg"))

# plt.imshow(mpimg.imread("hidden.jpg"))

# plt.imshow(hidden_img/255.0)

# plt.imshow(decoded_img)

# print(hidden_img/255.0)
# print(decoded_img)

s = Sten()
hidden_img = s.encode("cover.jpg", "hidden.jpg")
decoded_img = s.decode(hidden_img)

# +
#print(s.encode("cover.jpg", "hidden.jpg")/255.0)
#print(decoded_img)
