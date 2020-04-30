from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from matplotlib import cm
import imageio

class Sten():

    def __init__(self, bits):
        self.HEIGHT = 256
        self.WIDTH = 256
        self.BITS = bits


    def encode(self, cover_name, hidden_name, save_name):        
        cover_pix = Image.open(cover_name)
        hidden_pix = Image.open(hidden_name)
        
        img1_tmp = cover_pix.resize((self.HEIGHT,self.WIDTH), Image.ANTIALIAS)
        cover_pix = np.array(img1_tmp)
        
        img2_tmp = hidden_pix.resize((self.HEIGHT,self.WIDTH), Image.ANTIALIAS)
        hidden_pix = np.array(img2_tmp)
        
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
                cover_pix[x][y] = [output_R, output_G, output_B]
        np.save(save_name, cover_pix)
        return cover_pix

    def decode(self, save_name, output_name):
        working_tmp = np.load(save_name)
        working = np.uint8(working_tmp)
        output_image = np.empty((self.HEIGHT, self.WIDTH, 3))
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                hidden_R = '{0:08b}'.format(int(working[x][y][0]))
                recoverd_R = hidden_R[-self.BITS:] +  ''.join(['0' for a in range(8-self.BITS)])
                hidden_G = '{0:08b}'.format(int(working[x][y][1]))
                recoverd_G = hidden_G[-self.BITS:] + ''.join(['0' for a in range(8-self.BITS)])
                hidden_B = '{0:08b}'.format(int(working[x][y][2]))
                recoverd_B = hidden_B[-self.BITS:] + ''.join(['0' for a in range(8-self.BITS)])
                output_R = int(recoverd_R, 2) 
                output_G = int(recoverd_G, 2) 
                output_B = int(recoverd_B, 2) 
                working[x][y] = [output_R, output_G, output_B]
        np.save(output_name, working)
        return working




