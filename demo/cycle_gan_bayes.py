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

# # Bayesian Optimization

# +
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow_examples.models.pix2pix import pix2pix
from sten import Sten
from matplotlib.image import imread
from IPython.display import clear_output
from tqdm.auto import tqdm, trange
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

import os
import time
import glob
import random
import sys
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
# -

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ## Modify hyper-parameters

# +
STEN_X = int(sys.argv[1])

MAX0 = 11
MAX1 = 6

EPOCHS_RANGE = (2,5)
#(200, 500)
LAMBDA_RANGE = (8, 12)
STEPS_PER_EPOCH_RANGE = (2,3)
#(10, 50)
# -

# ## Helper functions

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(LAMBDA, real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return int(LAMBDA) * loss1


def identity_loss(LAMBDA, real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return int(LAMBDA) * 0.5 * loss


@tf.function
def train_step(LAMBDA,real_x, real_y): 
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)

        cycled_x = generator_f(fake_y, training=True) 

        fake_x = generator_f(real_y, training=True) 
        cycled_y = generator_g(fake_x, training=True) 

        same_x = generator_f(real_x, training=True) 
        same_y = generator_g(real_y, training=True) 

        disc_real_x = discriminator_x(real_x, training=True) 
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True) 
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)  
        gen_f_loss = generator_loss(disc_fake_x) 

        total_cycle_loss = calc_cycle_loss(LAMBDA,real_x, cycled_x) + calc_cycle_loss(LAMBDA,real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(LAMBDA, real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(LAMBDA, real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x) 
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y) 

    generator_g_gradients = tape.gradient(total_gen_g_loss,generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss,discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,discriminator_y.trainable_variables))


# ## Loss for Bayesian Optimization

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return -err


# ## Create the model

# +
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
# -

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# +
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# -

# ## Bayesian Optimization
#
# The Black_Box function parameters are different hyperparameters. The black-box function returns the average difference between the input and output as the metric to maximize. 

def Black_Box(EPOCHS,LAMBDA,steps_per_epochs):
    for epoch in trange(int(EPOCHS),desc='epochs'):
        for _ in trange(int(steps_per_epochs), desc='steps_per_epochs'):
            i = np.random.randint(1,MAX0)
            j = np.random.randint(1,MAX1)
            name = str(i) + '_' + str(j)
            image_x = np.load(os.getcwd() + "/encodedArray/bit_{0}/{1}.npy".format(STEN_X, name))
            image_y = np.load(os.getcwd() + "/decodedArray/bit_{0}/{1}.npy".format(STEN_X, name))
            train_step(LAMBDA,np.asarray([image_x/255.0], dtype='float32'), np.asarray([image_y/255.0], dtype='float32'))
    sum = 0.0
    for i in trange(1,MAX0):
        for j in trange(1,MAX1):
            name = str(i) + '_' + str(j)
            image_x = np.load(os.getcwd() + "/encodedArray/bit_{0}/{1}.npy".format(STEN_X, name))
            image_y = np.load(os.getcwd() + "/decodedArray/bit_{0}/{1}.npy".format(STEN_X, name))
            sum += mse(generator_g.predict(np.asarray([image_x/255.0], dtype='float32')), np.asarray([image_y/255.0], dtype='float32'))
    avg = sum / ((MAX0-1)*(MAX1-1))
    return avg


bounds = {
    'EPOCHS': EPOCHS_RANGE,
    'LAMBDA': LAMBDA_RANGE,
    'steps_per_epochs': STEPS_PER_EPOCH_RANGE 
}


optimizer = BayesianOptimization(
    f = Black_Box,
    pbounds = bounds,
    random_state = 1
)

logger = JSONLogger(path="./logs_{0}.json".format(STEN_X))
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(init_points=2,n_iter=10)
