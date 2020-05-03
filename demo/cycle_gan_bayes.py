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

# # CycleGAN Practice

# +
from __future__ import absolute_import, division, print_function, unicode_literals
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from IPython.display import clear_output
from sten import Sten
from tensorflow_examples.models.pix2pix import pix2pix
from tqdm.auto import tqdm, trange
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

#import tensorflow_datasets as tfds
#tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE
# -

SEED=1

tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_CHANNELS = 3
# pix2pix it has trained of an ordered pair
# pix2pix pretrained model and it returns a model
#generator_g, discriminato_x collection 1 horse 
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
# discriminator_x zebras 
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

#LAMBDA is the learning rate
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# +
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5 #cute the discriminator in half so we dont over shot

def generator_loss(generated):
      return loss_obj(tf.ones_like(generated), generated)


# -

def calc_cycle_loss(LAMBDA, real_image, cycled_image):
      loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
      return int(LAMBDA) * loss1


def identity_loss(LAMBDA, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return int(LAMBDA) * 0.5 * loss


# +
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# +
@tf.function
def train_step(LAMBDA,real_x, real_y): 
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    # G connected to Y
    # F connected to X
    fake_y = generator_g(real_x, training=True)
    
    cycled_x = generator_f(fake_y, training=True) 

    fake_x = generator_f(real_y, training=True) 
    cycled_y = generator_g(fake_x, training=True) 

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True) 
    same_y = generator_g(real_y, training=True) 

    disc_real_x = discriminator_x(real_x, training=True) 
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True) 
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    # we want to see how far the fake image is to the real one
    gen_g_loss = generator_loss(disc_fake_y)  
    gen_f_loss = generator_loss(disc_fake_x) 
    
    
    # it says real_x should be the same as Cycled_x and same does for Y
    total_cycle_loss = calc_cycle_loss(LAMBDA,real_x, cycled_x) + calc_cycle_loss(LAMBDA,real_y, cycled_y)
       
    # Total generator loss = adversarial loss + cycle loss
   
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(LAMBDA, real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(LAMBDA, real_x, same_x)
    
    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x) 
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y) 
  
  # Calculate the gradients for generator and discriminator
  # calculate the derivate and update the weights to improve 
  generator_g_gradients = tape.gradient(total_gen_g_loss,generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
  discriminator_x_gradients = tape.gradient(disc_x_loss,discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
  
 #Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,generator_g.trainable_variables))
  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,discriminator_x.trainable_variables))
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,discriminator_y.trainable_variables))


# -

def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return -err


MAX0=11
MAX1=6


# Bayesian Optimization
#
# The Black_Box function parameters are different hyperparameters, and it returns the average difference between images

# +
def Black_Box(EPOCHS,LAMBDA,steps_per_epochs):
   
    
    for epoch in trange(int(EPOCHS),desc='epochs'):
        start = time.time()

        for _ in trange(int(steps_per_epochs), desc='steps_per_epochs'):
            i=np.random.randint(1,MAX0)
            j= np.random.randint(1,MAX1)

            image_x=np.load("/home/gm3g/project/S20-team6-project/encodedArray/{}_{}.npy".format(i,j))
            image_y=np.load("/home/gm3g/project/S20-team6-project/decodedArray/{}_{}.npy".format(i,j))
            train_step(LAMBDA,np.asarray([image_x/255.0], dtype='float32'), np.asarray([image_y/255.0], dtype='float32'))
       
        
    sum = 0.0
    for i in trange(1,MAX0):
        for j in trange(1,MAX1):
            image_x=np.load("/home/gm3g/project/S20-team6-project/encodedArray/{}_{}.npy".format(i,j))
            image_y=np.load("/home/gm3g/project/S20-team6-project/decodedArray/{}_{}.npy".format(i,j))

            sum += mse(generator_g.predict(np.asarray([image_x/255.0], dtype='float32')), np.asarray([image_y/255.0], dtype='float32'))
    avg = sum / ((MAX0-1)*(MAX1-1))

    return avg

    
     
# -

bounds={'EPOCHS':(200,500),
        'LAMBDA':(8,12),
       'steps_per_epochs':(10,50)
       }


optimizer = BayesianOptimization(
    f=Black_Box,
    pbounds=bounds,
    random_state=1
)

# +
logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# Results will be saved in ./logs.json
optimizer.maximize(init_points=2,n_iter=10)

# +
##This will load logs.json file

# from bayes_opt.util import load_logs
# load_logs(new_optimizer, logs=["./logs.json"])
# #optimizer.maximize(n_iter=10)



