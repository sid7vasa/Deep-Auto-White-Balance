# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:56:48 2022

@author: santo
"""
import os
import argparse
import logging

import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

from models.unet import Generator, doubleConvBlock, downBlock, DeepWBnet

print(tf.test.is_gpu_available())


if __name__ == "__main__":
    net = DeepWBnet(input_shape=(512, 512, 3)).get_model()
    print(net.summary())
    plot_model(net, "unet.png")
