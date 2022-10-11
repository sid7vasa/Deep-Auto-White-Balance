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

from models.unet import Generator, doubleConvBlock

print(tf.test.is_gpu_available())


if __name__ =="__main__":
    net = Generator(input_shape=(512,512,3)).get_model()
    print(net.summary())
    db_conv  = downBlock(128)
    db_conv.build(input_shape=(1,512,512,3))
    print(db_conv.summary())
    
    
