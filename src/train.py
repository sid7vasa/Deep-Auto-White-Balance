# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:56:48 2022

@author: santo
"""
import os
import argparse
import logging
import yaml

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from models.unet import Generator, doubleConvBlock, downBlock, DeepWBnet
from dataset.utils import generate_tfrecords, load_tfrecords

print(tf.test.is_gpu_available())

def get_dataset(parameters):
    """
    Loads TF records (If not exists, creates TF Records).
    Preprocesses datset to the needed format.
    returns train and validation dataset tf.data instances.
    Parameters
    ----------
    parameters : parameters loaded from the parameters.yaml file.
        DESCRIPTION.
    Returns
    -------
    train_dataset : tf.data train dataset instance
        DESCRIPTION.
    val_dataset : tf.data validation dataset instance
        DESCRIPTION.
    """
    if not os.path.exists(os.path.join(
            parameters['dataset']['data_dir']['train'],
            'train.tfrecords')) or not os.path.exists(os.path.join(
                parameters['dataset']['data_dir']['val'], 'val.tfrecords')):
        print("Generating TF Records:")
        generate_tfrecords(parameters)
    else:
        print("Using existing TF Records")
    train_dataset, val_dataset = load_tfrecords(parameters)
    train_dataset = train_dataset.batch(
        parameters['dataset']['batch_size']).shuffle(buffer_size=100)
    val_dataset = val_dataset.batch(parameters['dataset']['batch_size'])
    return train_dataset, val_dataset

def visualize_datasets(train_dataset, val_dataset):
    """
    Visualize an example in the dataset by reversing the preprocessing steps.
    using matplotlib.
    Parameters
    ----------
    train_dataset : tf.data training instance
        DESCRIPTION.
    val_dataset : tf.data validation instance
        DESCRIPTION.
    Returns
    -------
    None.
    """
    for data in train_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
        picture = data[1].numpy()[0]
        picture = (picture*127.5) + 127.5
        picture = np.array(picture, dtype=np.uint8)
        plt.imshow(picture)
        plt.show()


if __name__ == "__main__":
    with open('../parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)
    
    # Creating/Loading TF Records data
    train_dataset, val_dataset = get_dataset(parameters)
    
    visualize_datasets(train_dataset, val_dataset)
    
    net = DeepWBnet(input_shape=(512, 512, 3)).get_model()
    print(net.summary())
    plot_model(net, "unet.png")
