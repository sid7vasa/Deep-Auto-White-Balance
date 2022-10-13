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


def plot_sample_outputs(net, dataset, val=False):
    """
    Takes random examples from the input tf.data instance and then plots the 
    generated output, corresponding inputs and ground truths.
    Parameters
    ----------
    dataset : tf.data validation instance
        DESCRIPTION.
    val : is validation dataset instance
    Returns
    -------
    img : TYPE
        DESCRIPTION.
    """
    _, axs = plt.subplots(1, 3, figsize=(8, 24))
    axs = axs.flatten()

    def un_normalize(img):
        img = (img * 127.5) + 127.5
        img = np.array(img, dtype=np.uint8)[0]
        return img
    if val:
        dataset = dataset.shuffle(buffer_size=100)

    for data in dataset.take(1):
        x_fake = un_normalize(net(data[0]))
        x_real_a = un_normalize(data[0])
        x_real_b = un_normalize(data[1])
        imgs = [x_real_a, x_real_b, x_fake]
        for ax, img in zip(axs, imgs):
            ax.imshow(img)
        plt.show()


if __name__ == "__main__":
    with open('../parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    # Creating/Loading TF Records data
    train_dataset, val_dataset = get_dataset(parameters)

    visualize_datasets(train_dataset, val_dataset)

    net = DeepWBnet(input_shape=(256, 256, 3)).get_model()

    optimizer = tf.keras.optimizers.Adam(
        lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    net.compile(optimizer=optimizer, loss=tf.keras.losses.mean_absolute_error,
                metrics=tf.keras.metrics.mean_absolute_error)

    history = net.fit(train_dataset,
                      batch_size=8,
                        epochs=100,
                        validation_data=val_dataset,
                        val_batch_size=8)

    print(net.summary())
    plot_sample_outputs(net, train_dataset, val=True)
    plot_model(net, "unet.png")
