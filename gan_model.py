{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import tensorflow as tf\
from tensorflow.keras import layers\
\
# Generator Model\
def build_generator():\
    model = tf.keras.Sequential()\
    \
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))\
    model.add(layers.BatchNormalization())\
    model.add(layers.LeakyReLU())\
    \
    model.add(layers.Reshape((8, 8, 256)))  # Reshape noise into a 8x8 image\
    \
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))\
    model.add(layers.BatchNormalization())\
    model.add(layers.LeakyReLU())\
\
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))\
    model.add(layers.BatchNormalization())\
    model.add(layers.LeakyReLU())\
\
    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))\
    \
    return model\
\
# Discriminator Model\
def build_discriminator():\
    model = tf.keras.Sequential()\
\
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[64, 64, 3]))\
    model.add(layers.LeakyReLU())\
    model.add(layers.Dropout(0.3))\
\
    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))\
    model.add(layers.LeakyReLU())\
    model.add(layers.Dropout(0.3))\
\
    model.add(layers.Flatten())\
    model.add(layers.Dense(1))  # Single output for real/fake classification\
    \
    return model\
}