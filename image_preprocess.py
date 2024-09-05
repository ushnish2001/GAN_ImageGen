{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import tensorflow as tf\
\
# Load and preprocess the CIFAR-10 dataset\
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()\
\
# Normalize images to [-1, 1] range\
train_images = (train_images - 127.5) / 127.5\
\
# Resize images if necessary\
train_images = tf.image.resize(train_images, [64, 64])\
\
# Batch and shuffle data\
BUFFER_SIZE = 60000\
BATCH_SIZE = 128\
\
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)\
}