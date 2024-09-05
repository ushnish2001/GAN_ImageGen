{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)\
\
def generator_loss(fake_output):\
    return cross_entropy(tf.ones_like(fake_output), fake_output)\
\
def discriminator_loss(real_output, fake_output):\
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)\
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)\
    return real_loss + fake_loss\
\
# Optimizers\
generator_optimizer = tf.keras.optimizers.Adam(1e-4)\
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)\
}