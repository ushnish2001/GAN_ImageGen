{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 EPOCHS = 10000\
NOISE_DIM = 100\
NUM_EXAMPLES_TO_GENERATE = 16\
\
# Seed for consistent image generation\
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])\
\
# Training loop\
@tf.function\
def train_step(images):\
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])\
\
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:\
        generated_images = generator(noise, training=True)\
\
        real_output = discriminator(images, training=True)\
        fake_output = discriminator(generated_images, training=True)\
\
        gen_loss = generator_loss(fake_output)\
        disc_loss = discriminator_loss(real_output, fake_output)\
\
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)\
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)\
\
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))\
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))\
\
def train(dataset, epochs):\
    for epoch in range(epochs):\
        for image_batch in dataset:\
            train_step(image_batch)\
\
        # Produce images after every epoch\
        generate_and_save_images(generator, epoch + 1, seed)\
}