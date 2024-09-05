{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import matplotlib.pyplot as plt\
\
def generate_and_save_images(model, epoch, test_input):\
    predictions = model(test_input, training=False)\
\
    fig = plt.figure(figsize=(4,4))\
\
    for i in range(predictions.shape[0]):\
        plt.subplot(4, 4, i+1)\
        plt.imshow((predictions[i] + 1) / 2)  # Scale images back to [0,1] range\
        plt.axis('off')\
\
    plt.savefig('image_at_epoch_\{:04d\}.png'.format(epoch))\
    plt.show()\
}