import cv2, os, numpy as np

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from encoder import build_encoder
from decoder import build_decoder
from discriminator import build_discriminator

from variables import *
from generateImages import imagesFromVideos
from saveData import saveModel, saveModelBkp, saveimages, generateImages, createVideo

optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# Build the encoder / decoder
encoder = build_encoder()
print "encoder.summary()"
print encoder.summary()
decoder = build_decoder()
print "decoder.summary()"
print decoder.summary()

img = Input(shape=img_shape)
# The generator takes the image, encodes it and reconstructs it
# from the encoding
encoded_repr = encoder(img)
reconstructed_img = decoder(encoded_repr)

# For the adversarial_autoencoder model we will only train the generator
# if discriminator is attached to generator, set this flag to fix discriminator
discriminator.trainable = False

# The discriminator determines validity of the encoding
validity = discriminator(encoded_repr)

# The adversarial_autoencoder model  (stacked generator and discriminator)
adversarial_autoencoder = Model(img, [reconstructed_img, validity])
adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)


def train(epochs, batch_size=128, sample_interval=50):
    # Load the dataset

    tempArray = []
    for entry in os.listdir(sourceImagesDir):
        try:
            tempArray.append(np.array(cv2.cvtColor(cv2.imread(sourceImagesDir + entry), cv2.COLOR_BGR2GRAY)))
        except:
            continue

    X_train = np.array(tempArray)

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        try:

            #  Train Discriminator

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, latent_dim))

            # Train the discriminator
            # let latent_real's output is close to 1
            d_loss_real = discriminator.train_on_batch(latent_real, valid)
            # let latent_fake's output is close to 0
            d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            # decrease reconstruction error and let discriminator's output is close to 1
            g_loss = adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])
            adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

        except:
            saveModel(decoder, discriminator)

        if epoch % sample_interval == 0:
            print "epoch: ", epoch
            # Plot the progress
            # print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
            #     epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))
            # save generated image samples
            saveimages(epoch, decoder)
            saveModel(decoder, discriminator)


# 32000 : walking
# 1200 : waving

epochs = 6000
sample_interval = 30
sample_count = epochs / sample_interval

# imagesFromVideos()
train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
saveModelBkp(decoder, discriminator)

# for i in range(20):
#     generateImages(i, decoder)

# createVideo()

'''                           
References:

https://machinelearningmastery.com/save-load-keras-deep-learning-models/

'''