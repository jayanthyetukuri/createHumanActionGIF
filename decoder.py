from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, LeakyReLU, Reshape, Input, Conv2D, Flatten, MaxPooling2D
from variables import img_shape, latent_dim, savedModelsDir
import numpy as np

def build_decoder():
    try:
        loaded_decoder = model_from_json(open(savedModelsDir + 'decoder_model.json').read())
        loaded_decoder.load_weights(savedModelsDir + 'decoder_model.h5')
        print 'Decoder model loaded from file...'
        return loaded_decoder
    except:
        model = Sequential()
        model.add(Dense(512, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(img_shape), activation='tanh'))
        model.add(Reshape(img_shape))

        model.add(Conv2D(32, (2, 2), activation="relu"))
        model.add(Conv2D(32, (2, 2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (2, 2), activation="relu"))
        model.add(Conv2D(64, (2, 2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (2, 2), activation="relu"))
        model.add(Conv2D(128, (2, 2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        # tanh is more robust: gradient not equal to 0 around 0
        model.add(Dense(np.prod(img_shape), activation='tanh'))
        model.add(Reshape(img_shape))
        print 'decoder summary'
        model.summary()
        z = Input(shape=(latent_dim,))
        img = model(z)
        return Model(z, img)