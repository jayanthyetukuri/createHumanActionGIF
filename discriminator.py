from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Input, Reshape, Conv2D, MaxPooling2D, Flatten
from variables import latent_dim

def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1, activation="sigmoid"))
    print 'discriminator summary'
    model.summary()
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)