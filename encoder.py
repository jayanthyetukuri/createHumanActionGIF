from variables import batch_size, latent_dim, epsilon_std, img_shape
from keras.layers import Input, Flatten, LeakyReLU, Dense, Lambda, Conv2D, MaxPooling2D
from keras.models import Model
import keras.backend as K


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def build_encoder():
    img = Input(shape=img_shape)
    h = Dense(512)(img)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(latent_dim)(h)

    h = Conv2D(32, (2, 2), activation="relu")(h)
    h = Conv2D(32, (2, 2), activation="relu")(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Conv2D(64, (2, 2), activation="relu")(h)
    h = Conv2D(64, (2, 2), activation="relu")(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Conv2D(128, (2, 2), activation="relu")(h)
    h = Conv2D(128, (2, 2), activation="relu")(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Flatten()(h)

    h = Dense(512)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(256)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(512)(h)
    h = LeakyReLU(alpha=0.2)(h)
    mu = Dense(latent_dim)(h)
    log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mu, log_var])
    return Model(img, z)