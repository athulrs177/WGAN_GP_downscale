import tensorflow as tf
from tensorflow.keras import layers, models, Input


def downsample_block(x, n_filters, strides=2):
    x = layers.Conv2D(n_filters, 3, strides=strides, padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x, training=True)
    return x 

def upsample_block(x, n_filters, strides=2):
    x = layers.Conv2DTranspose(n_filters, 3, strides, padding="same",
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.Conv2D(n_filters, 3, 1, padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_generator(input_shape, z_dim):
    H, W, n_features = input_shape
    coarse_input = Input(shape=(H, W, n_features), name="coarse_input")
    z_input = Input(shape=(z_dim,), name="z_input")

    z_tiled = layers.Dense(H * W * z_dim)(z_input)
    z_tiled = layers.Reshape((H, W, z_dim))(z_tiled)
    x = layers.Concatenate(axis=-1)([coarse_input, z_tiled])
    x = layers.UpSampling2D(size=10)(x)

    x = downsample_block(x, 64)
    x = downsample_block(x, 128)
    x = downsample_block(x, 256)
    x = downsample_block(x, 512)
    x = downsample_block(x, 1024)
    x = downsample_block(x, 2048, strides=1)

    x = upsample_block(x, 1024)
    x = upsample_block(x, 512)
    x = upsample_block(x, 256)
    x = upsample_block(x, 128)
    x = upsample_block(x, 64)

    output = layers.Conv2D(1, 1, padding="same", activation="tanh")(x)

    return models.Model(inputs=[coarse_input, z_input], outputs=output, name="wGAN_generator_with_noise")