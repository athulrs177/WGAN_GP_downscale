import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

def conv_block(x, filters, kernel_size=(3, 3), name='', strides=(2, 2),
               use_norm=True, use_dropout=False, weight_decay=1e-3):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
               kernel_regularizer=l2(weight_decay) if use_norm else None,
               name=name+'_conv')(x)
    x = LeakyReLU(negative_slope=0.2, name=name+'_lrelu')(x)
    if use_norm:
        x = BatchNormalization(name=name+'_bn')(x)
    if use_dropout:
        x = Dropout(0.3, name=name+'_dropout')(x, training=True)
    return x

def build_discriminator(input_shape):
    inputs = Input(shape=input_shape, name='discriminator_input')

    x = conv_block(inputs, 32, name='block1')
    x = conv_block(x, 64, name='block2')
    x = conv_block(x, 128, name='block3')
    x = conv_block(x, 256, name='block4')
    x = conv_block(x, 512, name='block5')

    shared_features = GlobalAveragePooling2D()(x)

    score_map = Conv2D(1, (3, 3), padding='same',
                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                       name='final_conv')(x)

    extreme_prob = Dense(1, activation='sigmoid', name='extreme_classifier')(shared_features)

    return Model(inputs, [score_map, extreme_prob], name='wGAN_discriminator_dual_head')
