import tensorflow as tf


def init_lstm(embed_dim: int, nlstm: int) -> tf.keras.Model:
    '''
    Initialises a LSTM model with the specified number of layers and embedding dimension.

    Layers and functions used:
      dropout layers, batch normalization, tanh activation function,
      softmax activation function, fully connected layer

    tanh is used as the activation function for the LSTM layers as it is the default
    '''
    modelIn = tf.keras.layers.Input(shape=(256, ))
    x = tf.keras.layers.Embedding(100000, embed_dim, input_length=256, mask_zero=True)(modelIn)
    x = tf.keras.layers.SpatialDropout1D(rate=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for _ in range(nlstm-1):
        x = tf.keras.layers.LSTM(units=30, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(units=30)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    modelOut = tf.keras.layers.Dense(2187, activation='softmax')(x)
    model = tf.keras.Model(inputs=modelIn, outputs=modelOut)

    model.summary()

    return model


def load_pretrained(checkpoint_path: str) -> tf.keras.Model:
    '''
    Nice wrapper for loading pretrained models
    '''
    return tf.keras.models.load_model(checkpoint_path)


if __name__ == '__main__':
    
    model = init_lstm(embed_dim=150, nlstm=1)
    model.summary()
