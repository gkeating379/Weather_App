import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Add, LayerNormalization, Dense, LSTM


def MultiHeadModel(seq_len, d_model, num_heads):
    '''Multi-headed Attention Model

    seq_len => int, number of timeseries elements
    d_model => int, number of dimensions in one record
    num_heads => number of heads for multihead

    Return
    model => tf model'''
    input = Input(shape=(seq_len, d_model), name='input')

    # MHA layer 1
    mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name='mha1')(input, input, input)
    add1 = Add(name='add1')([input, mha1])
    norm1 = LayerNormalization(name='norm1')(add1)

    # MHA layer 2
    mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name='mha2')(norm1, norm1, norm1)
    add2 = Add(name='add2')([input, mha2])
    norm2 = LayerNormalization(name='norm2')(add2)

    # MHA layer 3
    mha3 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name='mha3')(norm2, norm2, norm2)
    add3 = Add(name='add3')([input, mha3])
    norm3 = LayerNormalization(name='norm3')(add3)

    out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name='out')(norm3, norm3, norm3)

    model = tf.keras.models.Model(input, out)
    model.summary()

    return model


def LSTM_Model(seq_len, d_model):
    '''LSTM Model

    seq_len => int, number of timeseries elements
    d_model => int, number of dimensions in one record

    Return
    model => tf model'''
    input = Input(shape=(seq_len, d_model), name='input')

    lstm1 = LSTM(units=d_model, return_sequences=True, name='lstm1')(input)
    add1 = Add(name='add1')([input, lstm1])
    norm1 = LayerNormalization(name='norm1')(add1)

    lstm2 = LSTM(units=d_model, return_sequences=True, name='lstm2')(norm1)
    add2 = Add(name='add2')([input, lstm2])
    norm2 = LayerNormalization(name='norm2')(add2)

    lstm3 = LSTM(units=d_model, return_sequences=True, name='lstm3')(norm2)

    # pool = tf.keras.layers.GlobalAveragePooling1D()(norm)
    # flat = tf.keras.layers.Flatten(name='flatten')(lstm3)

    denorm = Dense(d_model, name='denorm')(lstm3)

    model = tf.keras.models.Model(input, denorm)
    model.summary()

    return model
