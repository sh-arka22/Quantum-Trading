import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, K.expand_dims(self.W)))
        a = K.softmax(e, axis=1)
        a = K.expand_dims(a)
        return K.sum(x * a, axis=1)

def build_attention_lstm_classifier(input_shape, num_classes=2):
    inp = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inp)
    att = Attention()(lstm_out)
    drop = Dropout(0.5)(att)
    dense = Dense(32, activation="relu")(drop)
    out = Dense(num_classes, activation="softmax")(dense)
    model = Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
