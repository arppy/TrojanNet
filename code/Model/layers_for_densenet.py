from tensorflow import keras
from tensorflow.keras import backend as K


class NormalizingLayer01(keras.layers.Layer):

    def __init__(self, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.mean = K.constant([0., 0., 0.], dtype=K.floatx())
        self.std = K.constant([255., 255., 255.], dtype=K.floatx())

    def call(self, inputs, **kwargs):
        out = (inputs - self.mean) / self.std
        return out
		
def load_model(model_path):
    return keras.models.load_model(model_path,
                                   custom_objects={'NormalizingLayer01': NormalizingLayer01
                                                   })		