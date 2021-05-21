from tensorflow import keras
from foolbox import zoo
import numpy as np
import math
from Model.layers_for_densenet import NormalizingLayer01

class TargetModel:
    def __init__(self):
        self.model = None
        self.framework = ''

    def load_model(self, model_name, dataset_shortname='mnist'):
        if model_name == 'inception':
            self.model = keras.applications.inception_v3.InceptionV3(weights=dataset_shortname)
            self.framework = 'keras'
        elif model_name == 'resnet':
            self.model = keras.applications.resnet50.ResNet50(weights=dataset_shortname)
            self.framework = 'keras'
        elif model_name == 'vgg':
            self.model = keras.applications.vgg16.VGG16(weights=dataset_shortname)
            self.framework = 'keras'
        elif model_name == 'abs' :
            url = 'https://github.com/bethgelab/AnalysisBySynthesis'
            self.model = zoo.get_model(url)
            self.framework = 'pytorch'
        elif model_name.endswith(".h5") :
            self.framework = 'keras'
            if "DenseNet_k60_L16_norm" in model_name :
                self.model = keras.models.load_model(model_name, compile=False, custom_objects={'NormalizingLayer01': NormalizingLayer01})
            else :
                self.model = keras.models.load_model(model_name,compile=False)
