import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sympy import Matrix,lambdify,eye,symbols
import matplotlib.pyplot as plt
import warnings

from physics_property.property import Property
from physics_property.band.band_data_in import BandDataIn
from physics_property.band.band_data_out import BandDataOut
from parameter.para4band.para4band import Para4Band


class Band(Property):
    def __init__(self):
        self.description = "the class used to do band data analysis"
        
        
    def init_calculate_model(self, model_path):
        super().init_calculate_model(model_path)
        self.matrix_function = Para4Band(self.model_path)
        

