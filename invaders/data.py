import numpy as np

class DataPreProcessing():
    """
    docstring
    """

    def __init__(self, **kwargs):
        self.y_min = kwargs['y_min']
        self.y_max = kwargs['y_max']
        self.x_min = kwargs['x_min']
        self.x_max = kwargs['x_max']
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)
