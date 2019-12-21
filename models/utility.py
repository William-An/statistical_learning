import numpy as np
class model_abs:
    """
    Abstract class for statistical learning model
    """

    def __init__(self):
        self.loss_vec = np.array([0])
        self.accuracy = 0

    def config(self):
        # Config parameters for model
        raise NotImplementedError
    
    def train(self, data, label):
        raise NotImplementedError

    def get_loss(self):
        return np.average(self.loss_vec)

    def evaluate(self, data, label):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
