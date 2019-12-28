import numpy as np
from tqdm import tqdm, trange
class model_abs:
    """
    Abstract class for statistical learning model
    """

    def __init__(self):
        self.loss_vec = np.array([0])
        self.accuracy = 0
        self.label_rectifier = lambda x : x

    def config(self):
        # Config parameters for model
        raise NotImplementedError
    
    def train(self, train_data, train_label):
        raise NotImplementedError

    def get_loss(self):
        return np.average(self.loss_vec)

    def evaluate(self, test_data, test_label):
        size = test_data.shape[0]
        incorrect = 0
        test_label = self.label_rectifier(test_label)

        # TODO Rewrite in matrix / vector ops
        for i in tqdm(range(size)):
            prediction = self.predict(test_data[i, :])

            # Correct target label
            label = test_label[i]
            if prediction != label:
                incorrect += 1
        correct = size - incorrect
        print("\nAccuracy: {:.2f}%\tCorrect: {}\tIncorrect: {}".format((correct / size) * 100, correct, incorrect))
        self.accuracy = correct / size
    
    def predict(self, data):
        raise NotImplementedError
