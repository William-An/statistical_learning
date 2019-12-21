import numpy as np
from tqdm import tqdm
from utility import *

class perceptron(model_abs):

    def __init__(self, single_sample_data, target_label = 1, lr = 0.005):
        super().__init__()
        self.learning_rate = lr
        self.weight = np.random.rand(single_sample_data.size)
        self.bias = np.random.rand(1)
        self.target_label = target_label

    def config(self, lr):
        self.learning_rate = lr
    
    def train(self, data, label):
        # Rectify label
        label = 1 if label == self.target_label else -1

        res = np.matmul(self.weight, data.T) + self.bias
        prediction = np.sign(res)

        # Whether misclassified
        if prediction * label <= 0:
            loss = -label * res
            self.loss_vec = np.append(self.loss_vec, loss)

            # Update via gradient
            self.weight = self.weight + self.learning_rate * label * data
            self.bias = self.bias + self.learning_rate * label
        else:
            self.loss_vec = np.append(self.loss_vec, 0)

    def evaluate(self, test_data, test_label):
        size = test_data.shape[0]
        incorrect = 0
        for i in tqdm(range(size)):
            prediction = self.predict(test_data[i, :])

            # Correct target label
            label = test_label[i]
            label = 1 if label == self.target_label else -1
            if prediction * label <= 0:
                incorrect += 1
        correct = size - incorrect
        print("\nAccuracy: {:.2f}%\tCorrect: {}\tIncorrect: {}".format((correct / size) * 100, correct, incorrect))
                
            

    
    def predict(self, data):
        return np.sign(np.matmul(self.weight, data.T) + self.bias)

if __name__ == "__main__":
    model = perceptron(np.array([1, 1, 1]))