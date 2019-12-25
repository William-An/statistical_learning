import numpy as np
from tqdm import tqdm, trange
from utility import *

class perceptron(model_abs):

    def __init__(self, single_sample_data, target_label = 1, lr = 0.005):
        super().__init__()
        self.learning_rate = lr
        self.weight = np.random.rand(single_sample_data.size)
        self.bias = np.random.rand(1)
        self.target_label = target_label
        rectifier = lambda label : 1 if label == self.target_label else -1
        self.label_rectifier = np.vectorize(rectifier)

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
        test_label = self.label_rectifier(test_label)

        # TODO Rewrite in matrix / vector ops
        for i in tqdm(range(size)):
            prediction = self.predict(test_data[i, :])

            # Correct target label
            label = test_label[i]
            if prediction * label <= 0:
                incorrect += 1
        correct = size - incorrect
        print("\nAccuracy: {:.2f}%\tCorrect: {}\tIncorrect: {}".format((correct / size) * 100, correct, incorrect))

    def predict(self, data):
        return np.sign(np.matmul(self.weight, data.T) + self.bias)

class perceptron_dual(perceptron):
    """
    Dual Representation of perceptron
    """
    def train(self, train_data, train_label):
        # Rectify label
        train_label = self.label_rectifier(train_label)

        # Create gram matrix for easy calculation
        self.gram = np.inner(train_data, train_data)

        # Initial alpha values if not exists
        if not hasattr(self, "alpha"):
            self.alpha = np.zeros((1, np.size(train_data, 0)))

        # loop through train data
        for i in trange(np.size(train_data, 0), desc = "Steps"):
            data = train_data[i]
            label = train_label[i]

            # Calculate result
            coeffs = self.alpha.T * train_label[:, np.newaxis]
            slice = self.gram[:, i]  # i x j column
            res = np.matmul(coeffs.T, slice[:, np.newaxis]) + self.bias

            # Update alphas and bias
            if label * res <= 0:
                self.alpha[0, i] += self.learning_rate
                self.bias += self.learning_rate * label
                
                # Update loss
                loss = -label * res
                self.loss_vec = np.append(self.loss_vec, loss)

        # Calculate weight based on alphas
        # Sum over the row to get the result from each components of x vec
        self.weight = np.sum(train_data * (self.alpha.T * train_label[:, np.newaxis]), axis = 0)

if __name__ == "__main__":
    model = perceptron(np.array([1, 1, 1]))