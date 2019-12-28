import numpy as np
from numpy import linalg as LA
from scipy import stats
from tqdm import tqdm, trange
from utility import *

class knn_classifier(model_abs):

    def __init__(self, k=3, p=2):
        """
        KNN Classifier constructor
        
        Keyword Arguments:
            k {int} -- Closest k neighbor used for classification (default: {3})
            p {int, numpy.inf} -- Parameter for LP distance, 1 for abs, 2 for RMS, inf for max (default: {2})
        """        
        self.k = k
        self.p = p
        self.kd_root = kd_tree()
        self.loss_vec = np.array([0])
        self.accuracy = 0
        self.label_rectifier = lambda x : x

    def config(self, k=3, p=2):
        """
        KNN Classifier configuration
        
        Keyword Arguments:
            k {int} -- Closest k neighbor used for classification (default: {3})
            p {int, numpy.inf} -- Parameter for LP distance, 1 for abs, 2 for RMS, inf for max (default: {2})
        """
        self.k = k
        self.p = p
    
    def train(self, train_data, train_label):
        """
        Training method: kd-tree construction, does not support training from different dataset
        
        Arguments:
            train_data {numpy.ndarray} -- training data with each instance at each row, feature space represented by columns
            train_label {numpy.ndarray} -- training labels
        """

        # TODO Add support for continuous training
        # Clear root
        self.kd_root = kd_tree()

        # Concatenate data and label
        self.ncols = train_data.shape[1]
        data = np.concatenate((train_data, train_label[:, np.newaxis]), axis=1)
        
        # Construct the k row matrix holder for prediction
        # krows: data (k), label, distance to data point
        self.krows = np.zeros((self.k, self.ncols + 2))
        self.krows[:, -1] = np.inf
        self._partition(data, self.kd_root)
    
    def _partition(self, data, curr_node, depth=0):
        """
        Partition data matrix by median value on each axis (axis = depth mod k)
        
        Arguments:
            data {numpy.ndarray} -- Concatenation of data and label
            curr_node {kd_tree} -- kd tree node
        
        Keyword Arguments:
            depth {int} -- Depth of current tree node (default: {0})
        """

        # Base case
        if data.size == 0:
            return
        else:
            # Recursive case
            nrows = data.shape[0]
            axis = depth % self.ncols  # Axis to used for partition
            axis_median = np.percentile(data, 50, axis=0, interpolation="nearest")[axis]  # Acquire the value cloest to median
            median_inds = np.where(data[:, axis] == axis_median)  # Search on the axis for median value
            lower_ind = np.where(data[:, axis] < axis_median)
            higher_ind = np.where(data[:, axis] > axis_median)

            # Access the median row over the axis and the lower and higher partitions
            elements = data[median_inds]
            median_row = elements[0, :]  # First row, and add the following to the lower part
            lower_rows = np.concatenate((data[lower_ind], elements[1:, :]))
            higher_rows = data[higher_ind]

            # Place the node and enter the partition process
            curr_node.value = median_row
            curr_node.left = kd_tree(curr_node)
            curr_node.right = kd_tree(curr_node)
            curr_node.depth = depth
            
            self._partition(lower_rows, curr_node.left, depth + 1)
            self._partition(higher_rows, curr_node.right, depth + 1)
    
    def predict(self, data):
        """
        Prediction method: kd-tree kth search algorithm, search the kth nearest instances in dataset relatively towards data
        then use majority vote to determine the class
        
        Arguments:
            data {numpy.ndarray} -- data used for prediction, single row from the test data or training data
        """        

        # Clear krows martix
        self.krows = np.zeros(self.krows.shape)
        self.krows[:, -1] = np.inf
        
        # Find the leaf node for data
        leaf_node = self.kd_root
        depth = 0
        while(isinstance(leaf_node.value, np.ndarray)):
            axis = depth % self.ncols
            pivot_value = leaf_node.value[axis]
            if data[axis] > pivot_value:
                leaf_node = leaf_node.right
            else:
                leaf_node = leaf_node.left
        
        # Find the kth neighbors 
        self._search_up(data, leaf_node.parent, leaf_node)
        k_labels = self.krows[:, -2]
        prediction = stats.mode(k_labels).mode[0]
        return prediction

    # Search up to root and search down to leaf
    def _search_up(self, data, curr_node, prev_child_node):
        """
        Search the kd tree upward
        
        Arguments:
            data {numpy.ndarray} -- Data used for evaluation, row vector
            curr_node {kd_tree} -- Current node used for finding distance
            prev_child_node {kd_tree} -- Previous node, should be curr_node.left or curr_node.right
        """        
        # Search kd tree for k closest neighbors
        # krows is sorted descending
        # print("[U] Searching up\tdepth: {}".format(curr_node.depth))
        if curr_node == self.kd_root:
            return
        else:
            # Calculate current distance
            dist = LA.norm(curr_node.value[:-1] - data, ord=self.p)
            row = np.append(curr_node.value, dist)

            # Check for if the current node can be counted as kth closest
            # If the current row is closer to the data
            if self.krows[-1, -1] > dist:
                # Insert the current row
                self.krows[-1, :] = row
                self.krows = self.krows[self.krows[:, -1].argsort()]  # Sort by distance
            
            # Search for child nodes
            # TODO Check for overlapping here?
            child_node = curr_node.left if curr_node.left != prev_child_node else curr_node.right
            self._search_down(data, child_node)

            # Search upward
            self._search_up(data, curr_node.parent, curr_node)

    
    def _search_down(self, data, curr_node):
        """
        Search the kd tree downward
        
        Arguments:
            data {numpy.ndarray} -- Data used for evaluation, row vector
            curr_node {kd_tree} -- Node should be focused to be searched
        """        
        # Search down for child node
        # Check for the depth axis and whether with the largest dist in 
        # krows the area is overlapped with the supersphere

        if not isinstance(curr_node.value, np.ndarray):
            # At the bottom of the tree
            return
        else:
            # print("[D] Searching down\tdepth: {}".format(curr_node.depth))
            root_node = curr_node.parent
            axis = root_node.depth % self.ncols
            data_value = data[axis]
            curr_value = curr_node.value[axis]
            median = root_node.value[axis]
            radius = self.krows[-1, -1]

            # Check for current node distance
            dist = LA.norm(curr_node.value[:-1] - data, ord=self.p)
            row = np.append(curr_node.value, dist)

            # Check for if the current node can be counted as kth closest
            # If the current row is closer to the data
            if self.krows[-1, -1] > dist:
                # Insert the current row
                self.krows[-1, :] = row
                self.krows = self.krows[self.krows[:, -1].argsort()]  # Sort by distance

            # Check for overlapping
            # Child Node is on the right of the root node and overlapped with supersphere
            # or on the left of thee root node and overlapped with the supersphere
            # Perform search on the child node left and right node
            if curr_value > median and data_value + radius >= median:
                self._search_down(data, curr_node.left)
                self._search_down(data, curr_node.right)
            elif curr_value <= median and data_value + radius <= median:
                self._search_down(data, curr_node.left)
                self._search_down(data, curr_node.right)

class kd_tree:
    def __init__(self, parent=None, depth=0):
        self.parent = parent
        self.value = None
        self.left = None
        self.right = None
        self.depth = depth