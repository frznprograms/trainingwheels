from queue import Queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels

class Node:
	def __init__(self, feature = None, split_criteria = None, children = None, is_leaf = False, prediction = None):
		self.feature = feature
		self.split_criteria = split_criteria
		self.children = children if children else {}
		self.is_leaf = is_leaf
		self.prediction = prediction


class DecisionTreeClassifier(trainingwheels):
	def __init__(self, name = "Decision Tree Classifier", numeric_cols = []):
		self.name = name
		self.root = None
		self.depth = 0
		self.numeric_cols = numeric_cols
		self.majority_class = None

	def fit(self, X_train, y_train, max_depth = None, num_splits = 3):
		'''
		Builds a decision tree based on given X and y training data.

		Args:
			X_train: 2D numpy array of training data (categorical columns must be encoded).
			y_train: 1D numpy array of training data labels (should be encoded as labels are categorical).
			max_depth: maximum allowable depth of the tree.
			num_splits: number of times to split the data of continuous features to find best split condition.

		Output:
			No output. Updates decision tree initialised in model instance by updating `self.root`.
		'''
		self.root = Node()
		queue = Queue()
		queue.put((self.root, X_train, y_train)) # Node, subset data (X and y)
		self.depth += 1
		majority_class = self.get_majority_class(y_train)
		self.majority_class = majority_class

		while not queue.empty():
			node, X, y = queue.get()

			# stop conditions
			if len(np.unique(y)) == 1 or len(X) <= 1 or (max_depth and self.depth >= max_depth):
				# end of tree reached -> node is a leaf
				node.is_leaf = True 
				node.prediction = self.get_majority_class(y)
				continue

			best_attr, best_split = self.get_best_attr(X, y, num_splits = num_splits)

			# ensure valid best_attr
			if best_attr is None:
				node.is_leaf = True
				node.prediction = self.get_majority_class(y)
				continue

			# handle numeric features
			if best_attr in self.numeric_cols:
				left_mask = X[:, best_attr] <= best_split
				right_mask = X[:, best_attr] > best_split

				node.feature = best_attr
				node.split_criteria = best_split
				left_node, right_node = Node(), Node()

				node.children = {"<= threshold": left_node, "> threshold": right_node}
				queue.put((left_node, X[left_mask], y[left_mask]))
				queue.put((right_node, X[right_mask], y[right_mask]))
				self.depth += 1

			else: 
				unique_values = list(np.unique(X[:, best_attr]))
				node.feature = best_attr
				node.split_criteria = unique_values
				node.children = {}

				for value in unique_values:
					mask = X[:, best_attr] == value
					child_node = Node()
					node.children[value] = child_node
					queue.put((child_node, X[mask], y[mask]))
				self.depth += 1

		return

	
	def predict_sample(self, sample, node=None):
		'''
		Uses the decision tree rooted in `self.root` to predict on a given sample X.

		Args:
			sample: a sample of X data (1D numpy array), preferably test data or data not used in training
			node: optional specification for node to begin tree traversal; should be left as default for most cases..

		Output:
			Predicition for given 1D sample data.
		'''
		# Start from the root node if no node is specified
		if node is None:
			node = self.root

		# If this is a leaf node, return its prediction
		if node.is_leaf:
			return node.prediction

		# Determine the traversal path based on the split type
		split_criteria = node.split_criteria
		feature_value = float(sample[node.feature])  # Ensure scalar extraction

		if isinstance(split_criteria, list):
			if feature_value in split_criteria:
				child_index = split_criteria.index(feature_value)
				if child_index in node.children:
					return self.predict_sample(sample, node.children[child_index])
			# If feature_value is not in split_criteria, fallback to majority class
			return self.get_majority_class_node(node)

		else:  # Continuous split
			if feature_value <= split_criteria:
				return self.predict_sample(sample, node.children["<= threshold"])
			else:
				return self.predict_sample(sample, node.children["> threshold"])


	
	def predict(self, X_test):
		'''
		Predicts labels of multiple instances of data.

		Args:
			X_test: 2D numpy array of testing data (categorical columns must be encoded).

		Output:
			List of predicted labels.
		'''
		predictions = []
		for i, sample in enumerate(X_test):
			prediction = self.predict_sample(sample)
			predictions.append(prediction)
		return predictions



	def predict_score(self, X_test, y_test):
		'''
		Predicts labels of multiple instances of data.

		Args:
			X_test: 2D numpy array of testing data (categorical columns must be encoded).
			y_test: 1D numpy array of testing data labels (should be encoded as labels are categorical).

		Output:
			Accuracy score based on testing data.
		'''
		predictions = self.predict(X_test)
		score = 0
		for i in range(len(predictions)):
			if predictions[i] == y_test[i]:
				score += 1

		return score * 100 / len(predictions)


	### HELPER FUNCTIONS ###


	def print_tree(self, node=None, depth=0):
		'''
		Prints tree structure.

		Args:
			node: the node to begin tree traversal (and subsequent printing).
			depth: the depth of the tree. Set to 0 for layer index, set to 1 for layer number.

		Output:
			List of predicted labels.
		'''
		if node is None:
			return
			
		indent = " " * (depth * 4)
		if node.is_leaf:
			print(f"{indent}Leaf Node (Prediction: {node.prediction})")
		else:
			print(f"{indent}Node (Feature: {node.feature}, Split: {node.split_criteria})")
			if node.children:
				for key, child in node.children.items():
					print(f"{indent} Child with value {key}: ")
					self.print_tree(node = child, depth = depth + 1)		
		return


	def get_majority_class(self, y):
		'''
		Gets majority class of a particular subset of the data (subset can also be the entire data).

		Args:
			y: 1D numpy array of labels.

		Output:
			Majority class, an integer.
		'''
		y = y.astype(int)
		majority_class = np.bincount(y).argmax()
		self.majority_class = majority_class
		return self.majority_class


	def get_majority_class_node(self, node):
		"""
		Fallback to the majority class at the given node.
		"""
		if node.is_leaf:
			return node.prediction
		else:
			# Traverse the children to find the majority class
			counts = {}
			for child in node.children.values():
				if child.is_leaf:
					counts[child.prediction] = counts.get(child.prediction, 0) + 1
			if counts:
				return max(counts, key=counts.get)
			return None  # Fallback if no children exist

	
	def entropy(self, subset):
		'''
		Calculates entropy of a subset of data (subset can also be the whole data).

		Args:
			subset: 1D numpy array of labels.

		Output:
			List of predicted labels.
		'''
		values, counts = np.unique(subset, return_counts=True)
		probabilities = counts / len(subset)
		entropy = -np.sum(probabilities * np.log2(probabilities))
		return entropy

	
	def get_best_attr(self, X_train, y_train, num_splits = 3):
		'''
		Calcaulates the best attribute for the decision tree's next split by maximising information gain.

		Args:
			X_train: 2D numpy array of training data (categorical columns must be encoded).
			y_train: 1D numpy array of training data labels (should be encoded as labels are categorical).
			num_splits: number of times to split the data of continuous features to find best split condition.

		Output:
			The best attribute for the tree to split on and the best split for that feature, if the feature is continuous.
		'''
		best_attr, best_split, best_score = None, None, -np.inf

		for col in range(X_train.shape[1]):
			# handle categorical columns
			if col not in self.numeric_cols:
				unique_values = np.unique(X_train[:, col])
				information_gain = self.entropy(y_train) # start with whole dataset entropy

				for value in unique_values:
					subset_x = X_train[:, col] == value
					subset_y = y_train[subset_x]

					# calculate weighted entropy
					prop = len(subset_y) / len(y_train)
					# update to get overall information gain
					information_gain -= prop * self.entropy(subset_y)

					if information_gain > best_score:
						best_attr, best_score = col, information_gain
			
			# handle numeric columns
			else:
				# sort rows first before analysing splits
				unique_values = np.sort(np.unique(X_train[:, col]))
				if len(unique_values) > 3:
					split_points = np.linspace(unique_values[0], unique_values[-1], num = num_splits)[1: -1]
				else:
					split_points = unique_values

				for split in split_points:
					information_gain = self.entropy(y_train)
					lower_split = X_train[:, col] <= split
					upper_split = X_train[:, col] > split

					# just in case there are empty subsets
					if np.sum(lower_split) == 0 or np.sum(upper_split) == 0:
						continue

					information_gain -= (len(y_train[lower_split]) / len(y_train)) * self.entropy(y_train[lower_split])
					information_gain -= (len(y_train[upper_split]) / len(y_train)) * self.entropy(y_train[upper_split])

					if information_gain > best_score:
						best_attr, best_split, best_score = col, split, information_gain

		return best_attr, best_split

