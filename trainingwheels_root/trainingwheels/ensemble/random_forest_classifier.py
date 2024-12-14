import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels


class RandomForestClassifier(trainingwheels):
	def __init__(self, name="Random Forest Classifier", numeric_cols=[]):
		self.name = name
		self.forest = []
		self.numeric_cols = numeric_cols

	def fit(self, X_train, y_train, n_estimators=100, prop=0.2, max_depth=None, num_splits=3):
		n_samples, n_features = X_train.shape
		
		for _ in range(n_estimators):
			# Bootstrap sampling of rows
			bootstrap_indices = np.random.choice(n_samples, size=int(prop * n_samples), replace=True)
			X_boot = X_train[bootstrap_indices]
			y_boot = y_train[bootstrap_indices]

			tree = ForestTree(numeric_cols=self.numeric_cols)
			tree.fit(X_boot, y_boot, max_depth=max_depth, num_splits=num_splits)
			self.forest.append(tree)  # Store tree and its feature subset

	
	def predict_sample(self, sample):
		sample_predictions = []
		for tree in self.forest:
			# Use only the features used by this tree
			prediction = tree.predict_sample(sample)
			sample_predictions.append(prediction)

		# Majority voting
		most_common = Counter(sample_predictions).most_common(1)
		return most_common[0][0]

	
	def predict(self, X_test):
		return [self.predict_sample(sample) for sample in X_test]

	
	def predict_score(self, X_test, y_test):
		predictions = self.predict(X_test)
		score = 0
		for i in range(len(predictions)):
			if predictions[i] == y_test[i]:
				score += 1

		return score * 100 / len(predictions)


class Node():
	def __init__(self, feature = None, split_criteria = None, children = None, is_leaf = False, prediction = None):
		self.feature = feature
		self.split_criteria = split_criteria
		self.children = children if children else {}
		self.is_leaf = is_leaf
		self.prediction = prediction


class ForestTree(RandomForestClassifier):
	def __init__(self, name = "A tree in the forest", numeric_cols = []):
		self.name = name
		self.root = None
		self.depth = 0
		self.numeric_cols = numeric_cols
		self.majority_class = None

	def fit(self, X_train, y_train, max_depth = None, num_splits = 3):
		self.root = Node()
		queue = Queue()
		queue.put((self.root, X_train, y_train)) # Node, subset data (X and y)
		self.depth += 1
		majority_class = self.get_majority_class(y_train)
		self.majority_class = majority_class
		n_features = X_train.shape[1]
		max_features = n_features // 2 # as a general rule, use half of features

		while not queue.empty():
			node, X, y = queue.get()

			# stop conditions
			if len(np.unique(y)) == 1 or len(X) <= 1 or (max_depth and self.depth >= max_depth):
				# end of tree reached -> node is a leaf
				node.is_leaf = True
				node.prediction = self.get_majority_class(y)
				continue

			#best_attr, best_split = self.get_best_attr(X, y, num_splits = num_splits)

			# Randomly select a subset of features
			selected_features = np.random.choice(n_features, size=max_features, replace=False)

			# Separate numeric and categorical features in the selected subset
			selected_numeric_cols = [col for col in selected_features if col in self.numeric_cols]

			# Find the best attribute and split based on the selected features
			best_attr, best_split = self.get_best_attr(X[:, selected_features], y, num_splits=num_splits)

			# ensure valid best_attr
			if best_attr is None:
				node.is_leaf = True
				node.prediction = self.get_majority_class(y)
				continue

			# Adjust the index of `best_attr` to match the original data
			best_attr = selected_features[best_attr]

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
					#print(f"Splitting: X shape {X.shape}, mask shape {mask.shape}, best_attr {best_attr}, value {value}")
	
					if mask.ndim != 1:
						raise ValueError("Mask has an invalid number of dimensions.")

					if X.ndim != 2:
						raise ValueError("X is not a 2-dimensional array.")
					
					child_node = Node()
					node.children[value] = child_node
					queue.put((child_node, X[mask], y[mask]))
				self.depth += 1

		return


	def predict_sample(self, sample, node=None):
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
		predictions = []
		for i, sample in enumerate(X_test):
			prediction = self.predict_sample(sample)
			predictions.append(prediction)
		return predictions



	def predict_score(self, X_test, y_test):
		predictions = self.predict(X_test)
		score = 0
		for i in range(len(predictions)):
			if predictions[i] == y_test[i]:
				score += 1

		return score * 100 / len(predictions)


	def print_tree(self, node, depth=0):
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
		values, counts = np.unique(subset, return_counts=True)
		probabilities = counts / len(subset)
		entropy = -np.sum(probabilities * np.log2(probabilities))
		return entropy

	def get_best_attr(self, X_train, y_train, num_splits = 3):
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

