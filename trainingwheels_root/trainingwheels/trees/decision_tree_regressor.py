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


class DecisionTreeRegressor(trainingwheels):
	def __init__(self, name = "Decision Tree Classifier", numeric_cols = []):
		self.name = name
		self.root = None
		self.depth = 0
		self.numeric_cols = numeric_cols
		self.majority_class = None
		self.default_prediction = None


	def fit(self, X_train, y_train, max_depth = None, num_splits = 3):
		self.root = Node()
		queue = Queue()
		queue.put((self.root, X_train, y_train)) # Node, subset data (X and y)
		self.depth += 1
		self.default_prediction = np.mean(y_train)

		while not queue.empty():
			node, X, y = queue.get()

			# stop conditions
			if len(np.unique(y)) == 1 or len(X) <= 1 or (max_depth and self.depth >= max_depth):
				# end of tree reached -> node is a leaf
				node.is_leaf = True
				node.prediction = np.mean(y)
				continue

			best_attr, best_split = self.get_best_attr(X, y, num_splits = num_splits)
			#print(best_attr)

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
		if node is None:
			node = self.root

		if node.is_leaf:
			return node.prediction

		feature_value = sample[node.feature]  # Check if this line extracts correctly

		if isinstance(node.split_criteria, list):  # Categorical split
			if feature_value in node.split_criteria:
				child_node = node.children.get(feature_value)
				if child_node:
					return self.predict_sample(sample, child_node)
			return self.default_prediction

		else:
			if feature_value <= node.split_criteria:
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
		predictions = np.array(predictions)

		rss = np.sum((predictions - y_test) ** 2)
		tss = np.sum((y_test - np.mean(y_test)) ** 2)
		r2_score = 1 - (rss / tss)

		return r2_score



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



	def rss(self, subset):
		avg = np.mean(subset)
		return np.sum((subset - avg) ** 2)



	def get_best_attr(self, X_train, y_train, num_splits = 3):
		best_attr, best_split, best_score = None, None, np.inf

		for col in range(X_train.shape[1]):
			#print(f"Evaluating column: {col}, type: {type(col)}")
			# handle categorical columns
			if col not in self.numeric_cols:
				rss = 0
				unique_values = np.unique(X_train[:, col])
				for value in unique_values:
					subset_x = X_train[:, col] == value
					subset_y = y_train[subset_x]

					error = self.rss(subset_y)
					rss += error

					if rss < best_score:
						#print(f"Updating best_attr: {col} (type: {type(col)})")
						best_attr, best_score = col, rss

			# handle numeric case
			else:
				unique_values = np.sort(np.unique(X_train[:, col]))
				if len(unique_values) > 3:
					split_points = np.linspace(unique_values[0], unique_values[-1], num = num_splits)[1: -1]
				else:
					split_points = unique_values

				for split in split_points:
					rss = 0
					lower_split = X_train[:, col] <= split
					upper_split = X_train[:, col] > split

					# just in case there are empty subsets
					if np.sum(lower_split) == 0 or np.sum(upper_split) == 0:
						continue

					rss += self.rss(y_train[lower_split]) + self.rss(y_train[upper_split])

					if rss < best_score:
						#print(f"Updating best_attr: {col} (type: {type(col)})")
						best_attr, best_split, best_score = col, split, rss

		return best_attr, best_split







