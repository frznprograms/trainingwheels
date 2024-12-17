import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels


class LinearRegression(trainingwheels):
	def __init__(self, name="Linear Regression Algorithm", bias=1):
		self.name = name
		self.coeffs_ = None
		self.bias = bias

	@staticmethod
	def rss(actual, predicted):
		s = np.sum(np.subtract(actual - predicted) ** 2)
		return s

	@staticmethod
	def mse(actual, predicted):
		return np.mean((predicted - results) ** 2)

	def __repr__(self):
		return f"LinearRegression(coefficients={self.coeffs_}, bias={self.bias})"

	def fit(self, X_train, y_train, alpha=0.001, max_iter=1000):
		'''
		Fits a Linear Regression model to given training data.

		Args:
			X_train: 2D numpy array of training data (categorical columns must be encoded).
			y_train: 1D numpy array of training data labels (should be encoded as labels are categorical).
			alpha: learning rate.
			max_iter: maximum number of gradient descent steps to perform.

		Output:
			No output. Updates coefficients and model bias directly.
		'''
		if X_train.ndim == 1:
			X_train = X_train.reshape(-1, 1)

		num_features = X_train.shape[1]
		# initialise coeffs_ as random values between 1 and 10
		self.coeffs_ = np.random.uniform(1, 10, size = num_features)

		def sgd(alpha, max_iter):
			'''
			Performs stochastic gradient descent on training data.
			'''
			for j in range(max_iter):
				# iterate over every instance i (stochastic method)
				for i in range(len(X_train)):
					actual = y_train[i]
					predicted = np.dot(self.coeffs_, X_train[i]) + self.bias

					# compute gradients
					error = actual - predicted
					gradient_w = np.clip(-2 * error * X_train[i], -1e4, 1e4)
					gradient_b = -1 * error
					# update coeffs_ and bias
					self.coeffs_ -= alpha * gradient_w
					self.bias -= alpha * gradient_b

		sgd(alpha, max_iter)
		return

	def find_best_alpha(self, X_train, y_train, X_val, y_val, values_to_test, metric="rss", show_graph=True):
		'''
		Performs search to identify best alpha (learning rate) for optimal performance).

		Args:
			X_train: 2D numpy array of training data (categorical columns must be encoded).
			y_train: 1D numpy array of training data labels (should be encoded as labels are categorical).
			X_val: 2D numpy array of validation data.
			y_val: 1D numpy array of validation data.
			values_to_test: list of values of alpha to test.
			metric: default set to rss, can take mse as well.
			show_graph: default set to True, indicates whether or not to plot performance curve.

		Output:
			No output.
		'''
		results = []
		for value in values_to_test:
			lm = LinearRegression()
			lm.fit(X_train, y_train, alpha = value)
			results.append(rss if metric == "rss" else mse)

		if show_graph:
			plt.plot(values_to_test, results, color = "blue")
			plt.xlabel("Learning Rate")
			plt.ylabel(f"{metric}")
			plt.show()

		results_array = np.array(results)
		min_idx = np.nanargmin(results_array) # avoid NaN issues
		print(f"Min error achieved was {results[min_idx]} at learning rate {values_to_test{min_idx}}")
		return 

	def predict(self, X_test):
		'''
		Predicts on a given set of testing data.

		Args:
			X_test: 2D numpy array of testing data (categorical columns must be encoded).

		Output:
			1D numpy array of results.
		'''
		if X_test.ndim == 1:
			X_test = X_test.reshape(-1, 1)
		results = np.dot(X_test, self.coeffs_) + self.bias

		return results


	def predict_score(self, X_test, y_test):
		'''
		Predicts on a given set of testing data, then scores the model by both mse and rss.

		Args:
			X_test: 2D numpy array of testing data (categorical columns must be encoded).
			y_test: 1D numpy array of testing labels.

		Output:
			Two values: mse and rss (in that order).
		'''
		n_samples = X_test.shape[0]
		results = self.predict(X_test)
		mse = mse(y_test, results)
		rss = rss(y_test, results)

		return mse, rss


	def plot_regression(self, X_train, y_train):
		'''
		Plots the model's regression curve after fit operation has been performed.
		Note this method only works for simple linear regression (i.e. only 1 x- and 1 y-variable.)

		Args:
			X_train: 2D numpy array of training data (categorical columns must be encoded).
			y_train: 1D numpy array of training data labels (should be encoded as labels are categorical).

		Output:
			Plot.
		'''
		if X_train.ndim == 1:
			X_train = X_train.reshape(-1, 1)

		if X_train.shape[1] > 1:
			raise ValueError("Too many dimensions for meaningful visualisation")

		fig, ax = plt.subplots()
		ax.scatter(X_train[:, 0], y_train, color="red", alpha=0.85)

		xmin, xmax = X_train[:, 0].min(), X_train[:, 0].max() # get range of values
		x_line = np.linspace(x_min, x_max, 100) # generate more points for the line

		y_line = self.predict(x_line)

		ax.plot(x_line, y_line, color="blue", linewidth=2)
		ax.set_xlabel("Feature")
		ax.set_ylabel("Target")
		ax.set_title("Linear Regression Fit")
		plt.plot()



