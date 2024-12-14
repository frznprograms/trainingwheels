import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels

class LinearRegression(trainingwheels):
  def __init__(self, name = "Linear Regression Algorithm", bias = 1):
	self.name = name
	self.coeffs_ = None
	self.bias = bias

  @staticmethod
  def rss(actual, predicted):
	  sum = np.sum(np.subtract(actual - predicted)**2)
	  

  def fit(self, X_train, y_train, alpha = 0.001, max_iter = 1000):
	if X_train.ndim == 1:
		X_train = X_train.reshape(-1, 1)
	num_features = X_train.shape[1]
	# Initialize coefficients as random values between 1 and 10
	self.coeffs_ = np.random.uniform(1, 10, size=num_features)

	# alpha is step size
	def sgd(alpha, max_iter):
	  for j in range(max_iter):
		# iterate over every instance (since we are using stochastic method)
		for i in range(len(X_train)):
		  actual = y_train[i]
		  predicted = np.dot(self.coeffs_, X_train[i]) + self.bias

		  # Compute gradients
		  error = actual - predicted
		  gradient_w = np.clip(-2 * error * X_train[i], -1e4, 1e4)
		  gradient_b = -2 * error
		  # Update coefficients and bias
		  self.coeffs_ -= alpha * gradient_w
		  self.bias -= alpha * gradient_b

	sgd(alpha, max_iter)
	return

  def find_best_alpha(self, X_train, y_train, X_test, y_test, values_to_test = [0.001, 0.01, 0.1], metric = "rss"):
	results = []
	for value in values_to_test:
	  lm = LinearRegression()
	  lm.fit(X_train, y_train, alpha = value)
	  mse, rss = lm.score(X_test, y_test)
	  results.append(rss if metric == "rss" else mse)

	plt.plot(values_to_test, results, color = "blue")
	plt.xlabel("Learning rate")
	plt.ylabel(f"{metric}")
	plt.show()

	results_array = np.array(results)
	min_idx = np.nanargmin(results_array)  # Avoid NaN issues
	print(f"Min Score achieved was {results[min_idx]} at learning rate {values_to_test[min_idx]}.")
	return


  def predict(self, X_test):
	if X_test.ndim == 1:
		X_test = X_test.reshape(-1, 1)
	results = np.dot(X_test, self.coeffs_) + self.bias

	return results

  def score(self, X_test, y_test):
	n_samples = X_test.shape[0]
	results = self.predict(X_test)
	mse = np.mean((results - y_test)**2)
	rss = rss(y_test, results)

	return mse, rss

  def plot_regression(self, X_train, y_train):
	if X_train.ndim == 1:
		X_train = X_train.reshape(-1, 1)

	if X_train.shape[1] > 1:
	  raise ValueError("Too many dimensions for meaningful visualisation.")

	fig, ax = plt.subplots()
	ax.scatter(X_train[:, 0], y_train, color='red', alpha=0.85)

	x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()  # Get range for feature
	x_line = np.linspace(x_min, x_max, 100)  # Create 100 points for the line

	# Compute the predicted y values using the learned coefficients
	y_line = self.predict(x_line)

	ax.plot(x_line, y_line, color='blue', linewidth=2)
	ax.set_xlabel('Feature')
	ax.set_ylabel('Target')
	ax.set_title('Linear Regression Fit')
	plt.plot()


