import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels

class LogisticRegression(trainingwheels):
	def __init__(self, name = "Logistic Regression Algorithm", bias = 1):
		self.name = name
		self.bias = bias
		self.coeffs_ = None

	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def fit(self, X_train, y_train, alpha = 0.001):
		if X_train.ndim == 1:
				X_train = X_train.reshape(-1, 1)

		# enable warm start
		if self.coeffs_ is None:
			self.coeffs_ = np.random.random(size = X_train.shape[1])

		def sgd(X_train, y_train, alpha):
			# linear combination of log odds
			raw_predictions = np.dot(X_train, self.coeffs_) + self.bias
			predicted_probs = sigmoid(raw_predictions)
			# get gradients and update coefficients
			coeff_derivative = np.dot(X_train.T, (predicted_probs - y_train))
			bias_derivative = np.sum(predicted_probs - y_train)
			self.coeffs_ -= alpha * coeff_derivative
			self.bias -= alpha * bias_derivative

		sgd(X_train, y_train, alpha)
		return

	def predict(self, X_test, threshold = 0.5):
		if self.coeffs_ is None:
			print("coeffs are None!")
		if X_test.ndim == 1:
				X_test = X_test.reshape(-1, 1)
		raw_predictions = np.dot(X_test, self.coeffs_) + self.bias
		predicted_probs = ML.sigmoid(raw_predictions)
		return [1 if prob > threshold else 0 for prob in predicted_probs]

	def predict_score(self, X_test, y_test, threshold = 0.5):
		'''
		returns accuracy of the model
		'''
		predictions = self.predict(X_test, threshold)
		score = 0
		for i in range(len(predictions)):
			if predictions[i] == y_test[i]:
				score += 1
		return (score / len(predictions))*100


