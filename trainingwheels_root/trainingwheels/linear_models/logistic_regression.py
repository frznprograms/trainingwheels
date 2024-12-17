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
		'''
		Fits a Linear Regression model to given training data.

		Args:
			X_train: 2D numpy array of training data (categorical columns must be encoded).
			y_train: 1D numpy array of training data labels (should be encoded as labels are categorical).
			alpha: learning rate.

		Output:
			No output. Updates coefficients and model bias directly.
		'''
		if X_train.ndim == 1:
				X_train = X_train.reshape(-1, 1)

		# enable warm start -> model can be trained again even with existing coeffs_
		if self.coeffs_ is None:
			self.coeffs_ = np.random.random(size = X_train.shape[1])

		def sgd(alpha):
			'''
			Performs stochastic gradient descent on training data.
			'''
			
			# linear combination of log odds
			raw_predictions = np.dot(X_train, self.coeffs_) + self.bias
			predicted_probs = sigmoid(raw_predictions)
			
			# get gradients and update coefficients
			coeff_derivative = np.dot(X_train.T, (predicted_probs - y_train))
			bias_derivative = np.sum(predicted_probs - y_train)
			self.coeffs_ -= alpha * coeff_derivative
			self.bias -= alpha * bias_derivative

		sgd(alpha)
		return

	def predict(self, X_test, threshold = 0.5):
		'''
		Predicts on a given set of testing data.

		Args:
			X_test: 2D numpy array of testing data (categorical columns must be encoded).
			threshold: default set to 0.5; determines decision boundary for logistic regression.

		Output:
			1D numpy array of results.
		'''
		if self.coeffs_ is None:
			print("Model coefficients have not been initialised.")
		
		if X_test.ndim == 1:
				X_test = X_test.reshape(-1, 1)
		
		raw_predictions = np.dot(X_test, self.coeffs_) + self.bias
		predicted_probs = ML.sigmoid(raw_predictions)
		
		return np.array([1 if prob > threshold else 0 for prob in predicted_probs])

	
	def predict_score(self, X_test, y_test, threshold = 0.5):
		'''
		Predicts on a given set of testing data, then scores the model by accuracy.

		Args:
			X_test: 2D numpy array of testing data (categorical columns must be encoded).
			y_test: 1D numpy array of testing labels.
			threshold: default set to 0.5; determines decision boundary for logistic regression.

		Output:
			Accuracy score of the model.
		'''
		
		predictions = self.predict(X_test, threshold)
		score = 0
		for i in range(len(predictions)):
			if predictions[i] == y_test[i]:
				score += 1
		return (score / len(predictions))*100


