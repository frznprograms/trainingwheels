import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels

class MixedNaiveBayes(trainingwheels):
  def __init__(self, name = "Naive Bayes Classifier", continuous_features = []):
    self.name = name
    self.priors = {}
    self.likelihoods = {}
    # indices for where there are continuous variables
    self.continuous_features = continuous_features

  def update_cont_features(self, new_features):
    self.continuous_features = new_features

  def fit(self, X_train, y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    # Set up prior probabilities
    for i in range(len(classes)):
      self.priors[classes[i]] = np.log((counts[i] + 1) / y_train.shape[0])

    # Set up likelihoods
    for i in range(X_train.shape[1]):
      if i in self.continuous_features:
        self.run_gaussian_fitting(X_train, y_train, i)
      else:
        self.likelihoods[i] = {}
        for cls in self.priors.keys():
          # filter data
          subset_data = X_train[y_train == cls][:, i]
          feature_values, feature_counts = np.unique(subset_data, return_counts=True)
          for j in range(len(feature_values)):
            #if feature_values[j] not in self.likelihoods[i]:
            self.likelihoods[i][feature_values[j]] = {}
            self.likelihoods[i][feature_values[j]][cls] = np.log((feature_counts[j] + 1) / subset_data.shape[0])
    return

  def run_gaussian_fitting(self, X_train, y_train, feature_idx):
    """
    Compute mean and variance for the continuous feature at index `feature_idx`.
    """
    self.likelihoods[feature_idx] = {}
    for cls in self.priors.keys():
      subset_data = X_train[y_train == cls][:, feature_idx]
      self.likelihoods[feature_idx][cls] = {
        "mean": np.mean(subset_data),
        "var": np.var(subset_data)
      }

  def predict(self, X_test):
    predictions = []
    # for every row in X_test
    for i in range(X_test.shape[0]):
      class_probs = {}
      # probability is prior *= likelihoods of every feature
      for cls in self.priors.keys():
        prob = self.priors[cls]

        for j in range(X_test.shape[1]):
          if j in self.continuous_features:
            prob *= self.run_gaussian_predict(X_test[i], j, cls)
          else:
            value = X_test[i][j]
            prob *= self.likelihoods[j].get(value, {}).get(cls, np.log(1e-16))

        class_probs[cls] = prob

      predicted_class = max(class_probs, key = class_probs.get)
      predictions.append(predicted_class)

    return np.array(predictions)

  def run_gaussian_predict(self, X_test_row, feature_idx, cls):
    """
    Compute the Gaussian likelihood for a continuous feature during prediction.
    """
    mean = self.likelihoods[feature_idx][cls]["mean"]
    var = self.likelihoods[feature_idx][cls]["var"]
    value = X_test_row[feature_idx]

    # Avoid division by zero for variance
    if var == 0: var = 1e-6
    likelihood = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((value - mean) ** 2) / (2 * var))
    return likelihood

  def predict_score(self, X_test, y_test):
    '''
    return accuracy by default
    '''
    results = self.predict(X_test)
    score = 0
    for i in range(len(results)):
      if results[i] == y_test[i]:
        score += 1

    return score * 100 / y_test.shape[0]
