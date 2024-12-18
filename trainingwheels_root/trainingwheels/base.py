from itertools import product
import pandas as pd
import numpy as np


class trainingwheels:
	def __init__(self):
		pass

	@staticmethod
	def euclidean_dist(a, b):
		return np.linalg.norm(a-b, ord=2)

	@staticmethod
	def rss(actual, predicted):
		return np.sum(np.subtract(actual - predicted) ** 2)

	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def grid_search(self, models, fit_hyperparams, X_train, y_train, model_hyperparams=None):
		'''
		Performs grid search for hyperparameter optimisation.

		Args:
			models: dictionary of models to test.
			fit_hyperparams: parameters to be tuned for the fit() method of the model.
			X_train: 2D numpy array of training data (categorical features must be encoded).
			y_train: 1D numpy array of training labels (categorical features must be encoded).
			model_hyperparams: parameters to be set in the model __init__ method, not the fit function.

		Output:
			Best model instance, best score achieved by the model's predict_score() method, and the hyperparams
			to achieve that score.

		'''
		best_model, best_score, best_params = None, -np.inf, None

		for model in models:
				model_class = type(model)

				# Retrieve hyperparameter grids
				param_grid = fit_hyperparams.get(model_class, {})
				if model_hyperparams:
					model_grid = model_hyperparams.get(model_class, {})

				# Generate all possible combinations of hyperparameters
				param_combinations = list(product(*param_grid.values()))

				for params in param_combinations:
						param_dict = dict(zip(param_grid.keys(), params))
						model_instance = model_class(**model_grid)
						model_instance.fit(X_train, y_train, **param_dict)
						score = model_instance.predict_score(X_train, y_train)

						# Update best model if performance/score improves
						if score > best_score:
								best_model, best_score, best_params = model_instance, score, param_dict

		return best_model, best_score, best_params

	
	def grid_search_cv(self, models, fit_hyperparams, X_train, y_train, model_hyperparams=None, cv=5):
		'''
		Performs grid search for hyperparameter optimisation, with cross-validation.

		Args:
			models: dictionary of models to test.
			fit_hyperparams: parameters to be tuned for the fit() method of the model.
			X_train: 2D numpy array of training data (categorical features must be encoded).
			y_train: 1D numpy array of training labels (categorical features must be encoded).
			model_hyperparams: parameters to be set in the model __init__ method, not the fit function.
			cv: number of folds for cross-validation.

		Output:
			Best model instance, best score achieved by the model's predict_score() method, and the hyperparams
			to achieve that score.

		'''
		best_model, best_score, best_params = None, -np.inf, None

		# Shuffle the data
		indices = np.arange(len(X_train))
		np.random.shuffle(indices)
		X_train = X_train[indices]
		y_train = y_train[indices]

		for model in models:
				model_class = type(model)

				# Retrieve hyperparameter grids
				param_grid = fit_hyperparams.get(model_class, {})
				if model_hyperparams:
					model_grid = model_hyperparams.get(model_class, {})

				param_combinations = list(product(*param_grid.values()))

				for params in param_combinations:
						param_dict = dict(zip(param_grid.keys(), params))
						model_instance = model_class(**model_grid)

						# k-fold cross-validation
						model_scores = []
						fold_size = len(X_train) // cv
						for i in range(cv):
								# Create validation and training folds
								X_val = X_train[i * fold_size : (i + 1) * fold_size]
								y_val = y_train[i * fold_size : (i + 1) * fold_size]
								X_train_cv = np.concatenate((X_train[:i * fold_size], X_train[(i + 1) * fold_size:]), axis=0)
								y_train_cv = np.concatenate((y_train[:i * fold_size], y_train[(i + 1) * fold_size:]), axis=0)

								# Train model
								model_instance.fit(X_train_cv, y_train_cv, **param_dict)
								# Evaluate on validation set
								score = model_instance.predict_score(X_val, y_val)
								model_scores.append(score)

						avg_score = np.mean(model_scores)

						# Update best score and parameters
						if avg_score > best_score:
								best_model, best_score, best_params = model_class(), avg_score, param_dict

		# Train the best model on the full training data
		best_model.fit(X_train, y_train, **best_params)

		return best_model, best_score, best_params


