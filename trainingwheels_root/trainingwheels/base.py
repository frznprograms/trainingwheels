class trainingwheels:
  def __init__(self):
	self.name = "training wheels, the beginners package for machine learning"

  @staticmethod
  def euclidean_dist(a, b):
	  return np.linalg.norm(a - b, ord=2)

  @staticmethod
  def rss(actual, predicted):
	  sum = np.sum(np.subtract(actual - predicted)**2)

  @staticmethod
  def sigmoid(z):
	return 1 / (1 + np.exp(-z))

  @staticmethod
  # sequences is a list of lists
  def cartesian_product(*sequences):
	indices = [0] * len(sequences) # number of parameters to tune
	while True:
	  yield tuple(sequences[i][indices[i]] for i in range(len(sequences)))
	  for i in reversed(range(len(sequences))):
		indices[i] += 1
		if indices[i] < len(sequences[i]):
		  break
		indices[i] = 0
	  else:
		return

  def grid_search(self, models, model_hyperparams, X_train, y_train):
	best_model, best_score, best_params = None, -np.inf, None

	for model in models:
	  model_class = type(model)
	  param_grid = model_hyperparams[model_class]
	  # generate all possible combinations of hyperparameters
	  param_combinations = self.cartesian_product(*param_grid.values())

	  for params in param_combinations:
		param_dict = dict(zip(param_grid.keys(), params))
		model_instance = model_class()
		model_instance.fit(X_train, y_train, **param_dict)
		score = model_instance.predict_score(X_train, y_train)

		# update best model if performance/score improves
		if score > best_score:
		  best_model, best_score, best_params = model_instance, score, param_dict

	return best_model, best_score, best_params

  def grid_search_cv(self, models, model_hyperparams, X_train, y_train, cv = 5):
	best_model, best_score, best_params = None, -np.inf, None

	for model in models:
	  model_class = type(model)
	  param_grid = model_hyperparams[model_class]
	  param_combinations = list(self.cartesian_product(*param_grid.values()))

	  for params in param_combinations:
		param_dict = dict(zip(param_grid.keys(), params))
		model_instance = model_class()
		model_instance.coeffs_, model_instance.bias = None, 1

		# scores for cross-validation
		model_scores = []
		# k-fold cross-validation
		fold_size = len(X_train) // cv
		for i in range(cv):
		  # split data into training and validation
		  X_val = X_train[i * fold_size : (i+1) * fold_size]
		  y_val = y_train[i * fold_size : (i+1) * fold_size]
		  X_train_cv = np.concatenate((X_train[:i * fold_size], X_train[(i+1) * fold_size:]), axis = 0)
		  y_train_cv = np.concatenate((y_train[:i * fold_size], y_train[(i+1) * fold_size:]), axis = 0)

		  # train model
		  model_instance.fit(X_train_cv, y_train_cv, **param_dict)
		  # evaluate on validation set
		  score = model_instance.predict_score(X_val, y_val)
		  model_scores.append(score)

		avg_score = np.mean(model_scores)
		# update best score and parameters
		if avg_score > best_score:
		  best_model, best_score, best_params = model_instance, avg_score, param_dict

	return best_model, best_score, best_params



