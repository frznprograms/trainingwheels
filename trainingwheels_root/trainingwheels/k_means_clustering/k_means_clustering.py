import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ..base import trainingwheels

class KMeansClustering(trainingwheels):
	def __init__(self, k, name = "K-Means Clustering Algorithm"):
		self.k = k
		super().__init__(name)
		self.centroids = []
		self.clusters = {}

	@staticmethod
	def euclidean_dist(a, b):
		return np.linalg.norm(a - b, ord=2)

	def __repr__(self):
		return f"KMeansClustering(centroids={self.centroids})"


	def fit(self, X_train, tolerance = 0.05, max_iter=1000):
		'''
		Fits appropriate clusters to data.

		Args:
			X_train: 2D numpy array of training data (categorical features must be encoded).
			tolerance: sum of distance of cluster points from centroids at which convergence is achieved.
			max_iter: maximum number of iterations.

		Output:
			No output. Model centroids and clusters are directly updated.

		'''
		self.centroids = [] # reset cluster centroids in event of re-call of fit()

		# Helper function; determines if convergence has been achieved
		def is_converged(tolerance, next_centroids, t, max_iter):
			if t == 0:
				return False # skip first iteration as next_centroids do not exist

			sum = 0
			for i in range(len(next_centroids)):
				sum += euclidean_dist(next_centroids[i], self.centroids[i])
			if sum < tolerance or t > max_iter:
				return True

			return False

		# Initialize centroids randomly from data points (faster convergence that choosing random points in feature space)
		self.centroids = []
		random_indices = np.random.choice(range(len(X_train)), self.k, replace = False)
		for j in random_indices:
			self.centroids.append(X_train[j])

		for t in range(max_iter):
			clusters = {i: [] for i in range(self.k)}
			# each cluster stores its points

			# Assign points to the nearest centroid
			for instance in X_train:
				distances = np.array([ML.euclidean_dist(instance, centroid) for centroid in self.centroids])
				closest_centroid_idx = np.argmin(distances)
				clusters[closest_centroid_idx].append(instance)

			# Update centroids by calculating the mean of points in each cluster
			next_centroids = []
			for i in range(self.k):
				if clusters[i]:
					new_centroid = np.mean(clusters[i], axis=0)
					next_centroids.append(new_centroid)
				else:
					# retain current centroid if it has no points
					next_centroids.append(self.centroids[i])

			if is_converged(tolerance, next_centroids, t, max_iter):
				self.clusters = clusters
				break

			self.centroids = next_centroids

		return


	def predict_score(self, X_test):
		'''
		Fits appropriate clusters to data.

		Args:
			X_test: 2D numpy array of testing data (categorical features must be encoded).

		Output:
			List of results, indicating which of the centroids each point belongs to.
			Sum of all clusters' within-cluster-sum-of-squares.

		'''
		# update centroids and clusters as well
		results = self.predict(X_test)
		total_wcss = 0
		for i, cluster_points in model.clusters.items():
				total_wcss += np.sum((cluster_points - self.centroids[i]) ** 2)

		return results, total_wcss


	def predict(self, X_test):
		'''
		Predicts cluster of points (i.e. nearest centroid).

		Args:
			X_test: 2D numpy array of testing data (categorical features must be encoded).

		Output:
			List of results, indicating which of the centroids each point belongs to.
			(points are matched by index, i.e. if results[0] = 1, the first point belongs to cluster 1)

		'''
		if self.centroids == []:
			raise TypeError("Model does not have any centroids initialised; Train the model first.")

		results = []
		for instance in X_test:
			distances = np.array([ML.euclidean_dist(instance, centroid) for centroid in self.centroids])
			closest_centroid_idx = np.argmin(distances)
			self.clusters[closest_centroid_idx].append(instance)
			results.append(closest_centroid_idx)

		return results

	
	def plot_centroids(self, X_train):
		'''
		Plots centroids and their clusters.
		** NOTE ** 
		Model must first be trained before this method can be used for a meaningful visualisation.

		Args:
			X_train: 2D numpy array of training data (categorical features must be encoded).

		Output:
			Plot.

		'''
		num_dimensions = X_train.shape[1]
		if num_dimensions > 2:
			raise ValueError("More than 2 features/dimensions detected. Unable to plot effective visualisation.")

		# get k colours for k clusters
		colours = plt.cm.get_cmap("tab10", self.k)

		fig, ax = plt.subplots()
		for i in range(self.k):
			cluster_points = np.array(self.clusters[i])
			ax.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha = 0.8, c = colours(i))
			ax.scatter(self.centroids[i][0], self.centroids[i][1], color = colours(i), edgecolor = 'black', s = 100, marker ='X')
		plt.title("Model clusters and Centroids")
		plt.show()

	
	def test_k(self, X_train, test_k_values):
		'''
		Naive hyperparameter tuning to find the optimal number of clusters for best model performance.
		(Assessed by within-cluster-sum-of-squares)

		Args:
			X_train: 2D numpy array of training data (categorical features must be encoded).
			test_k_valeus: list of values to test for the number of clusters.

		Output:
			Plot.

		'''
		wcss_results = []

		for k in test_k_values:
			model = KMeansClustering(k)
			model.fit(X_train)

			total_wcss = 0
			for i, cluster_points in model.clusters.items():
				total_wcss += np.sum((np.array(cluster_points) - model.centroids[i]) ** 2)
			wcss_results.append(total_wcss)

		plt.plot(test_k_values, wcss_results, 'bo-', label = "WCSS")
		plt.xticks(test_k_values)
		plt.xlabel("Number of clusters k")
		plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
		plt.title("WCSS against k values for KMeansClustering")
		plt.show()


