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


	def fit(self, X_train, tolerance = 0.05, max_iter=1000):
		self.centroids = []

		# Helper function
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
		# update centroids and clusters as well
		results = self.predict(X_test)
		total_wcss = 0
		for i, cluster_points in model.clusters.items():
				total_wcss += np.sum((cluster_points - self.centroids[i]) ** 2)

		return results, total_wcss


	def predict(self, X_test):
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
		Plot centroids together with their clusters
		[Only available for 2-dimensional data i.e. m = 2 in array shape (n, m)]
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
		Iterate through different k values to find optimal k for data
		[Default metric: Within-Cluster Sum of Squares]
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


