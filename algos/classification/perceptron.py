import numpy as np

class Perceptron(object):
	"""
	Perceptron classifier.

	Parameters
	------------
	eta: float
		Learning rate (between 0.0 and 1.0).
	n_iter: int
		Number of iterations over the training set.

	Attributes
	------------
	w_: 1d-array
		Weights after fitting 
	errors_: list
		Number of misclassification for each iterations

	"""

	def __init__(self, eta=0.1, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1])
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self



	def predict(self, X):
		return self.activate(np.dot(X, self.w_[1:]) + self.w_[0])

	def activate(self, value):
		return np.where(value >= 0, 1, -1)
