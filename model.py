import numpy as np

class LogisticRegression:
	def __init__(self, n_features, lr=0.01, epochs=1000):
		self.lr = lr
		self.epochs = epochs
		self.weights = np.random.rand(n_features + 1, 1)
		self.xmean = 0
		self.xstd = 1
		self.n_features = n_features

	def normalize(self, X):
		# self.xmean = X.mean()
		# self.xstd = X.std()
		# X = (X - self.xmean) / self.xstd
		return X

	def forward(self, X):
		# X : shape : (*, n_features)
		# W : shape : (n_features, out_features)
		return 1 / (1 + np.exp(-X @ self.weights))
	
	def loss(self, y, y_pred):
		return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()

	def backward(self, y_pred, y, X):
		grad = ((y_pred - y).reshape(-1, 1) * X).mean(axis=0)
		self.weights -= (self.lr * grad).reshape(-1, 1)

	def predict(self, X):
		# X = (X - self.xmean) / self.xstd
		X = np.insert(X, 0, 1, axis=1)
		return self.forward(X)

	def accuracy(self, y_pred, y):
		y_pred = y_pred.squeeze()
		y_pred = (y_pred > 0.5).astype(int)
		y = (y > 0.5).astype(int)
		print((y == y_pred).mean())

	def fit(self, X, y):
		X = self.normalize(X)
		# print(X[:10])
		X = np.insert(X, 0, 1, axis=1)
		for epoch in range(self.epochs):
			y_pred = self.forward(X)
			y_pred = y_pred.squeeze()
			self.backward(y_pred, y, X)
			# if epoch % 100 == 0:
				# print(f"Epoch : {epoch}, Loss: {self.loss(y, y_pred)}")
		y_pred = self.forward(X)
		y_pred = y_pred.squeeze()