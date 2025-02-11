import numpy as np
import json

class BinaryLogisticRegression:
	def __init__(self, n_features, weights=None, momentum=0):
		if weights is not None:
			self.weights = weights
		else:
			self.weights = np.random.rand(n_features + 1, 1)
		self.n_features = n_features
		self.momentum = momentum
		self.velocity = np.zeros_like(self.weights)

	def forward(self, X):
		return 1 / (1 + np.exp(-X @ self.weights))
	
	def loss(self, y, y_pred):
		return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()

	def backward(self, y_pred, y, X, lr):
		grad = ((y_pred - y).reshape(-1, 1) * X).mean(axis=0)
		self.velocity = self.momentum * self.velocity + lr * grad.reshape(-1, 1)
		self.weights -= self.velocity

	def _predict(self, X):
		y_pred = self.forward(X)
		y_pred = y_pred.squeeze()
		return (y_pred > 0.5).astype(int)

	def predict_proba(self, X):
		X = np.insert(X, 0, 1, axis=1)
		return self.forward(X).squeeze()

	def predict(self, X):
		X = np.insert(X, 0, 1, axis=1)
		return self._predict(X)

	def _accuracy(self, X, y):
		y_pred = self._predict(X)
		return (y == y_pred).mean() * 100

	def accuracy(self, X, y):
		y_pred = self.predict(X)
		return (y == y_pred).mean() * 100

	def fit(self, X, y, lr=0.2, epochs=1000, batch_size=None):
		X = np.insert(X, 0, 1, axis=1)
		y = y.astype(float)
		losses = []
		accuracies = []
		if batch_size is None:
			batch_size = X.shape[0]
		for epoch in range(epochs):
			idx = np.random.choice(X.shape[0], X.shape[0], replace=False)
			losses_batch = []
			accuracies_batch = []
			for i in range(0, X.shape[0], batch_size):
				X_batch = X[idx[i:max(i + batch_size, X.shape[0])]]
				y_batch = y[idx[i:max(i + batch_size, X.shape[0])]]
				y_pred = self.forward(X_batch)
				y_pred = y_pred.squeeze()
				loss = self.loss(y_batch, y_pred)
				self.backward(y_pred, y_batch, X_batch, lr)
				accuracy = self._accuracy(X_batch, y_batch)
				losses_batch.append(loss)
				accuracies_batch.append(accuracy)
			losses.append(np.mean(losses_batch))
			accuracies.append(np.mean(accuracies_batch))
		print(f"loss: {loss:.4f} - accuracy: {accuracies[-1]:.2f}, batch_size: {batch_size}")

		return losses, accuracies
	
class MulticlassLogisticRegression:
	def __init__(self, momentum=0):
		self.models = {}
		self.momentum = momentum

	def fit(self, house, df, features, lr=0.2, epochs=1000, batch_size=None):
		X_train = np.array([df[feature] for feature in features]).T
		y_train = (df["Hogwarts House"] == house).to_numpy(dtype=float)

		if house not in self.models:
			self.models[house] = BinaryLogisticRegression(X_train.shape[1], momentum=self.momentum)
		return self.models[house].fit(X_train, y_train, lr, epochs, batch_size)

	def _predict(self, Xs):
		predictions = np.column_stack([
			self.models["Gryffindor"].predict_proba(Xs[0]),
			self.models["Hufflepuff"].predict_proba(Xs[1]),
			self.models["Ravenclaw"].predict_proba(Xs[2]),
			self.models["Slytherin"].predict_proba(Xs[3])
		])
		
		preds = np.argmax(predictions, axis=1)

		return preds

	def predict(self, Xs):
		hogwarts_House = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

		predictions = np.column_stack([
			self.models["Gryffindor"].predict_proba(Xs[0]),
			self.models["Hufflepuff"].predict_proba(Xs[1]),
			self.models["Ravenclaw"].predict_proba(Xs[2]),
			self.models["Slytherin"].predict_proba(Xs[3])
		])
		
		house_idx = np.argmax(predictions, axis=1)
		final_predictions = np.array([hogwarts_House[idx] for idx in house_idx])

		return final_predictions
	
	def accuracy(self, Xs, y):
		predictions = self.predict(Xs)
		return (predictions == y).mean() * 100
	
	def save(self, filename):
		weights = {}
		for house, model in self.models.items():
			weights[house] = model.weights.tolist()
		with open(filename, "w") as f:
			json.dump(weights, f)

	def load(self, filename):
		with open(filename, "r") as f:
			weights = json.load(f)
		for house, w in weights.items():
			self.models[house] = BinaryLogisticRegression(len(w) - 1, weights=w)