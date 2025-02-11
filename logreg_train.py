from model import MulticlassLogisticRegression
import pandas as pd
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# The order in this list maps to columns in the probability array:
houses_list = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# Your color mapping for each house
colors_dict = {
	"Gryffindor": "red",
	"Hufflepuff": "yellow",
	"Ravenclaw": "blue",
	"Slytherin": "green"
}

def plot_decision_boundaries(model, df, house_features):
	"""
	model: dict of 4 one-vs-rest classifiers.
	X: feature array of shape (n_samples, 2).
	y: array/list of house labels (e.g. strings "Gryffindor", etc.).
	"""
	x_min, x_max = df[house_features["Gryffindor"][0]].min() - 0.25, df[house_features["Gryffindor"][0]].max() + 0.25
	y_min, y_max = df[house_features["Gryffindor"][1]].min() - 0.25, df[house_features["Gryffindor"][1]].max() + 0.25

	ax = plt.subplot(1, 2, 1)
	df.plot.scatter(x=house_features["Gryffindor"][0], y=house_features["Gryffindor"][1], c=df['Hogwarts House'].apply(lambda x: colors_dict[x]), ax=ax)

	for house, binary_model in model.models.items():
		theta = binary_model.weights
		x1 = np.linspace(x_min, x_max, 100)
		x2 = - (theta[0] + theta[1] * x1) / theta[2]
		plt.plot(x1, x2, label=house, color=colors_dict[house])

	# set labels
	ax.set_xlabel(house_features["Gryffindor"][0])
	ax.set_ylabel(house_features["Gryffindor"][1])
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.legend()

	ax = plt.subplot(1, 2, 2)

	xx1, xx2 = np.meshgrid(
		np.linspace(x_min, x_max, 300),
		np.linspace(y_min, y_max, 300)
	)
	grid_points = np.c_[xx1.ravel(), xx2.ravel()]
	grid_points = [grid_points] * 4
	
	preds = model._predict(grid_points)
	Z = preds.reshape(xx1.shape)

	cmap = ListedColormap([colors_dict[h] for h in houses_list])
	
	ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	
	df.plot.scatter(x=house_features["Gryffindor"][0], y=house_features["Gryffindor"][1], c=df['Hogwarts House'].apply(lambda x: colors_dict[x]), ax=ax)
	
	ax.set_xlabel(house_features["Gryffindor"][0])
	ax.set_ylabel(house_features["Gryffindor"][1])

	plt.draw()
	plt.pause(0.01)

def logreg_train(dataset, batch_size, plot_boundaries, anim_boundaries, plot_loss, momentum):
	"""
	Trains a logistic regression model for each house.

	Parameters:
	dataset (str): The path to the dataset file.
	houses (list): The list of Hogwarts houses.

	Returns:
	MulticlassLogisticRegression: The trained logistic regression model.
	"""
	courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
	stds = {}
	means = {}

	df = pd.read_csv(dataset)
	for course in courses:
		stds[course] = df[course].std()
		means[course] = df[course].mean()
		df[course] = (df[course] - means[course]) / stds[course]
		df[course] = df[course].fillna(0)

	model = MulticlassLogisticRegression(momentum=momentum)

	houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

	if anim_boundaries or plot_boundaries:
		house_features = {
			"Gryffindor" : ["Ancient Runes", "Herbology"],
			"Hufflepuff" : ["Ancient Runes", "Herbology"],
			"Ravenclaw" : ["Ancient Runes", "Herbology"],
			"Slytherin" : ["Ancient Runes", "Herbology"]
		}

		features_set = set(house_features["Gryffindor"])
		for house, features in house_features.items():
			if set(features) != features_set:
				raise ValueError(f"House '{house}' does not share the same features as other houses")
			if len(features) != 2:
				raise ValueError(f"House '{house}' does not have exactly two features")
		
		fig = plt.figure(figsize=(16, 8))
		plt.ion()
		for epoch in range(200):
			for house in houses:
				losses, accuracies = model.fit(house, df, house_features[house], lr=0.05, epochs=1, batch_size=batch_size)
			if anim_boundaries:
				plt.clf()
				plot_decision_boundaries(model, df, house_features)
		
		plt.ioff()
		plt.clf()
		plot_decision_boundaries(model, df, house_features)
		plt.show()
	else:
		house_features = {
			"Gryffindor" : ["History of Magic", "Transfiguration", "Flying", "Defense Against the Dark Arts"],
			"Hufflepuff" : ["Defense Against the Dark Arts", "Transfiguration", "Astronomy", "Ancient Runes"],
			"Ravenclaw" : ["Muggle Studies", "Charms", "Defense Against the Dark Arts"],
			"Slytherin" : ["Defense Against the Dark Arts", "Herbology", "Arithmancy", "Ancient Runes"]
		}

		house_losses = {}
		house_accuracies = {}
		for house in houses:
			losses, accuracies = model.fit(house, df, house_features[house], lr=0.2, epochs=200, batch_size=batch_size)
			house_losses[house] = losses
			house_accuracies[house] = accuracies

		if plot_loss:
			fig = plt.figure(figsize=(16, 8))
			ax = plt.subplot(1, 2, 1)
			for house, losses in house_losses.items():
				ax.plot(losses, label=house)
			ax.set_title("Loss")
			ax.set_xlabel("Epoch")
			ax.set_ylabel("Loss")
			ax.legend()

			ax = plt.subplot(1, 2, 2)
			for house, accuracies in house_accuracies.items():
				ax.plot(accuracies, label=house)
			ax.set_title("Accuracy")
			ax.set_xlabel("Epoch")
			ax.set_ylabel("Accuracy")
			ax.legend()

			plt.show()

	model.save("models.json")
	with open("means.json", "w") as f:
		json.dump(means, f)
	with open("stds.json", "w") as f:
		json.dump(stds, f)
	with open("house_features.json", "w") as f:
		json.dump(house_features, f)

def main():
	parser = argparse.ArgumentParser(description='Train a logistic regression model for each house.')
	parser.add_argument('--dataset', type=str, default='datasets/dataset_train.csv', help='Path to the dataset file')
	parser.add_argument('--batch_size', type=int, default=None, help='The batch size for training')
	parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for training')
	parser.add_argument('--plot_boundaries', action='store_true', help='Plot the decision boundaries')
	parser.add_argument('--anim_boundaries', action='store_true', help='Animate the decision boundaries')
	parser.add_argument('--plot_loss', action='store_true', help='Plot the loss and accuracy')
	args = parser.parse_args()

	if (args.plot_boundaries and args.plot_loss) or (args.anim_boundaries and args.plot_loss):
		raise ValueError("Cannot plot loss and accuracy with decision boundaries or animated decision boundaries")

	logreg_train(args.dataset, args.batch_size, args.plot_boundaries, args.anim_boundaries, args.plot_loss, args.momentum)

if __name__ == '__main__':
	main()