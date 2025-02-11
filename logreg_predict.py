from model import MulticlassLogisticRegression
import pandas as pd
import numpy as np
import argparse
import json

def logreg_predict(dataset):
	"""
	Predicts the house for each student in the dataset.

	Parameters:
	dataset (str): The path to the dataset file.

	Returns:
	pd.DataFrame: A dataframe containing the student names and their predicted houses.
	"""
	courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
	# stds are stored in stds.json
	# means are stored in means.json
	with open("stds.json", "r") as f:
		stds = json.load(f)
	with open("means.json", "r") as f:
		means = json.load(f)

	# house_features = {
	# 	"Gryffindor" : ["History of Magic", "Transfiguration", "Flying", "Defense Against the Dark Arts"],
	# 	"Hufflepuff" : ["Defense Against the Dark Arts", "Transfiguration", "Astronomy", "Ancient Runes"],
	# 	"Ravenclaw" : ["Muggle Studies", "Charms", "Defense Against the Dark Arts"],
	# 	"Slytherin" : ["Defense Against the Dark Arts", "Herbology", "Arithmancy", "Ancient Runes"]
	# }

	# load the gile house_features.json
	with open("house_features.json", "r") as f:
		house_features = json.load(f)

	df = pd.read_csv(dataset)

	for course in courses:
		df[course] = (df[course] - means[course]) / stds[course]
		df[course] = df[course].fillna(0)

	Xs_train = [np.array([df[feature] for feature in features]).T for features in house_features.values()]
	model = MulticlassLogisticRegression()
	model.load("models.json")

	y_pred = model.predict(Xs_train)
	result = pd.read_csv('datasets/dataset_truth.csv')

	print(f"test acc {np.round((result["Hogwarts House"].to_numpy() == y_pred).mean(), 3)}")


def main():
	parser = argparse.ArgumentParser(description='Predict the house for each student in the dataset.')
	parser.add_argument('--dataset', type=str, default='datasets/dataset_test.csv', help='Path to the dataset file')
	args = parser.parse_args()

	logreg_predict(args.dataset)

if __name__ == '__main__':
	main()