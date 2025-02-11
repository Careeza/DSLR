import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def ft_pair_plot(df, courses, save):
	hogwarts_House = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
	colors_dict = {
		"Gryffindor": "red",
		"Hufflepuff": "yellow",
		"Ravenclaw": "blue",
		"Slytherin": "green"
	}
	
	fig = plt.figure(figsize=(3 * len(courses), 3 * len(courses)))
	for i, feature_1 in enumerate(courses):
		for j, feature_2 in enumerate(courses):
			if feature_1 != feature_2:
				ax = plt.subplot(len(courses), len(courses), i * len(courses) + j + 1)
				df.plot.scatter(x=feature_1, y=feature_2, c=df['Hogwarts House'].apply(lambda x: colors_dict[x]), ax=ax)
			else:
				ax = plt.subplot(len(courses), len(courses), i * len(courses) + j + 1)
				for house in hogwarts_House:
					df[df["Hogwarts House"] == house][feature_1].hist(alpha=0.5, label=house, color=colors_dict[house])
	plt.tight_layout()
	if save:
		plt.savefig("pair_plot.png")
	else:
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Describe the dataset')
	parser.add_argument('--dataset', type=str, default='datasets/dataset_train.csv', help='Path to the dataset file')
	parser.add_argument('--courses', type=str, nargs='+', default=['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'], help='List of courses to plot')
	parser.add_argument('--save', action='store_true', help='Save the plot to the file scatter.png')
	args = parser.parse_args()
	df = pd.read_csv(args.dataset)
	ft_pair_plot(df, args.courses, args.save)
