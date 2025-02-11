import pandas as pd
from tools import get_KL_matrixes
import numpy as np
import argparse
import matplotlib.pyplot as plt

def plot_similarity_heatmap(df, courses, save):
	if len(courses) < 2:
		raise ValueError("Please provide at least 2 courses to compare")

	kl_matrixes = get_KL_matrixes(df, courses)

	frobenius_df = pd.DataFrame(index=courses, columns=courses)

	for i, name1 in enumerate(courses):
		for j, name2 in enumerate(courses):
			frobenius_norm = np.linalg.norm(kl_matrixes[name1] - kl_matrixes[name2])
			frobenius_df.loc[name1, name2] = frobenius_norm

	frobenius_df = frobenius_df.astype(float)

	plt.figure(figsize=(9, 9))
	data = frobenius_df.values
	im = plt.imshow(data, cmap='coolwarm')

	plt.colorbar(im, fraction=0.046, pad=0.04)

	plt.xticks(np.arange(len(courses)), labels=courses, rotation=45, ha='right')
	plt.yticks(np.arange(len(courses)), labels=courses, rotation=0)

	for i in range(len(courses)):
		for j in range(len(courses)):
			text = plt.text(j, i, f"{data[i, j]:.2f}",
						ha="center", va="center", color="w")

	plt.title("Similarity between courses")
	plt.tight_layout()

	if save:
		plt.savefig("similarity_heatmap.png")
	else:
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Describe the dataset')
	parser.add_argument('--dataset', type=str, default='datasets/dataset_train.csv', help='Path to the dataset file')
	parser.add_argument('--courses', type=str, nargs='+', default=['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'], help='List of courses to plot')
	parser.add_argument('--save', action='store_true', help='Save the plot to the file scatter.png')
	args = parser.parse_args()
	df = pd.read_csv(args.dataset)
	plot_similarity_heatmap(df, args.courses, args.save)
