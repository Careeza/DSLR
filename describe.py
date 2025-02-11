import pandas as pd
import argparse
from tools import ft_count, ft_mean, ft_std, ft_min, ft_percentile, ft_max, get_KL_matrixes

def describe(filename, courses):
	"""
	Reads a CSV file and calculates descriptive statistics for specified courses.

	Parameters:
	filename (str): The path to the CSV file.

	Returns:
	pd.DataFrame: A dataframe containing the descriptive statistics.
	"""
	df = pd.read_csv(filename)
	
	for course in courses:
		if course not in df.columns:
			raise ValueError(f"Course '{course}' is missing from the dataset")
	
	stats = []
	
	kl_matrixes = get_KL_matrixes(df)

	for course in courses:
		values = df[course].to_list()
		col_stats = {
			'Feature': course,
			'Count': ft_count(values),
			'Mean': ft_mean(values),
			'Std': ft_std(values),
			'Min': ft_min(values),
			'25%': ft_percentile(values, 0.25),
			'50%': ft_percentile(values, 0.50),
			'75%': ft_percentile(values, 0.75),
			'Max': ft_max(values),
			'Homogeneity': kl_matrixes[course].mean().mean()
		}
		stats.append(col_stats)
	
	result = pd.DataFrame(stats)
	result = result.set_index('Feature')
	
	pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
	return result

def main():
	parser = argparse.ArgumentParser(description='Describe the dataset.')
	parser.add_argument('--dataset', type=str, default='datasets/dataset_train.csv', help='Path to the dataset file')
	parser.add_argument('--courses', type=str, nargs='+', default=['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'], help='List of courses to plot')
	args = parser.parse_args()
	
	print(describe(args.dataset, args.courses))

if __name__ == '__main__':
	main()