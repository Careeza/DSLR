import numpy as np
from scipy.stats import gaussian_kde, entropy
import pandas as pd

def ft_count(L):
	"""
	Counts the number of non-NaN elements in the list.

	Parameters:
	L (list): The list of values.

	Returns:
	int: The count of non-NaN elements.
	"""
	tot = 0
	for l in L:
		if not np.isnan(l):
			tot += 1
	return tot

def ft_sum(L):
	"""
	Calculates the sum of non-NaN elements in the list.

	Parameters:
	L (list): The list of values.

	Returns:
	float: The sum of non-NaN elements.
	"""
	tot = 0
	for l in L:
		if not np.isnan(l):
			tot += l
	return tot

def ft_mean(L):
	"""
	Calculates the mean of non-NaN elements in the list.

	Parameters:
	L (list): The list of values.

	Returns:
	float: The mean of non-NaN elements, or NaN if the list is empty.
	"""
	count = ft_count(L)
	if count == 0:
		return np.nan
	return ft_sum(L) / ft_count(L)

def ft_std(L, ddof=0):
	"""
	Calculates the standard deviation of non-NaN elements in the list.

	Parameters:
	L (list): The list of values.
	ddof (int): Delta degrees of freedom.

	Returns:
	float: The standard deviation of non-NaN elements, or NaN if the list is empty.
	"""
	mean = ft_mean(L)
	if np.isnan(mean):
		return np.nan
	tot = 0
	count = 0
	for l in L:
		if not np.isnan(l):
			tot += (l - mean)**2
			count += 1
	return (tot/(count - ddof))**0.5

def ft_max(L):
	"""
	Finds the maximum value among non-NaN elements in the list.

	Parameters:
	L (list): The list of values.

	Returns:
	float: The maximum value, or None if the list is empty.
	"""
	maxi = None
	for l in L:
		if not np.isnan(l):
			if maxi is None or l > maxi:
				maxi = l
	return maxi

def ft_min(L):
	"""
	Finds the minimum value among non-NaN elements in the list.

	Parameters:
	L (list): The list of values.

	Returns:
	float: The minimum value, or None if the list is empty.
	"""
	mini = None
	for l in L:
		if not np.isnan(l):
			if mini is None or l < mini:
				mini = l
	return mini

def ft_percentile(L, p):
	"""
	Calculates the p-th percentile of non-NaN elements in the list.

	Parameters:
	L (list): The list of values.
	p (float): The percentile to calculate (between 0 and 1).

	Returns:
	float: The p-th percentile value, or NaN if the list is empty or p is out of range.
	"""
	L = sorted([l for l in L if not np.isnan(l)])
	if len(L) == 0:
		return np.nan
	if p < 0 or p > 1:
		return np.nan
	index = (len(L) - 1) * p
	if index.is_integer():
		return L[int(index)]
	else:
		return (L[int(index)] * (1 - index % 1) + L[int(index) + 1] * (index % 1))
	

def get_KL_matrixes(df, courses=["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]):
	for course in courses:
		df[course] = (df[course] - df[course].mean()) / df[course].std()

	kl_matrixes = {}
	for course in courses:
		slytherin_grades = df[df["Hogwarts House"] == "Slytherin"][course].dropna().to_numpy()
		ravenclaw_grades = df[df["Hogwarts House"] == "Ravenclaw"][course].dropna().to_numpy()
		gryffindor_grades = df[df["Hogwarts House"] == "Gryffindor"][course].dropna().to_numpy()
		hufflepuff_grades = df[df["Hogwarts House"] == "Hufflepuff"][course].dropna().to_numpy()

		kde1 = gaussian_kde(slytherin_grades)
		kde2 = gaussian_kde(ravenclaw_grades)
		kde3 = gaussian_kde(gryffindor_grades)
		kde4 = gaussian_kde(hufflepuff_grades)

		x_values = np.linspace(-1, 1, 1000)

		pdf1 = kde1(x_values)
		pdf2 = kde2(x_values)
		pdf3 = kde3(x_values)
		pdf4 = kde4(x_values)

		pdf1 /= pdf1.sum()
		pdf2 /= pdf2.sum()
		pdf3 /= pdf3.sum()
		pdf4 /= pdf4.sum()

		kl_matrix = pd.DataFrame(
			np.zeros((4, 4)),
			columns=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"],
			index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]
		)

		grades = [pdf1, pdf2, pdf3, pdf4]

		for i in range(4):
			for j in range(4):
				if i != j:
					kl_matrix.iloc[i, j] = entropy(grades[i], grades[j])  # KL Divergence
				else:
					kl_matrix.iloc[i, j] = 0

		kl_matrixes[course] = kl_matrix

	return kl_matrixes