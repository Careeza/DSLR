import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ft_hist(df):
    # Define houses and their colors
    hogwarts_House = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors_dict = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }
    
    # List of courses to plot
    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
              'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
              'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # Plot histograms for each course
    for i, course in enumerate(courses):
        ax = plt.subplot(3, 5, i + 1)
        for house in hogwarts_House:
            df[df["Hogwarts House"] == house][course].hist(
                alpha=0.5, 
                label=house, 
                color=colors_dict[house]
            )
        plt.title(course)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read the training dataset
    df = pd.read_csv('datasets/dataset_train.csv')
    ft_hist(df)