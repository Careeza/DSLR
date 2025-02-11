"""
Ce module permet de visualiser la distribution des notes par matière pour chaque maison de Poudlard
sous forme d'histogrammes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def ft_hist(df, courses, save):
    """
    Crée des histogrammes montrant la distribution des notes pour chaque matière,
    séparés par maison de Poudlard.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données des étudiants avec leurs notes et maisons
    courses : list
        Liste des matières à visualiser
    save : bool
        Si True, sauvegarde le graphique dans 'histograms.png'
        Si False, affiche le graphique

    Returns
    -------
    None

    Notes
    -----
    Les histogrammes sont colorés selon un code couleur spécifique :
    - Gryffondor : rouge
    - Poufsouffle : jaune
    - Serdaigle : bleu
    - Serpentard : vert
    """
    # Define houses and their colors
    hogwarts_House = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors_dict = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }
    
    ncols = int(np.ceil(np.sqrt(len(courses))))
    nrows = int(np.ceil(len(courses) / ncols))
    # Create figure
    fig = plt.figure(figsize=(ncols * 5, nrows * 5))
    
    # Plot histograms for each course
    for i, course in enumerate(courses):
        ax = plt.subplot(nrows, ncols, i + 1)
        for house in hogwarts_House:
            df[df["Hogwarts House"] == house][course].hist(
                alpha=0.5, 
                label=house, 
                color=colors_dict[house]
            )
        plt.title(course)
        plt.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig("histograms.png")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise la distribution des notes par matière pour chaque maison de Poudlard')
    parser.add_argument('--dataset', type=str, default='datasets/dataset_train.csv', 
                       help='Chemin vers le fichier de données')
    parser.add_argument('--courses', type=str, nargs='+', 
                       default=['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                               'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                               'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'],
                       help='Liste des matières à visualiser')
    parser.add_argument('--save', action='store_true', 
                       help='Sauvegarde le graphique dans le fichier histograms.png')
    parser.add_argument('--homogeneous', action='store_true', 
                       help='Affiche uniquement la matière Arithmancy qui a une distribution homogène entre les maisons')
    args = parser.parse_args()
    
    # Lecture et traitement des données
    df = pd.read_csv(args.dataset)
    if args.homogeneous:
        courses = ['Arithmancy']
        ft_hist(df, courses, args.save)
    else:
        ft_hist(df, args.courses, args.save)