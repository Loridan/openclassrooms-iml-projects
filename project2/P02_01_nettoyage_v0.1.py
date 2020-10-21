import numpy as np
import pandas as pd

# Veuillez telecharger la table dans le repertoire courant, attention size > 3Go
# url = 'https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv'

filename = 'en.openfoodfacts.org.products.csv'

# chargement des données en local
data = pd.read_table(filename, sep='\t', error_bad_lines=False, index_col=False, dtype='unicode')

# info
data.info()

# aperçu des premieres lignes
print(data.head())

