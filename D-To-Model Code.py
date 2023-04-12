# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:14:59 2023

@author: peter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("../IndustrialEvaporator3.csv", sep=";")
df['Dewpoint'] = df['Dewpoint'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['Intake Temp'] = df['Intake Temp'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['In-Process Air Temp'] = df['In-Process Air Temp'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['Exhaust Temp'] = df['Exhaust Temp'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['Mass Air Flow'] = df['Mass Air Flow'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['Bed Temp'] = df['Bed Temp'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['Filter Pressure'] = df['Filter Pressure'].apply(lambda x: float(x.split()[0].replace(',', '.')))
df['Bed Pressure'] = df['Bed Pressure'].apply(lambda x: float(x.split()[0].replace(',', '.')))

X = df

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)
distances = np.sqrt(np.sum(np.square(X_scaled - pca.inverse_transform(principal_components)), axis=1))
plt.plot(distances)
plt.title('Vzdialenosť k modelu\nPočet komponentov je 3')
plt.xlabel('Prípad')
plt.ylabel('Vzdialenosť')
plt.show()
