# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:49:38 2023

@author: peter
"""

import pandas as pd
import numpy as np

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

from sklearn.decomposition import PCA
pca = PCA(n_components=7)
pca.fit(X_scaled)

import matplotlib.pyplot as plt
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_, 'o-', linewidth=2, color='blue')
plt.title('Graf vlastných hodnôt')
plt.xlabel('Komponent')
plt.ylabel('Vlastné hodnoty')
plt.show()