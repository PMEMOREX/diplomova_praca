# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:40:41 2023

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

pca = PCA()
pca.fit(X_scaled)

x_loading_pc1 = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
x_loading_pc2 = np.negative(pca.components_[1] * np.sqrt(pca.explained_variance_[1]))
var_names = list(X.columns)

plt.figure(figsize=(8, 8))
plt.scatter(x_loading_pc1, x_loading_pc2, s=100)
for i, txt in enumerate(var_names):
    plt.annotate(txt, (x_loading_pc1[i], x_loading_pc2[i]), fontsize=12)
plt.xlabel('X-loading pre komponent 1')
plt.ylabel('X-loading pre komponent 2')
plt.title('X-loading pre PC1 oproti X-loading PC2')
plt.show()