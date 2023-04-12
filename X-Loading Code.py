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
var_names = list(X.columns)

plt.plot(var_names, x_loading_pc1)
plt.xticks(rotation=90)
plt.xlabel('Variable')
plt.ylabel('X-Loading')
plt.title('X-Loading pre PC1')
plt.show()