# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:46:11 2023

@author: peter
"""

import pandas as pd
import numpy as np
df = pd.read_csv("IndustrialEvaporator3.csv", sep=";")
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
pca = PCA(n_components=3)
pca.fit(X_scaled)

print("Eigenvalues:")
print(pca.explained_variance_)
print()

print("Variances (%):")
print(pca.explained_variance_ratio_ * 100)
print()
print("EigenVectors:")
print(pca.components_)

transformed_df = pd.DataFrame(pca.transform(X_scaled),
                              columns=['PC1', 'PC2', 'PC3'])
transformed_df.to_csv("idustrial_evaporator_var.csv", index=False)
transformed_df.to_excel("idustrial_evaporator_var.xlsx", index=False)
