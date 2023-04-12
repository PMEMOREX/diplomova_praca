# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:36:23 2023

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
pca = PCA(n_components=3)
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_

xlabel_loadings = pca.components_
xlabel_variable_contributions = np.square(xlabel_loadings) * explained_variance_ratio[:, np.newaxis]
xlabel_variable_importance = pd.DataFrame(xlabel_variable_contributions.sum(axis=0), index=df.columns, columns=["Importance"])
xlabel_variable_importance = xlabel_variable_importance.sort_values(by="Importance", ascending=False)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
variable_importance = np.square(loadings).sum(axis=1)
variable_importance_sorted = sorted(variable_importance, reverse=True)
variable_importance = pd.DataFrame(variable_importance_sorted, index=xlabel_variable_importance.index, columns=["Importance"])

import matplotlib.pyplot as plt
plt.bar(xlabel_variable_importance.index, variable_importance["Importance"])
plt.xlabel("Variable")
plt.ylabel("Power")
plt.title("Variable Importance (3 PCA)")
plt.xticks(rotation=90)
plt.show()