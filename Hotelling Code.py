# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 23:21:31 2023

@author: peter
"""

import pandas as pd
import numpy as np
from scipy.stats import f

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
pca = PCA(n_components=4)
pca.fit(X_scaled)

scores = pca.transform(X_scaled)
n = len(df)
p = len(df.columns)
alpha = 0.05 # hladina významnosti
T2 = (n - 1) * p / (n - p) * f.ppf(1 - alpha, p, n - p)
T2_scores = np.sum(scores**2 / np.var(scores, axis=0), axis=1)
anomalie = np.where(T2_scores > T2)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(range(1, n+1)-5.5, T2_scores, 'bo', markersize=5)
ax.axhline(y=T2, color='r', linestyle='-')
ax.set_xlabel('Index')
ax.set_ylabel('Hotelling T2')
ax.set_title('Hotelling T2 graf')

# označenie anomálií na grafe
if len(anomalie[0]) > 0:
    ax.plot(anomalie[0]+1, T2_scores[anomalie], 'ro', markersize=5, label='Anomálie')
    ax.legend()

plt.show()