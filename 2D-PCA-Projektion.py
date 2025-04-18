import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Lade die Cluster-Zuordnung
cluster_df = pd.read_csv("./Quanten_Univers_analyse/clusterzuordnung.csv")

# Lade die Distanzmatrix
dist_matrix = pd.read_csv("./Quanten_Univers_analyse/vektorvergleich_top500.csv")

# Extrahiere eindeutige Dateinamen
all_files = sorted(set(dist_matrix["Datei 1"]) | set(dist_matrix["Datei 2"]))

# Initialisiere leere symmetrische Matrix
import numpy as np
matrix = pd.DataFrame(index=all_files, columns=all_files, dtype=float)
for _, row in dist_matrix.iterrows():
    f1, f2, dist = row["Datei 1"], row["Datei 2"], float(row["Distanz"])
    matrix.loc[f1, f2] = dist
    matrix.loc[f2, f1] = dist
np.fill_diagonal(matrix.values, 0.0)

# PCA erfordert vollständige Matrix, fehlende Werte füllen
matrix.fillna(matrix.mean(), inplace=True)

# PCA auf Distanzmatrix anwenden
pca = PCA(n_components=2)
coords = pca.fit_transform(matrix)

# Ergebnisse in DataFrame speichern
pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
pca_df["Datei"] = matrix.index
merged_df = pd.merge(pca_df, cluster_df, on="Datei", how="left")

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=merged_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
plt.title("2D-PCA der Distanzmatrix (farblich nach Cluster)")
plt.tight_layout()
plt.savefig("/mnt/data/pca_cluster_visualisierung.png", dpi=150)
plt.show()
