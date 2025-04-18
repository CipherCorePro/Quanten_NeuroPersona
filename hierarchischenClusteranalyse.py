import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# === Datei laden ===
csv_path = "./Quanten_Univers_analyse/vektorvergleich_top500.csv"
df = pd.read_csv(csv_path)

# === Distanzmatrix erstellen ===
alle_dateien = sorted(set(df["Datei 1"]) | set(df["Datei 2"]))
matrix = pd.DataFrame(index=alle_dateien, columns=alle_dateien, dtype=float)

for _, row in df.iterrows():
    f1, f2, dist = row["Datei 1"], row["Datei 2"], float(row["Distanz"])
    matrix.loc[f1, f2] = dist
    matrix.loc[f2, f1] = dist

np.fill_diagonal(matrix.values, 0.0)
matrix = matrix.fillna(np.nanmedian(matrix.values))

# === Clustering ===
condensed = squareform(matrix.values)
linkage_matrix = linkage(condensed, method="average")
cluster_labels = fcluster(linkage_matrix, 5, criterion="maxclust")
cluster_df = pd.DataFrame({"Datei": matrix.index, "Cluster": cluster_labels})
cluster_df_sorted = cluster_df.sort_values("Cluster")

# === Matrix nach Cluster sortieren ===
sorted_index = cluster_df_sorted["Datei"].values
matrix_sorted = matrix.loc[sorted_index, sorted_index]

# === Heatmap erzeugen ===
plt.figure(figsize=(18, 14))
sns.heatmap(matrix_sorted, cmap="magma", square=True, xticklabels=False, yticklabels=False, cbar_kws={"label": "Euklidische Distanz"})
plt.title("Cluster-sortierte Heatmap der Hamming-Vektordistanzen")
plt.tight_layout()

# === Clusterergebnisse speichern ===
cluster_df_sorted.to_csv("./Quanten_Univers_analyse/clusterzuordnung.csv", index=False)
print("[✔] Clusterzuordnung erfolgreich gespeichert unter: ./Quanten_Univers_analyse/clusterzuordnung.csv")

# === Plot anzeigen ===
plt.show()

