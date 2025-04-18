# -*- coding: utf-8 -*-
# Distanzanalyse mit Heatmap & Hierarchischem Clustering

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# === Pfade ===
CSV_PATH = "./Quanten_Univers_analyse/vektorvergleich_top500.csv"
OUTPUT_FOLDER = "./Quanten_Univers_analyse"
HEATMAP_PATH = os.path.join(OUTPUT_FOLDER, "vektordistanzmatrix.png")
DENDROGRAM_PATH = os.path.join(OUTPUT_FOLDER, "clustering_dendrogramm.png")
CLUSTER_CSV_PATH = os.path.join(OUTPUT_FOLDER, "clusterzuordnung.csv")

# === Sicherstellen, dass Ausgabeordner existiert ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def perform_clustering(distance_matrix: np.ndarray, labels: list[str], num_clusters: int = 5):
    """
    Führt hierarchisches Clustering auf Basis der Distanzmatrix durch und plottet Dendrogramm + Clusterzuordnung.
    """
    print("[Clustering] Starte hierarchisches Clustering...")
    
    # Umwandlung der symmetrischen Matrix in condensed vector
    condensed = squareform(distance_matrix, checks=False)

    # Berechne Linkage-Matrix
    linkage_matrix = linkage(condensed, method="average")

    # Zeichne Dendrogramm
    plt.figure(figsize=(14, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title("Hierarchisches Clustering der Aktivierungsprofile (Dendrogramm)")
    plt.tight_layout()
    plt.savefig(DENDROGRAM_PATH, dpi=150)
    plt.close()
    print(f"[Clustering] Dendrogramm gespeichert: {DENDROGRAM_PATH}")

    # Clusterlabels extrahieren
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
    df_clusters = pd.DataFrame({"Datei": labels, "Cluster": cluster_labels})
    df_clusters.to_csv(CLUSTER_CSV_PATH, index=False)
    print(f"[Clustering] Clusterzuordnung exportiert: {CLUSTER_CSV_PATH}")


def load_distance_matrix_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Lädt die Vektorvergleichsdaten (Datei1, Datei2, Distanz) und erzeugt eine bereinigte symmetrische Distanzmatrix.

    Bereinigt automatisch NaN- und Inf-Werte, sodass die Matrix für Clustering nutzbar ist.
    """
    print(f"[Laden] Lade Vergleichsdaten aus: {csv_path}")
    df = pd.read_csv(csv_path)

    # Alle eindeutigen Dateinamen erfassen
    alle_dateien = sorted(set(df["Datei 1"]) | set(df["Datei 2"]))
    matrix = pd.DataFrame(index=alle_dateien, columns=alle_dateien, dtype=float)

    # Einfüllen der Distanzen (symmetrisch)
    for _, row in df.iterrows():
        f1, f2, dist = row["Datei 1"], row["Datei 2"], float(row["Distanz"])
        matrix.loc[f1, f2] = dist
        matrix.loc[f2, f1] = dist

    # Diagonale mit 0.0 setzen (Distanz zu sich selbst)
    np.fill_diagonal(matrix.values, 0.0)

    # --- NEU: Bereinigung ---
    matrix = matrix.astype(float)

    # NaN-Werte ersetzen (durch maximale bekannte Distanz als Platzhalter für "sehr unähnlich")
    max_dist = np.nanmax(matrix.values)
    matrix.fillna(max_dist, inplace=True)

    # Auch +/- Inf ersetzen (selten, aber sicher ist sicher)
    matrix.replace([np.inf, -np.inf], max_dist, inplace=True)

    print(f"[Laden] Distanzmatrix erfolgreich bereinigt. Größe: {matrix.shape}")
    return matrix



def plot_distance_heatmap(matrix: pd.DataFrame, output_path: str):
    """
    Erstellt eine Heatmap aus der Distanzmatrix und speichert sie.
    """
    print(f"[Plot] Erstelle Heatmap der Distanzmatrix...")
    plt.figure(figsize=(18, 16))
    sns.heatmap(matrix, cmap="magma", square=True, xticklabels=False, yticklabels=False,
                cbar_kws={"label": "Euklidische Distanz"})
    plt.title("Heatmap der Hamming-Vektordistanzen (Ähnlichkeit der Module)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Heatmap gespeichert: {output_path}")


def run_full_analysis():
    """
    Führt Heatmap-Visualisierung und Clustering der Top-500 Distanzmatrix durch.
    """
    print("[Analyse] Starte Distanzmatrix-Analyse...")

    # 1. Matrix laden
    matrix = load_distance_matrix_from_csv(CSV_PATH)

    # 2. Heatmap erzeugen
    plot_distance_heatmap(matrix, HEATMAP_PATH)

    # 3. Clustering durchführen
    perform_clustering(matrix.to_numpy(), labels=matrix.index.tolist(), num_clusters=5)

    print("\n[✔] Analyse abgeschlossen.")


# === Hauptausführung ===
if __name__ == "__main__":
    run_full_analysis()
