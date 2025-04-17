import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Pfade definieren ===
csv_path = "./Quanten_Univers_analyse/sprunganalyse.csv"
output_folder = os.path.dirname(csv_path)
png_output_path = os.path.join(output_folder, "sprunganalyse.png")
excel_output_path = os.path.join(output_folder, "sprunganalyse_matrix.xlsx")

# === CSV einlesen ===
df = pd.read_csv(csv_path)

# === Modul + Instanz extrahieren ===
df['Modul'] = df['Datei'].str.replace("quantum_log_", "").str.split("_act_nq").str[0]
df['Instanz'] = df['Modul'] + "_" + df['Datei'].str.extract(r"(_act_nq\d+_.+)\.json")[0]

# === Pivot-Tabelle erzeugen ===
pivot = df.pivot_table(index="Instanz", columns="Shot", aggfunc="size", fill_value=0)

# === In binäre Matrix umwandeln (0/1)
pivot[pivot > 1] = 1

# === Heatmap erzeugen ===
plt.figure(figsize=(20, max(8, len(pivot) * 0.25)))
sns.heatmap(
    pivot,
    cmap="YlGnBu",
    linewidths=0.1,
    linecolor="grey",
    cbar_kws={"label": "Anzahl Sprünge"},
    vmax=1
)
plt.title("Heatmap der synchronisierten Sprünge", fontsize=14)
plt.xlabel("Shot Index")
plt.ylabel("Modulinstanz")
plt.tight_layout()

# === Speichern der Heatmap ===
plt.savefig(png_output_path, dpi=150)
plt.close()

# === Export der Sprungmatrix als Excel-Datei ===
pivot.to_excel(excel_output_path)

print(f"[✔] Heatmap gespeichert unter: {os.path.abspath(png_output_path)}")
print(f"[✔] Excel-Tabelle gespeichert unter: {os.path.abspath(excel_output_path)}")
