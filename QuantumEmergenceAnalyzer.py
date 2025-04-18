# -*- coding: utf-8 -*-
# Quantum Emergence Analyzer v4.1
# ===============================
#
# Dieses Skript analysiert die detaillierten Quanten-Logs (JSON-Dateien),
# die vom QuantumStepLogger in der NeuroPersona-Simulation erzeugt wurden.
# Es führt eine mehrstufige Analyse durch, um Muster und Anomalien
# im Verhalten der Quantenaktivierungen (insbesondere der Hamming-Gewichte
# über mehrere Shots hinweg) zu identifizieren.
#
# Analysepipeline:
# 1. Laden aller relevanten Log-Dateien.
# 2. Visualisierung der Hamming-Gewichtsverläufe pro Aktivierung (Shot-Profile).
# 3. Detektion von "Sprüngen" (signifikante Änderungen des Hamming-Gewichts zwischen Shots).
# 4. Berechnung der Ähnlichkeit (Euklidische Distanz) zwischen den Hamming-Profilen verschiedener Aktivierungen (mittels Truncation).
# 5. Visualisierung der Distanzmatrix als Heatmap.
# 6. Berechnung und Export der Sprungfrequenz (wie oft springt welches Modul?).
# 7. Erstellung einer Zeitachse aller detektierten Sprünge.
# 8. Analyse von Korrelationen (treten Sprünge in bestimmten Modulen oft gleichzeitig auf?).
# 9. Analyse des Sprung-Timings (Einfluss früher Sprünge auf spätere).
#
# Die Ergebnisse werden als Plots und CSV-Dateien im OUTPUT_FOLDER gespeichert.

import os
import json
import matplotlib.pyplot as plt
import csv
from collections import defaultdict, Counter
# Korrigierter Import: pdist und squareform hinzugefügt
from scipy.spatial.distance import euclidean, pdist, squareform
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns  # Hinzugefügt für bessere Heatmaps

# === Konfiguration ===
LOG_FOLDER = "./quantum_logs"
OUTPUT_FOLDER = "./Quanten_Univers_analyse"
MIN_SHOTS = 2 # Mindestens 2 Shots für Sprung/Vektor
SPRUNG_GRENZE = 2 # Minimale Delta-Änderung für einen Sprung
TOP_N_SIMILAR = 500 # Anzahl der ähnlichsten Paare für CSV-Export

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Schritt 1: Logs laden ===
def load_all_logs(folder: str) -> tuple[defaultdict[str, list], defaultdict[str, list]]:
    """
    Lädt alle JSON-Logdateien aus dem angegebenen Ordner und extrahiert
    die Hamming-Gewichte und Zeitstempel der Messungen ('measurement' Einträge).
    """
    all_measurements = defaultdict(list)
    all_timestamps = defaultdict(list)
    print(f"[Laden] Suche Logs in: {os.path.abspath(folder)}")
    log_files_found = 0
    measurement_entries_found = 0
    processed_files_count = 0

    try:
        filenames = [f for f in os.listdir(folder) if f.endswith(".json") and "repaired" not in f]
        log_files_found = len(filenames)
        print(f"[Laden] {log_files_found} potenzielle Log-Dateien gefunden.")

        for filename in filenames:
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_measurements = 0
                    temp_measurements = []
                    temp_timestamps = []
                    for entry in data:
                        if entry.get("type") == "measurement":
                            try:
                                hamming_weight = int(entry["hamming_weight"])
                                temp_measurements.append(hamming_weight)
                                temp_timestamps.append(entry.get("timestamp", ""))
                                file_measurements += 1
                            except (KeyError, ValueError, TypeError) as e:
                                print(f"[Warnung] Ungültiger Measurement-Eintrag in {filename}: {e} - Eintrag: {entry}")

                    # Füge Daten nur hinzu, wenn mindestens MIN_SHOTS vorhanden sind
                    if file_measurements >= MIN_SHOTS:
                        all_measurements[filename] = temp_measurements
                        all_timestamps[filename] = temp_timestamps
                        measurement_entries_found += file_measurements
                        processed_files_count += 1
                    # else:
                    #     print(f"[Info] Datei {filename} hat weniger als {MIN_SHOTS} Messungen ({file_measurements}), wird ignoriert.")

            except json.JSONDecodeError:
                print(f"[Fehler] Ungültiges JSON-Format in Datei: {filename}")
            except Exception as e:
                print(f"[Fehler] Konnte Datei nicht verarbeiten {filename}: {e}")
    except FileNotFoundError:
        print(f"[Fehler] Log-Ordner nicht gefunden: {folder}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim Laden der Logs: {e}")

    print(f"[Laden] {processed_files_count} Dateien mit >= {MIN_SHOTS} Messungen verarbeitet.")
    print(f"[Laden] {measurement_entries_found} Measurement-Einträge insgesamt in verarbeiteten Dateien.")
    if not all_measurements:
        print("[Warnung] Keine gültigen Messdaten mit ausreichenden Shots gefunden.")
    return all_measurements, all_timestamps


# === Schritt 2: Hamming-Verläufe plotten ===
def plot_hamming_profiles(hamming_data: defaultdict[str, list]):
    """
    Erstellt einen Linienplot, der die Verläufe der Hamming-Gewichte
    über die Shots für jede Logdatei visualisiert.
    """
    if not hamming_data:
        print("[Plot] Keine Hamming-Daten zum Plotten vorhanden.")
        return

    plt.figure(figsize=(14, 7))
    plot_count = 0
    for file, values in hamming_data.items():
        # Die Filterung nach MIN_SHOTS geschieht bereits beim Laden
        plot_count += 1
        label = file.replace("quantum_log_", "").replace(".json", "")
        plt.plot(range(len(values)), values, marker="o", linestyle='-', linewidth=1, markersize=3, alpha=0.7, label=label)

    if plot_count == 0:
        print("[Plot] Keine Daten zum Plotten der Hamming-Profile gefunden (nach Filterung).")
        plt.close()
        return

    plt.title(f"Hamming-Gewichte pro Shot (n={plot_count} Aktivierungen mit >= {MIN_SHOTS} Shots)")
    plt.xlabel("Shot Index")
    plt.ylabel("Hamming-Gewicht")
    plt.grid(True, alpha=0.6)
    # Nur Legende anzeigen, wenn nicht zu viele Linien da sind
    if plot_count <= 30:
        plt.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        print("[Plot] Zu viele Linien (>30) für eine übersichtliche Legende im Hamming-Plot.")
        plt.tight_layout()

    path = os.path.join(OUTPUT_FOLDER, "hamming_profile_plot.png")
    try:
        plt.savefig(path, dpi=150)
        print(f"[Info] Hamming-Profil Plot gespeichert: {path}")
    except Exception as e:
        print(f"[Fehler] Konnte Hamming-Plot nicht speichern: {e}")
    finally:
        plt.close()


# === Schritt 3: Sprungdetektion ===
def detect_spruenge(hamming_data: defaultdict[str, list]) -> defaultdict[str, list]:
    """
    Analysiert die Hamming-Gewichtsverläufe und identifiziert "Sprünge".
    """
    spruenge = defaultdict(list)
    sprung_count_total = 0
    for file, werte in hamming_data.items():
        # Iteriere durch die Werte, beginnend beim zweiten Shot (Index 1)
        for i in range(1, len(werte)):
            try:
                # Stelle sicher, dass Werte numerisch sind
                val_curr = int(werte[i])
                val_prev = int(werte[i - 1])
                delta = abs(val_curr - val_prev)
                if delta >= SPRUNG_GRENZE:
                    spruenge[file].append((i, val_prev, val_curr, delta))
                    sprung_count_total += 1
            except (ValueError, TypeError):
                print(f"[Warnung Sprung] Nicht-numerischer Wert in {file} bei Index {i} oder {i-1}. Überspringe Vergleich.")
                continue # Gehe zum nächsten Index
    print(f"[Analyse] {sprung_count_total} Sprünge (Delta >= {SPRUNG_GRENZE}) in {len(spruenge)} Dateien detektiert.")
    return spruenge


# === Schritt 4: Vektordistanzen vergleichen (Überarbeitet) ===
def compare_hamming_vectors_advanced(hamming_data: defaultdict[str, list]) -> tuple[np.ndarray | None, list[str]]:
    """
    Vergleicht Hamming-Gewichtsvektoren mittels Truncation auf Mindestlänge.
    Gibt die volle Distanzmatrix UND die zugehörigen Labels zurück.

    Args:
        hamming_data (defaultdict[str, list]): Dictionary mit Hamming-Gewichtsvektoren.

    Returns:
        tuple[np.ndarray | None, list[str]]:
            Die quadratische Distanzmatrix oder None bei Fehler,
            und die Liste der Dateinamen (Labels) für die Matrix.
    """
    # Filterung nach MIN_SHOTS wurde bereits beim Laden durchgeführt
    valid_files = sorted(hamming_data.keys())
    if len(valid_files) < 2:
        print("[Vergleich] Weniger als 2 gültige Vektoren für Distanzvergleich vorhanden.")
        return None, []

    vectors_raw = {f: hamming_data[f] for f in valid_files}

    # --- STRATEGIE: TRUNCATE ---
    try:
        # Finde die *kürzeste* Länge unter den gültigen Vektoren
        min_len = min(len(v) for v in vectors_raw.values())
        if min_len < MIN_SHOTS: # Sollte nicht passieren, aber sicher ist sicher
             print(f"[Vergleich Fehler] Mindestlänge {min_len} ist kleiner als MIN_SHOTS {MIN_SHOTS}.")
             return None, []
        print(f"[Vergleich] Kürze alle Vektoren auf Mindestlänge: {min_len} Shots")
        # Kürze alle Vektoren auf diese Mindestlänge und konvertiere zu float/int
        vectors_truncated = []
        valid_files_after_truncate = []
        for f in valid_files:
             try:
                 truncated_vector = [int(x) for x in vectors_raw[f][:min_len]]
                 vectors_truncated.append(np.array(truncated_vector))
                 valid_files_after_truncate.append(f)
             except (ValueError, TypeError) as e:
                 print(f"[Warnung Truncate] Konnte Vektor nicht konvertieren/kürzen für {f}: {e}. Überspringe.")

        if len(vectors_truncated) < 2:
             print("[Vergleich] Zu wenige Vektoren nach Truncation/Konvertierung übrig.")
             return None, []

        matrix = np.array(vectors_truncated)
        final_valid_labels = valid_files_after_truncate

    except ValueError: # Tritt auf, wenn vectors_raw leer ist
        print("[Vergleich Fehler] Keine gültigen Vektoren zum Bestimmen der Mindestlänge gefunden.")
        return None, []
    except Exception as e:
        print(f"[Vergleich Fehler] Unerwarteter Fehler beim Truncating: {e}")
        return None, []

    # Berechne paarweise Distanzen
    try:
        # Sicherstellen, dass die Matrix numerisch ist und keine NaNs/Infs enthält
        if not np.issubdtype(matrix.dtype, np.number):
            print("[Vergleich Fehler] Matrix ist nicht numerisch.")
            return None, []
        if np.isnan(matrix).any() or np.isinf(matrix).any():
            print("[Vergleich Fehler] Matrix enthält NaN oder Inf Werte NACH Truncation.")
            # Optional: Zeige problematische Zeilen
            # nan_rows = np.isnan(matrix).any(axis=1)
            # inf_rows = np.isinf(matrix).any(axis=1)
            # print("Zeilen mit NaN:", np.where(nan_rows)[0])
            # print("Zeilen mit Inf:", np.where(inf_rows)[0])
            return None, []

        dists = pdist(matrix, metric="euclidean")
        dist_matrix = squareform(dists)
        print(f"[Vergleich] Distanzmatrix ({dist_matrix.shape}) berechnet.")
        return dist_matrix, final_valid_labels
    except ValueError as ve:
         print(f"[Fehler] Fehler bei Distanzberechnung (pdist/squareform): {ve}")
         print("[Debug Info] Matrix Shape:", matrix.shape)
         print("[Debug Info] Matrix DType:", matrix.dtype)
         print(f"[Debug Info] Enthält NaN: {np.isnan(matrix).any()}")
         print(f"[Debug Info] Enthält Inf: {np.isinf(matrix).any()}")
         return None, []
    except Exception as e:
         print(f"[Fehler] Unerwarteter Fehler bei Distanzberechnung: {e}")
         return None, []

# === Schritt 5: Heatmap Plotten (Neu) ===
def plot_distance_heatmap(dist_matrix: np.ndarray | None, labels: list[str], filename: str = "distanz_heatmap.png"):
    """
    Plottet eine Heatmap der Distanzmatrix mit Achsenbeschriftungen.
    Sortiert optional die Achsen nach Modulnamen.

    Args:
        dist_matrix (np.ndarray | None): Quadratische Distanzmatrix oder None.
        labels (list[str]): Liste der Labels für die Achsen (aus valid_files).
        filename (str): Dateiname zum Speichern des Plots.
    """
    if dist_matrix is None or dist_matrix.size == 0 or not labels:
        print("[Plot Heatmap] Keine gültigen Daten für Heatmap.")
        return

    # Kürzere Labels für bessere Lesbarkeit (nur Modulname)
    short_labels = []
    module_names_for_sort = [] # Separate Liste für Sortierung
    for label in labels:
        try:
            modulname = label.replace("quantum_log_", "").split("_act_nq")[0]
            # Beispiel: Zeitstempel hinzufügen, falls benötigt
            # ts_part = label.split('_')[-2][:6] # Extrahiere JJMMTT
            # short_labels.append(f"{modulname}_{ts_part}")
            short_labels.append(modulname)
            module_names_for_sort.append(modulname)
        except Exception:
            fallback_label = label[:15] # Fallback: Kürze Label
            short_labels.append(fallback_label)
            module_names_for_sort.append(fallback_label)

    # --- Optional: Sortieren nach Modulnamen ---
    try:
        sort_indices = np.argsort(module_names_for_sort)
        dist_matrix_sorted = dist_matrix[sort_indices][:, sort_indices] # Sortiere Zeilen und Spalten
        sorted_short_labels = [short_labels[i] for i in sort_indices]
        print("[Plot Heatmap] Achsen nach Modulnamen sortiert.")
        plot_matrix = dist_matrix_sorted
        plot_labels = sorted_short_labels
    except Exception as sort_err:
        print(f"[Warnung Plot Heatmap] Sortierung fehlgeschlagen: {sort_err}. Verwende ursprüngliche Reihenfolge.")
        plot_matrix = dist_matrix
        plot_labels = short_labels
    # --- Ende Sortierung ---

    # Erstelle DataFrame mit den (sortierten) Daten
    df_heatmap = pd.DataFrame(plot_matrix, index=plot_labels, columns=plot_labels)

    # Figurgröße anpassen, je nach Anzahl Labels
    base_size = 8
    label_factor = 0.15 # Faktor pro Label
    max_fig_size = 30 # Maximale Größe begrenzen
    fig_size = min(max_fig_size, base_size + len(plot_labels) * label_factor)
    plt.figure(figsize=(fig_size + 2, fig_size)) # +2 für Colorbar

    # Schriftgröße für Achsen anpassen
    tick_fontsize = max(4, int(10 - len(plot_labels) * 0.05))

    # Seaborn Heatmap verwenden
    try:
        sns.heatmap(df_heatmap, cmap="magma_r", annot=False, # Keine Annotationen bei großen Matrizen
                    linewidths=0.1, linecolor='lightgrey', # Dezente Linien
                    cbar_kws={'label': 'Euklidische Distanz'})
        plt.xticks(rotation=90, fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)
        plt.title("Heatmap der Hamming-Vektordistanzen (Ähnlichkeit der Aktivierungsprofile)")
        plt.tight_layout() # Passt Layout an

        # Speichern
        path = os.path.join(OUTPUT_FOLDER, filename)
        plt.savefig(path, dpi=150)
        print(f"[Info] Distanz-Heatmap gespeichert: {path}")
    except Exception as e:
        print(f"[Fehler] Konnte Distanz-Heatmap nicht erstellen/speichern: {e}")
    finally:
        plt.close()


# === Schritt 6: Top-N Ähnlichkeiten exportieren (Angepasst) ===
def export_vergleich_top_n_csv(dist_matrix: np.ndarray | None, labels: list[str], top_n: int):
    """
    Extrahiert die Top-N ähnlichsten Vektorpaarungen aus der Distanzmatrix
    und exportiert sie in eine CSV-Datei.

    Args:
        dist_matrix (np.ndarray | None): Die quadratische Distanzmatrix.
        labels (list[str]): Die Liste der zugehörigen Labels (Dateinamen).
        top_n (int): Die Anzahl der ähnlichsten Paare, die exportiert werden sollen.
    """
    if dist_matrix is None or not labels or len(labels) != dist_matrix.shape[0]:
        print("[Export TopN] Ungültige Eingabedaten für Top-N Export.")
        return

    vergleich_liste = []
    num_labels = len(labels)
    # Extrahiere die untere Dreiecksmatrix (ohne Diagonale)
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            try:
                # Überprüfe auf NaN oder Inf in der Distanzmatrix
                dist = dist_matrix[i, j]
                if not np.isnan(dist) and not np.isinf(dist):
                    vergleich_liste.append((labels[i], labels[j], float(dist)))
                # else:
                #     print(f"[Warnung TopN] Ungültige Distanz ({dist}) zwischen {labels[i]} und {labels[j]}. Überspringe.")
            except IndexError:
                print(f"[Fehler TopN] Indexfehler beim Zugriff auf Distanzmatrix für Paar ({i}, {j}). Shape: {dist_matrix.shape}")
                continue
            except Exception as e:
                print(f"[Fehler TopN] Unerwarteter Fehler beim Extrahieren des Vergleichs ({i},{j}): {e}")
                continue


    # Sortiere nach kleinster Distanz
    try:
        vergleich_liste.sort(key=lambda x: x[2])
    except Exception as e:
        print(f"[Fehler TopN] Fehler beim Sortieren der Vergleichsliste: {e}")
        return # Abbruch, wenn Sortierung fehlschlägt

    top_vergleich = vergleich_liste[:top_n]

    path = os.path.join(OUTPUT_FOLDER, f"vektorvergleich_top{top_n}.csv")
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Datei 1", "Datei 2", "Distanz"])
            for f1, f2, dist in top_vergleich:
                writer.writerow([f1, f2, f"{dist:.4f}"])
        print(f"[Export] Vektordistanzen (Top {len(top_vergleich)}) exportiert nach: {path}")
    except Exception as e:
        print(f"[Fehler] Konnte Top-N Vergleichs-CSV nicht schreiben: {e}")

# === Schritt 7: Sprünge in CSV exportieren ===
def export_spruenge_csv(spruenge: defaultdict[str, list]):
    """Exportiert die detektierten Sprünge in eine CSV-Datei."""
    path = os.path.join(OUTPUT_FOLDER, "sprunganalyse.csv")
    exported_count = 0
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Datei", "Shot", "Von", "Nach", "Delta"])
            for file, liste in spruenge.items():
                for shot, frm, to, delta in liste:
                    writer.writerow([file, shot, frm, to, delta])
                    exported_count += 1
        print(f"[Export] {exported_count} Sprünge in Sprunganalyse exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Sprunganalyse CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Sprünge: {e}")

# === Schritt 8: Sprungfrequenz pro Modul ===
def export_sprungfrequenz_summary(spruenge: defaultdict[str, list]):
    """Zählt und exportiert die Sprungfrequenz pro Modul."""
    counter = Counter()
    unknown_format_count = 0
    for filename, sprung_liste in spruenge.items():
        num_spruenge_in_file = len(sprung_liste)
        try:
            modulname = filename.replace("quantum_log_", "").split("_act_nq")[0]
            counter[modulname] += num_spruenge_in_file
        except IndexError:
            print(f"[Warnung Frequenz] Konnte Modulnamen nicht aus '{filename}' extrahieren.")
            counter["Unbekanntes_Format"] += num_spruenge_in_file
            unknown_format_count += 1

    path = os.path.join(OUTPUT_FOLDER, "sprungfrequenz_module.csv")
    exported_modules = 0
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Modul", "Sprunganzahl"])
            # Sortiert nach Anzahl (absteigend)
            for modul, count in sorted(counter.items(), key=lambda item: item[1], reverse=True):
                writer.writerow([modul, count])
                exported_modules += 1
        print(f"[Export] Sprungfrequenzen für {exported_modules} Module exportiert nach: {path}")
        if unknown_format_count > 0:
            print(f"[Warnung Frequenz] {unknown_format_count} Einträge konnten keinem Modul zugeordnet werden (siehe 'Unbekanntes_Format' in CSV).")
    except IOError as e:
        print(f"[Fehler] Konnte Sprungfrequenz CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Sprungfrequenzen: {e}")

# === Schritt 9: Zeitliche Sprungmuster exportieren ===
def export_sprung_timeline(spruenge: defaultdict[str, list], timestamps: defaultdict[str, list]):
    """Erstellt und exportiert eine zeitlich geordnete Liste aller Sprünge."""
    timeline = []
    missing_ts_count = 0
    for file, liste_spruenge in spruenge.items():
        timestamps_for_file = timestamps.get(file) # Sicherer Zugriff
        if timestamps_for_file:
            for shot_index, frm, to, delta in liste_spruenge:
                ts = timestamps_for_file[shot_index] if shot_index < len(timestamps_for_file) else "Timestamp fehlt"
                if ts == "Timestamp fehlt": missing_ts_count += 1
                timeline.append((ts, file, shot_index, frm, to, delta))
        else:
            # Fallback, wenn keine Timestamps für die Datei existieren
            missing_ts_count += len(liste_spruenge)
            for shot_index, frm, to, delta in liste_spruenge:
                timeline.append(("N/A", file, shot_index, frm, to, delta))

    # Sortiere die Timeline nach Zeitstempel
    try:
        timeline.sort(key=lambda x: x[0] if isinstance(x[0], str) and x[0] not in ["N/A", "Timestamp fehlt"] else "9999-12-31T23:59:59.999999Z")
    except Exception as sort_err:
         print(f"[Warnung Timeline] Sortierung nach Zeitstempel fehlgeschlagen: {sort_err}. Liste bleibt unsortiert.")


    path = os.path.join(OUTPUT_FOLDER, "sprungzeitachse.csv")
    exported_count = 0
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Datei", "Shot", "Von", "Nach", "Delta"])
            for row in timeline:
                writer.writerow(row)
                exported_count += 1
        print(f"[Export] {exported_count} Sprünge in Zeitachse exportiert nach: {path}")
        if missing_ts_count > 0:
             print(f"[Warnung Timeline] Für {missing_ts_count} Sprünge fehlte der Zeitstempel.")
    except IOError as e:
        print(f"[Fehler] Konnte Sprung-Zeitachsen CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Zeitachse: {e}")


# === Schritt 10: Korrelation zwischen Modulen ===
def export_sprungkorrelationen(spruenge: defaultdict[str, list]):
    """Analysiert und exportiert, welche Module oft im gleichen Shot-Index springen."""
    module_shots = defaultdict(set)
    for file, liste in spruenge.items():
        try:
            modulname = file.replace("quantum_log_", "").split("_act_nq")[0]
            for shot_index, *_ in liste:
                module_shots[modulname].add(shot_index)
        except IndexError:
             # Wird bereits in Frequenz-Funktion geloggt, hier ggf. still bleiben
             pass

    korrelationen = []
    module_namen = list(module_shots.keys())
    for i in range(len(module_namen)):
        for j in range(i + 1, len(module_namen)):
            m1, m2 = module_namen[i], module_namen[j]
            # Ignoriere Vergleich eines Moduls mit sich selbst (sollte nicht passieren, aber sicher)
            if m1 == m2: continue
            gemeinsame_shots = module_shots[m1].intersection(module_shots[m2])
            if gemeinsame_shots:
                korrelationen.append((m1, m2, len(gemeinsame_shots)))

    korrelationen.sort(key=lambda x: x[2], reverse=True)

    path = os.path.join(OUTPUT_FOLDER, "sprungkorrelationen.csv")
    exported_count = 0
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Modul 1", "Modul 2", "Gemeinsame Sprung-Shots"])
            for row in korrelationen:
                writer.writerow(row)
                exported_count += 1
        print(f"[Export] {exported_count} Sprungkorrelations-Paare exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Sprungkorrelationen CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Korrelationen: {e}")


# === Schritt 11: Sprung-Timing Analyse ===
def analyze_sprung_timing(csv_filepath: str, sprung_grenze: int, log_folder: str):
    """
    Analysiert den Einfluss früher Sprünge (Shot 0->1) auf spätere (Shot 1->2).
    """
    if not os.path.exists(csv_filepath):
        print(f"[Timing Analyse Fehler] Eingabedatei nicht gefunden: {csv_filepath}")
        return

    try:
        # Lese nur die relevanten Dateinamen aus der Sprunganalyse
        df = pd.read_csv(csv_filepath, usecols=['Datei']).drop_duplicates()
        relevant_files = df['Datei'].tolist()
        if not relevant_files:
             print("[Timing Analyse] Keine Dateien in der Sprunganalyse gefunden.")
             return
    except Exception as e:
        print(f"[Timing Analyse Fehler] Fehler beim Lesen der CSV '{csv_filepath}': {e}")
        return

    fruehe_spruenge_nachfolge_delta = [] # Speichert Delta[1->2] wenn Sprung[0->1] == True
    spaete_spruenge_delta = []           # Speichert Delta[1->2] wenn Sprung[0->1] == False UND Sprung[1->2] == True

    print(f"\n[Timing Analyse] Lade Original-Logs für {len(relevant_files)} relevante Dateien...")
    loaded_3shot_logs = 0
    logs_with_3_shots = {} # Speicher für die geladenen Daten

    for filename in relevant_files:
         filepath = os.path.join(log_folder, filename)
         if os.path.exists(filepath):
             try:
                 with open(filepath, "r", encoding="utf-8") as f:
                     data = json.load(f)
                     measurements = [int(entry["hamming_weight"]) for entry in data if entry.get("type") == "measurement" and "hamming_weight" in entry]
                     if len(measurements) == 3:
                         logs_with_3_shots[filename] = measurements
                         loaded_3shot_logs += 1
             except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError) as e:
                 # OSError hinzugefügt, falls Datei nicht lesbar ist
                 print(f"[Timing Warnung] Konnte Log nicht laden/parsen: {filename} - {e}")
         # else: # Diese Meldung kann sehr lang werden, ggf. auskommentieren
         #     print(f"[Timing Warnung] Original-Log nicht gefunden: {filename}")

    if loaded_3shot_logs == 0:
        print("[Timing Analyse] Keine Original-Logs mit genau 3 Shots gefunden oder geladen. Analyse abgebrochen.")
        return

    print(f"[Timing Analyse] Verarbeite {loaded_3shot_logs} Logs mit genau 3 Shots...")
    # Analysiere die geladenen 3-Shot-Logs
    for datei, shots in logs_with_3_shots.items():
        h0, h1, h2 = shots[0], shots[1], shots[2]

        delta_0_1 = abs(h1 - h0)
        delta_1_2 = abs(h2 - h1)

        sprung_0_1 = delta_0_1 >= sprung_grenze
        sprung_1_2 = delta_1_2 >= sprung_grenze

        if sprung_0_1:
            # Es gab einen frühen Sprung, speichere das Delta des *nächsten* Schritts
            fruehe_spruenge_nachfolge_delta.append(delta_1_2)
        elif sprung_1_2:
            # Es gab *keinen* frühen Sprung, aber einen späten. Speichere das Delta des späten Sprungs.
            spaete_spruenge_delta.append(delta_1_2)
        # Else: Weder früher noch später Sprung ODER nur früher Sprung ohne späten -> nicht relevant für diese spezifische Analyse

    # --- Auswertung ---
    print("\n--- Timing Analyse: Verhalten NACH einem frühen Sprung (Shot 0->1) ---")
    if fruehe_spruenge_nachfolge_delta:
        n_frueh = len(fruehe_spruenge_nachfolge_delta)
        avg_delta_nach_frueh = np.mean(fruehe_spruenge_nachfolge_delta)
        std_delta_nach_frueh = np.std(fruehe_spruenge_nachfolge_delta)
        anzahl_stabil_nach_frueh = sum(1 for d in fruehe_spruenge_nachfolge_delta if d < sprung_grenze)
        print(f"Anzahl Fälle mit frühem Sprung (Shot 0->1): {n_frueh}")
        print(f"Durchschnittl. Delta (Shot 1->2) *nach* frühem Sprung: {avg_delta_nach_frueh:.2f}")
        print(f"Standardabw. Delta (Shot 1->2) *nach* frühem Sprung: {std_delta_nach_frueh:.2f}")
        if n_frueh > 0:
             print(f"Anzahl Fälle, die sich danach 'stabilisierten' (Delta < {sprung_grenze}): {anzahl_stabil_nach_frueh} ({anzahl_stabil_nach_frueh / n_frueh * 100:.1f}%)")
    else:
        print("Keine Daten für frühe Sprünge (gefolgt von Shot 2) gefunden.")

    print("\n--- Timing Analyse: Späte Sprünge (Shot 1->2) OHNE frühen Sprung ---")
    if spaete_spruenge_delta:
        n_spaet = len(spaete_spruenge_delta)
        avg_delta_spaet_ohne_frueh = np.mean(spaete_spruenge_delta)
        std_delta_spaet_ohne_frueh = np.std(spaete_spruenge_delta)
        print(f"Anzahl Fälle mit spätem Sprung (Shot 1->2), aber KEINEM frühen: {n_spaet}")
        print(f"Durchschnittl. Delta (Shot 1->2) bei spätem Sprung (ohne frühen): {avg_delta_spaet_ohne_frueh:.2f}")
        print(f"Standardabw. Delta (Shot 1->2) bei spätem Sprung (ohne frühen): {std_delta_spaet_ohne_frueh:.2f}")
    else:
        print("Keine Daten für späte Sprünge (ohne frühe) gefunden.")


# === Hauptfunktion ===
def run_analysis():
    """
    Führt die gesamte Analysepipeline aus.
    """
    print("[Analyse gestartet] Lade Logs...")
    logs, timestamps = load_all_logs(LOG_FOLDER)

    if not logs:
        print("[Abbruch] Keine gültigen Logdaten zum Analysieren gefunden.")
        return

    print(f"\n[Analyse] {len(logs)} Logdateien mit Messdaten werden verarbeitet.")

    print("\n[Schritt 2] Erstelle Plot der Hamming-Profile...")
    plot_hamming_profiles(logs)

    print("\n[Schritt 3] Detektiere Sprünge...")
    spruenge = detect_spruenge(logs)

    print("\n[Schritt 4] Vergleiche Hamming-Vektoren...")
    dist_matrix, matrix_labels = compare_hamming_vectors_advanced(logs)

    # Schritt 5 & 6: Heatmap plotten und Top-N exportieren (falls Distanzberechnung erfolgreich)
    if dist_matrix is not None and matrix_labels:
        print("\n[Schritt 5] Plotte Distanz-Heatmap...")
        plot_distance_heatmap(dist_matrix, matrix_labels, filename="distanz_heatmap_verbessert.png")

        print(f"\n[Schritt 6] Exportiere Top-{TOP_N_SIMILAR} ähnlichste Paare...")
        export_vergleich_top_n_csv(dist_matrix, matrix_labels, top_n=TOP_N_SIMILAR)
    else:
         print("[Info] Überspringe Heatmap-Plot und Top-N Vektorvergleich CSV (Fehler bei Distanzberechnung).")


    # Weitere Exporte (Schritte 7-10)
    print("\n[Schritte 7-10] Exportiere weitere Analyseergebnisse...")
    export_spruenge_csv(spruenge)
    # export_vergleich_csv(vergleich_liste) # Auskommentiert, da sehr groß und in top_n enthalten
    export_sprungfrequenz_summary(spruenge)
    export_sprung_timeline(spruenge, timestamps)
    export_sprungkorrelationen(spruenge)

    # Schritt 11: Timing Analyse
    print("\n[Schritt 11] Führe Timing-Analyse durch...")
    sprunganalyse_csv_path = os.path.join(OUTPUT_FOLDER, "sprunganalyse.csv")
    if os.path.exists(sprunganalyse_csv_path):
         analyze_sprung_timing(sprunganalyse_csv_path, SPRUNG_GRENZE, LOG_FOLDER)
    else:
         print("[Warnung Timing] 'sprunganalyse.csv' nicht gefunden. Überspringe Timing-Analyse.")


    print("\n[Analyse abgeschlossen] Alle Ergebnisse im Ordner:", os.path.abspath(OUTPUT_FOLDER))

# === Skriptausführung ===
if __name__ == "__main__":
    run_analysis()