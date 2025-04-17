# -*- coding: utf-8 -*-
# Quantum Emergence Analyzer v4.0
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
# 4. Berechnung der Ähnlichkeit (Euklidische Distanz) zwischen den Hamming-Profilen verschiedener Aktivierungen.
# 5. Berechnung und Export der Sprungfrequenz (wie oft springt welches Modul?).
# 6. Erstellung einer Zeitachse aller detektierten Sprünge.
# 7. Analyse von Korrelationen (treten Sprünge in bestimmten Modulen oft gleichzeitig auf?).
#
# Die Ergebnisse werden als Plots und CSV-Dateien im OUTPUT_FOLDER gespeichert.

import os
import json
import matplotlib.pyplot as plt
import csv
from collections import defaultdict, Counter
from scipy.spatial.distance import euclidean # Für Vektorabstand
from itertools import combinations # Für Paarvergleiche
from datetime import datetime # Für Zeitstempel-Verarbeitung (optional)
import numpy as np # Für Berechnungen (mean, std)
import pandas as pd

# === Konfiguration ===
# Dieser Abschnitt definiert die Pfade und Parameter für die Analyse.

# LOG_FOLDER: Der Ordner, in dem die JSON-Logdateien des QuantumStepLogger liegen.
LOG_FOLDER = "./quantum_logs"

# OUTPUT_FOLDER: Der Ordner, in dem die Analyseergebnisse (Plots, CSVs) gespeichert werden.
# Er wird automatisch erstellt, falls er nicht existiert.
OUTPUT_FOLDER = "./Quanten_Univers_analyse"

# MIN_SHOTS: Die minimale Anzahl von Messungen (Shots) pro Logdatei,
# damit diese in die Analyse einbezogen wird (z.B. für Vektorvergleiche).
MIN_SHOTS = 2 # Mindestens 2 Shots nötig, um einen Sprung zu detektieren oder einen Vektor zu bilden

# SPRUNG_GRENZE: Die minimale absolute Änderung im Hamming-Gewicht zwischen
# zwei aufeinanderfolgenden Shots, die als signifikanter "Sprung" gewertet wird.
SPRUNG_GRENZE = 2 # Beispiel: Änderung von Hamming 1 auf 3 (Delta=2) gilt als Sprung

# Stelle sicher, dass der Ausgabeordner existiert
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Schritt 1: Logs laden ===
def load_all_logs(folder: str) -> tuple[defaultdict[str, list], defaultdict[str, list]]:
    """
    Lädt alle JSON-Logdateien aus dem angegebenen Ordner und extrahiert
    die Hamming-Gewichte und Zeitstempel der Messungen ('measurement' Einträge).

    Args:
        folder (str): Der Pfad zum Ordner mit den JSON-Logdateien.

    Returns:
        tuple[defaultdict[str, list], defaultdict[str, list]]:
            Ein Tupel enthält zwei Dictionaries:
            1. all_measurements: Dictionary, bei dem Schlüssel der Dateiname und
               Wert eine Liste der Hamming-Gewichte für diese Datei ist.
            2. all_timestamps: Dictionary, bei dem Schlüssel der Dateiname und
               Wert eine Liste der Zeitstempel für die Messungen ist.
    """
    all_measurements = defaultdict(list)
    all_timestamps = defaultdict(list)
    print(f"[Laden] Suche Logs in: {os.path.abspath(folder)}")
    log_files_found = 0
    measurement_entries_found = 0

    try:
        # Liste alle Dateien im Ordner auf
        for filename in os.listdir(folder):
            # Verarbeite nur JSON-Dateien, ignoriere eventuell reparierte Dateien
            if filename.endswith(".json") and "repaired" not in filename:
                log_files_found += 1
                filepath = os.path.join(folder, filename)
                try:
                    # Öffne und lese die JSON-Datei
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        file_measurements = 0
                        # Iteriere durch die Einträge in der JSON-Datei
                        for entry in data:
                            # Extrahiere nur Messungs-Einträge
                            if entry.get("type") == "measurement":
                                try:
                                    # Füge Hamming-Gewicht und Zeitstempel den Listen hinzu
                                    # Stelle sicher, dass Hamming-Gewicht eine Zahl ist
                                    hamming_weight = int(entry["hamming_weight"])
                                    all_measurements[filename].append(hamming_weight)
                                    # Füge Zeitstempel hinzu, falls vorhanden, sonst leeren String
                                    all_timestamps[filename].append(entry.get("timestamp", ""))
                                    file_measurements += 1
                                except (KeyError, ValueError, TypeError) as e:
                                    print(f"[Warnung] Ungültiger Measurement-Eintrag in {filename}: {e} - Eintrag: {entry}")
                        if file_measurements > 0:
                             measurement_entries_found += file_measurements
                             # print(f"[Laden] {file_measurements} Messungen aus {filename} geladen.") # Debugging
                except json.JSONDecodeError:
                    # Fehlerbehandlung für ungültige JSON-Dateien
                    print(f"[Fehler] Ungültiges JSON-Format in Datei: {filename}")
                except Exception as e:
                    # Fange andere Lese-/Verarbeitungsfehler ab
                    print(f"[Fehler] Konnte Datei nicht verarbeiten {filename}: {e}")
    except FileNotFoundError:
        print(f"[Fehler] Log-Ordner nicht gefunden: {folder}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim Laden der Logs: {e}")

    print(f"[Laden] {log_files_found} JSON-Dateien gefunden, {measurement_entries_found} Measurement-Einträge insgesamt extrahiert.")
    if not all_measurements:
        print("[Warnung] Keine gültigen Messdaten gefunden.")
    return all_measurements, all_timestamps

# === Schritt 2: Hamming-Verläufe plotten ===
def plot_hamming_profiles(hamming_data: defaultdict[str, list]):
    """
    Erstellt einen Linienplot, der die Verläufe der Hamming-Gewichte
    über die Shots für jede Logdatei visualisiert.

    Args:
        hamming_data (defaultdict[str, list]): Das Dictionary mit den Hamming-Gewichten,
                                              erzeugt von `load_all_logs`.
    """
    # Erstelle eine neue Matplotlib-Figur
    plt.figure(figsize=(14, 7)) # Etwas breiter für die Legende
    plot_count = 0

    # Iteriere durch alle Logdateien und ihre Hamming-Gewichtslisten
    for file, values in hamming_data.items():
        # Plotte nur, wenn genügend Shots vorhanden sind
        if len(values) >= MIN_SHOTS:
            plot_count += 1
            # Erzeuge ein Label für die Legende (gekürzter Dateiname)
            label = file.replace("quantum_log_", "").replace(".json", "")
            # Plotte die Werte gegen den Shot-Index (0, 1, 2, ...)
            plt.plot(range(len(values)), values, marker="o", linestyle='-', linewidth=1, markersize=4, label=label)

    if plot_count == 0:
        print("[Plot] Keine Daten mit genügend Shots zum Plotten der Hamming-Profile gefunden.")
        plt.close() # Schließe die leere Figur
        return

    # Konfiguriere den Plot
    plt.title("Hamming-Gewichte pro Shot (pro Aktivierung)")
    plt.xlabel("Shot Index")
    plt.ylabel("Hamming-Gewicht")
    plt.grid(True, alpha=0.6) # Gitter zur besseren Lesbarkeit
    # Platziere die Legende außerhalb des Plots, um Überlappung zu vermeiden
    plt.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1.02, 0.5))
    # Passe das Layout an, um Platz für die Legende zu schaffen
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # [left, bottom, right, top]

    # Speichere den Plot als PNG-Datei
    path = os.path.join(OUTPUT_FOLDER, "hamming_profile_plot.png")
    try:
        plt.savefig(path, dpi=150) # Höhere DPI für bessere Qualität
        print(f"[Info] Hamming-Profil Plot gespeichert: {path}")
    except Exception as e:
        print(f"[Fehler] Konnte Hamming-Plot nicht speichern: {e}")
    finally:
        # Schließe die Figur, um Speicher freizugeben
        plt.close()

# === Schritt 3: Sprungdetektion ===
def detect_spruenge(hamming_data: defaultdict[str, list]) -> defaultdict[str, list]:
    """
    Analysiert die Hamming-Gewichtsverläufe und identifiziert "Sprünge",
    d.h. Änderungen zwischen aufeinanderfolgenden Shots, die größer oder
    gleich SPRUNG_GRENZE sind.

    Args:
        hamming_data (defaultdict[str, list]): Das Dictionary mit den Hamming-Gewichten.

    Returns:
        defaultdict[str, list]: Ein Dictionary, bei dem Schlüssel der Dateiname ist
                               und Wert eine Liste von Tupeln ist. Jedes Tupel
                               repräsentiert einen Sprung und enthält:
                               (Shot-Index (wo der Sprung *endet*),
                                Hamming-Gewicht *vor* dem Sprung,
                                Hamming-Gewicht *nach* dem Sprung,
                                Absolute Differenz (Delta)).
    """
    spruenge = defaultdict(list)
    sprung_count_total = 0
    # Iteriere durch jede Datei und ihre Hamming-Werte
    for file, werte in hamming_data.items():
        # Iteriere durch die Werte, beginnend beim zweiten Shot (Index 1)
        for i in range(1, len(werte)):
            # Berechne die absolute Differenz zum vorherigen Shot
            delta = abs(werte[i] - werte[i - 1])
            # Prüfe, ob die Differenz die Sprunggrenze erreicht oder überschreitet
            if delta >= SPRUNG_GRENZE:
                # Füge den Sprung zum Dictionary hinzu
                spruenge[file].append((i, werte[i - 1], werte[i], delta))
                sprung_count_total += 1
    print(f"[Analyse] {sprung_count_total} Sprünge (Delta >= {SPRUNG_GRENZE}) in {len(spruenge)} Dateien detektiert.")
    return spruenge

# === Schritt 4: Vektordistanzen vergleichen ===
def compare_hamming_vectors(hamming_data: defaultdict[str, list]) -> list[tuple[str, str, float]]:
    """
    Vergleicht die Hamming-Gewichtsverläufe (als Vektoren) von allen
    Logdatei-Paaren mithilfe der Euklidischen Distanz. Ein kleinerer Abstand
    bedeutet ähnlichere Profile über die Shots hinweg.

    Args:
        hamming_data (defaultdict[str, list]): Das Dictionary mit den Hamming-Gewichten.

    Returns:
        list[tuple[str, str, float]]: Eine Liste von Tupeln, sortiert nach Distanz (aufsteigend).
                                      Jedes Tupel enthält:
                                      (Dateiname 1, Dateiname 2, Euklidische Distanz).
    """
    vergleich = []
    compared_pairs = 0
    # Erzeuge alle möglichen Paare von Logdateien (ohne Wiederholung)
    for (file1, vec1), (file2, vec2) in combinations(hamming_data.items(), 2):
        # Bestimme die minimale Länge der beiden Vektoren für den Vergleich
        min_len = min(len(vec1), len(vec2))
        # Führe Vergleich nur durch, wenn beide Vektoren mindestens MIN_SHOTS lang sind
        if min_len >= MIN_SHOTS:
            # Berechne die Euklidische Distanz zwischen den Vektoren (bis zur min_len)
            dist = euclidean(vec1[:min_len], vec2[:min_len])
            vergleich.append((file1, file2, dist))
            compared_pairs += 1
    # Sortiere die Vergleiche nach der Distanz (kleinste Distanz zuerst)
    vergleich.sort(key=lambda x: x[2])
    print(f"[Analyse] {compared_pairs} Hamming-Vektor-Paare verglichen.")
    return vergleich

# === Schritt 5a: Sprünge in CSV exportieren ===
def export_spruenge_csv(spruenge: defaultdict[str, list]):
    """
    Exportiert die detektierten Sprünge in eine CSV-Datei.

    Args:
        spruenge (defaultdict[str, list]): Das Dictionary mit den Sprungdaten,
                                          erzeugt von `detect_spruenge`.
    """
    path = os.path.join(OUTPUT_FOLDER, "sprunganalyse.csv")
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Schreibe die Kopfzeile der CSV
            writer.writerow(["Datei", "Shot", "Von", "Nach", "Delta"])
            # Iteriere durch alle Dateien mit Sprüngen
            for file, liste in spruenge.items():
                # Iteriere durch alle Sprünge in der Liste für diese Datei
                for shot, frm, to, delta in liste:
                    writer.writerow([file, shot, frm, to, delta])
        print(f"[Export] Sprunganalyse exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Sprunganalyse CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Sprünge: {e}")


# === Schritt 5b: Vektordistanzen exportieren ===
def export_vergleich_csv(vergleiche: list[tuple[str, str, float]]):
    """
    Exportiert die berechneten Vektordistanzen zwischen den Hamming-Profilen
    in eine CSV-Datei.

    Args:
        vergleiche (list[tuple[str, str, float]]): Die Liste der Vergleiche,
                                                   erzeugt von `compare_hamming_vectors`.
    """
    path = os.path.join(OUTPUT_FOLDER, "vektordistanzen.csv")
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Schreibe die Kopfzeile
            writer.writerow(["Datei 1", "Datei 2", "Distanz"])
            # Schreibe jede Zeile (Paar + Distanz)
            for f1, f2, dist in vergleiche:
                # Formatiere die Distanz für bessere Lesbarkeit
                writer.writerow([f1, f2, f"{dist:.4f}"])
        print(f"[Export] Vektordistanzen exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Vektordistanzen CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Vektordistanzen: {e}")

# === Schritt 5c: Sprungfrequenz pro Modul ===
def export_sprungfrequenz_summary(spruenge: defaultdict[str, list]):
    """
    Zählt, wie oft jedes Modul/jeder Knotentyp einen Sprung aufweist,
    und exportiert die Ergebnisse in eine CSV-Datei, sortiert nach Häufigkeit.

    Args:
        spruenge (defaultdict[str, list]): Das Dictionary mit den Sprungdaten.
    """
    counter = Counter()
    # Iteriere durch die Dateinamen im Sprünge-Dictionary
    for filename in spruenge:
        try:
            # Extrahiere den Modulnamen aus dem Dateinamen
            # Annahme: Format "quantum_log_MODULNAME_act_nqX_..."
            modulname = filename.replace("quantum_log_", "").split("_act_nq")[0]
            # Alternative Extraktion (robuster, falls "_act_nq" fehlt):
            # modulname = filename.replace("quantum_log_", "").replace(".json", "").split("_")[0]

            # Zähle die Anzahl der Sprünge für dieses Modul
            counter[modulname] += len(spruenge[filename])
        except IndexError:
             print(f"[Warnung] Konnte Modulnamen nicht aus '{filename}' extrahieren.")
             counter["Unbekanntes_Format"] += len(spruenge[filename])

    path = os.path.join(OUTPUT_FOLDER, "sprungfrequenz_module.csv")
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Schreibe Kopfzeile
            writer.writerow(["Modul", "Sprunganzahl"])
            # Schreibe die gezählten Frequenzen, sortiert nach Anzahl (absteigend)
            for modul, count in sorted(counter.items(), key=lambda item: item[1], reverse=True):
                writer.writerow([modul, count])
        print(f"[Export] Sprungfrequenzen exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Sprungfrequenz CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Sprungfrequenzen: {e}")

# === Schritt 5d: Zeitliche Sprungmuster exportieren ===
def export_sprung_timeline(spruenge: defaultdict[str, list], timestamps: defaultdict[str, list]):
    """
    Erstellt eine zeitlich geordnete Liste aller Sprünge unter Verwendung
    der Zeitstempel aus den Logdateien und exportiert sie in eine CSV-Datei.

    Args:
        spruenge (defaultdict[str, list]): Das Dictionary mit den Sprungdaten.
        timestamps (defaultdict[str, list]): Das Dictionary mit den Zeitstempeln
                                              der Messungen.
    """
    timeline = []
    # Iteriere durch alle Dateien, die Sprünge enthalten
    for file, liste_spruenge in spruenge.items():
        # Prüfe, ob Zeitstempel für diese Datei vorhanden sind
        if file in timestamps:
            timestamps_for_file = timestamps[file]
            # Iteriere durch jeden Sprung in dieser Datei
            for shot_index, frm, to, delta in liste_spruenge:
                # Hole den Zeitstempel für den entsprechenden Shot (Index 'i')
                # Der Sprung wird beim Shot-Index 'i' detektiert (zwischen i-1 und i)
                # Wir verwenden den Zeitstempel des Shots 'i', an dem der Sprung endet.
                ts = timestamps_for_file[shot_index] if shot_index < len(timestamps_for_file) else "Timestamp fehlt"
                # Füge die Informationen zur Timeline-Liste hinzu
                timeline.append((ts, file, shot_index, frm, to, delta))
        else:
            # Fallback, wenn keine Timestamps gefunden wurden
            for shot_index, frm, to, delta in liste_spruenge:
                timeline.append(("N/A", file, shot_index, frm, to, delta))

    # Sortiere die Timeline nach Zeitstempel
    # Behandle fehlende Timestamps ('N/A') als sehr früh oder sehr spät, z.B. durch Filtern oder spezielle Sortierlogik
    # Einfache Sortierung: 'N/A' kommt ans Ende oder Anfang je nach Python-Version/Verhalten
    timeline.sort(key=lambda x: x[0] if x[0] != "N/A" else "9999-12-31T23:59:59.999999Z") # Sortiere N/A ans Ende

    path = os.path.join(OUTPUT_FOLDER, "sprungzeitachse.csv")
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Schreibe Kopfzeile
            writer.writerow(["Timestamp", "Datei", "Shot", "Von", "Nach", "Delta"])
            # Schreibe die sortierten Timeline-Einträge
            for row in timeline:
                writer.writerow(row)
        print(f"[Export] Zeitachse der Sprünge exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Sprung-Zeitachsen CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Zeitachse: {e}")


# === Schritt 5e: Korrelation zwischen Modulen ===
def export_sprungkorrelationen(spruenge: defaultdict[str, list]):
    """
    Analysiert, welche Module dazu neigen, im *gleichen Shot-Index*
    (innerhalb ihrer jeweiligen Aktivierung) Sprünge aufzuweisen.
    Exportiert Paare von Modulen und die Anzahl der gemeinsamen Shot-Indizes,
    in denen beide einen Sprung hatten.

    Args:
        spruenge (defaultdict[str, list]): Das Dictionary mit den Sprungdaten.
    """
    # Sammle für jedes Modul die Set der Shot-Indizes, in denen es gesprungen ist
    module_shots = defaultdict(set)
    # KORRIGIERT: Iteriere über die Elemente des Dictionaries (Dateiname, Sprungliste)
    for file, liste in spruenge.items(): # <--- HIER WAR DER FEHLER (vorher filename)
        try:
            # Extrahiere Modulnamen aus dem korrekten Dateinamen 'file'
            modulname = file.replace("quantum_log_", "").split("_act_nq")[0]
            # Füge die Shot-Indizes (wo der Sprung endet) zum Set hinzu
            for shot_index, *_ in liste: # *_ ignoriert 'Von', 'Nach', 'Delta'
                module_shots[modulname].add(shot_index)
        except IndexError:
             print(f"[Warnung Korr] Konnte Modulnamen nicht aus '{file}' extrahieren.") # Verwende 'file' in Fehlermeldung

    korrelationen = []
    # Hole eine Liste der eindeutigen Modulnamen
    module_namen = list(module_shots.keys())
    # Vergleiche jedes Modulpaar genau einmal
    for i in range(len(module_namen)):
        for j in range(i + 1, len(module_namen)):
            m1, m2 = module_namen[i], module_namen[j]
            # Finde die Schnittmenge der Shot-Indizes (gemeinsame Sprung-Shots)
            gemeinsame_shots = module_shots[m1].intersection(module_shots[m2])
            # Wenn es gemeinsame Sprung-Shots gibt, speichere das Paar und die Anzahl
            if gemeinsame_shots:
                korrelationen.append((m1, m2, len(gemeinsame_shots)))

    # Sortiere die Korrelationen nach der Anzahl gemeinsamer Shots (absteigend)
    korrelationen.sort(key=lambda x: x[2], reverse=True)

    path = os.path.join(OUTPUT_FOLDER, "sprungkorrelationen.csv")
    try:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Schreibe Kopfzeile
            writer.writerow(["Modul 1", "Modul 2", "Gemeinsame Sprung-Shots"])
            # Schreibe die Korrelationsdaten
            for row in korrelationen:
                writer.writerow(row)
        print(f"[Export] Sprungkorrelationen exportiert nach: {path}")
    except IOError as e:
        print(f"[Fehler] Konnte Sprungkorrelationen CSV nicht schreiben: {e}")
    except Exception as e:
        print(f"[Fehler] Unerwarteter Fehler beim CSV-Export der Korrelationen: {e}")

# === Ergänzung: Sprung-Timing Analyse ===
def analyze_sprung_timing(csv_filepath: str, sprung_grenze: int, log_folder: str):
    """
    Analysiert, ob frühe Sprünge (zwischen Shot 0 und 1) das Verhalten
    in späteren Shots (Sprung zwischen Shot 1 und 2) beeinflussen. Lädt dazu
    die Original-Logs, um den vollständigen Shot-Verlauf zu erhalten.

    Args:
        csv_filepath (str): Pfad zur 'sprunganalyse.csv'-Datei.
        sprung_grenze (int): Der Schwellenwert, der einen Sprung definiert.
        log_folder (str): Pfad zum Ordner mit den originalen JSON-Logdateien.
    """
    if not os.path.exists(csv_filepath):
        print(f"Fehler: Datei nicht gefunden: {csv_filepath}")
        return

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Fehler beim Lesen der CSV '{csv_filepath}': {e}")
        return

    fruehe_spruenge_daten = defaultdict(list) # Speichert Delta für Shot 1->2 nach frühem Sprung
    spaete_spruenge_daten = defaultdict(list) # Speichert Delta für Shot 1->2 nach spätem Sprung (Shot 0->1 war ruhig)

    # Finde alle Dateien, die in der Sprunganalyse vorkommen
    relevant_files = df['Datei'].unique()
    all_logs = {} # Brauchen die originalen Hamming-Werte

    print("\n[Timing Analyse] Lade Original-Logs für vollständige Shot-Verläufe...")
    loaded_original_logs = 0
    for filename in relevant_files:
         filepath = os.path.join(log_folder, filename)
         if os.path.exists(filepath):
             try:
                 with open(filepath, "r", encoding="utf-8") as f:
                     data = json.load(f)
                     measurements = [int(entry["hamming_weight"]) for entry in data if entry.get("type") == "measurement" and "hamming_weight" in entry]
                     # Wir benötigen genau 3 Shots für diese spezifische Analyse
                     if len(measurements) == 3:
                         all_logs[filename] = measurements
                         loaded_original_logs += 1
                     # else:
                     #     print(f"[Timing Warnung] Log '{filename}' hat nicht genau 3 Messungen ({len(measurements)}), wird übersprungen.")
             except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                 print(f"[Timing Warnung] Konnte Log nicht laden/parsen: {filename} - {e}")
         # else:
         #     print(f"[Timing Warnung] Original-Log nicht gefunden: {filename}")

    if loaded_original_logs == 0:
        print("[Timing Analyse] Keine Original-Logs mit genau 3 Shots gefunden. Analyse abgebrochen.")
        return

    print(f"[Timing Analyse] Verarbeite {loaded_original_logs} Logs mit 3 Shots...")
    # Gehe die Original-Logs durch, um das Verhalten nach Sprüngen zu analysieren
    for datei, shots in all_logs.items():
        if len(shots) != 3: continue # Doppelte Sicherstellung

        h0, h1, h2 = shots[0], shots[1], shots[2]

        # Prüfe auf Sprung 0 -> 1
        delta_0_1 = abs(h1 - h0)
        sprung_0_1 = delta_0_1 >= sprung_grenze

        # Prüfe auf Sprung 1 -> 2
        delta_1_2 = abs(h2 - h1)
        sprung_1_2 = delta_1_2 >= sprung_grenze

        # Speichere Daten für die Analyse
        if sprung_0_1:
            # Es gab einen frühen Sprung, speichere das Delta des *nächsten* Schritts
            fruehe_spruenge_daten[datei].append(delta_1_2)
        elif sprung_1_2:
            # Es gab *keinen* frühen Sprung, aber einen späten.
            # Speichere das Delta des späten Sprungs selbst.
            spaete_spruenge_daten[datei].append(delta_1_2)

    # --- Auswertung ---
    print("\n--- Timing Analyse: Verhalten NACH einem frühen Sprung (Shot 0->1) ---")
    all_deltas_nach_frueh = [d for sublist in fruehe_spruenge_daten.values() for d in sublist]
    if all_deltas_nach_frueh:
        avg_delta_nach_frueh = np.mean(all_deltas_nach_frueh)
        std_delta_nach_frueh = np.std(all_deltas_nach_frueh)
        anzahl_stabil_nach_frueh = sum(1 for d in all_deltas_nach_frueh if d < sprung_grenze)
        print(f"Anzahl Fälle mit frühem Sprung (Shot 0->1): {len(all_deltas_nach_frueh)}")
        print(f"Durchschnittl. Delta (Shot 1->2) *nach* frühem Sprung: {avg_delta_nach_frueh:.2f}")
        print(f"Standardabw. Delta (Shot 1->2) *nach* frühem Sprung: {std_delta_nach_frueh:.2f}")
        print(f"Anzahl Fälle, die sich danach 'stabilisierten' (Delta < {sprung_grenze}): {anzahl_stabil_nach_frueh} ({anzahl_stabil_nach_frueh / len(all_deltas_nach_frueh) * 100:.1f}%)")
    else:
        print("Keine Daten für frühe Sprünge gefunden.")

    print("\n--- Timing Analyse: Späte Sprünge (Shot 1->2) OHNE frühen Sprung ---")
    all_deltas_spaet_ohne_frueh = [d for sublist in spaete_spruenge_daten.values() for d in sublist]
    if all_deltas_spaet_ohne_frueh:
        avg_delta_spaet_ohne_frueh = np.mean(all_deltas_spaet_ohne_frueh)
        std_delta_spaet_ohne_frueh = np.std(all_deltas_spaet_ohne_frueh)
        print(f"Anzahl Fälle mit spätem Sprung (Shot 1->2), aber KEINEM frühen: {len(all_deltas_spaet_ohne_frueh)}")
        print(f"Durchschnittl. Delta (Shot 1->2) bei spätem Sprung (ohne frühen): {avg_delta_spaet_ohne_frueh:.2f}")
        print(f"Standardabw. Delta (Shot 1->2) bei spätem Sprung (ohne frühen): {std_delta_spaet_ohne_frueh:.2f}")
    else:
        print("Keine Daten für späte Sprünge (ohne frühe) gefunden.")


# === Hauptfunktion ===
def run_analysis():
    """
    Führt die gesamte Analysepipeline aus:
    Logs laden, Plot erstellen, Analysen durchführen und Ergebnisse exportieren.
    """
    print("[Analyse gestartet] Lade Logs...")
    # Lade Hamming-Gewichte und Zeitstempel
    logs, timestamps = load_all_logs(LOG_FOLDER)

    # Breche ab, wenn keine Logs geladen werden konnten
    if not logs:
        print("[Abbruch] Keine gültigen Logdaten zum Analysieren gefunden.")
        return

    print(f"\n[Analyse] {len(logs)} Logdateien mit Messdaten werden verarbeitet.")

    # Führe die einzelnen Analyseschritte durch
    print("\n[Schritt 2] Erstelle Plot der Hamming-Profile...")
    plot_hamming_profiles(logs)

    print("\n[Schritt 3] Detektiere Sprünge...")
    spruenge = detect_spruenge(logs)

    print("\n[Schritt 4] Vergleiche Hamming-Vektoren...")
    vergleich = compare_hamming_vectors(logs)

    print("\n[Schritt 5] Exportiere Ergebnisse...")
    export_spruenge_csv(spruenge)
    export_vergleich_csv(vergleich)
    export_sprungfrequenz_summary(spruenge)
    export_sprung_timeline(spruenge, timestamps)
    export_sprungkorrelationen(spruenge)

    print("\n[Schritt 6] Führe Timing-Analyse durch...")
    sprunganalyse_csv_path = os.path.join(OUTPUT_FOLDER, "sprunganalyse.csv")
    analyze_sprung_timing(sprunganalyse_csv_path, SPRUNG_GRENZE, LOG_FOLDER)


    print("\n[Analyse abgeschlossen] Alle Ergebnisse im Ordner:", os.path.abspath(OUTPUT_FOLDER))

# === Skriptausführung ===
if __name__ == "__main__":
    # Dieser Block wird nur ausgeführt, wenn das Skript direkt gestartet wird.
    run_analysis()