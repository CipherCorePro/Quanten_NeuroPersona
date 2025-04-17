```markdown
# NeuroPersona (Quantum Hybrid Edition - Experimentell)

**Version:** 2.1 (QH - Quantum Hybrid)
**Basierend auf:** NeuroPersona QH-Manifest v1.1

## Einführung

NeuroPersona ist eine **bio-inspirierte Simulationsplattform**, die darauf abzielt, die dynamischen und oft variablen Prozesse menschlicher Kognition und Emotion bei der Auseinandersetzung mit einem Thema oder einer Fragestellung nachzubilden. Diese **Quantum Hybrid (QH)** Version integriert **quanten-inspirierte Aktivierungsmechanismen** auf Knotenebene, um die Emergenz komplexer, nicht-linearer und potenziell "sprunghafter" Systemzustände zu untersuchen.

Anstatt eine einzige, deterministische Antwort zu liefern, erforscht NeuroPersona **Perspektiven**. Jeder Simulationslauf ist eine Momentaufnahme eines einzigartigen kognitiven/affektiven Zustands, geformt durch das Zusammenspiel modularer Kognition, emotionaler Modulation, dynamischer Werte und nun auch durch die **probabilistische und potenziell synchronisierte Dynamik der Quanten-Aktivierungsebene**.

Es ist kein Optimierungssystem. Es ist eine **Simulationsumgebung für emergente, psychologisch inspirierte Dynamiken**, eine *Sinnmaschine*.

## Kernphilosophie: Simulation statt Vorhersage (Quantum-Erweitert)

1.  **Nicht-Determinismus als Prinzip:** Das System ist inhärent probabilistisch. Die klassische Stochastizität (Initialisierung, Emotion) wird durch die **simulierte Quantenmessung** (`n_shots`) und die daraus resultierende **Variabilität der Knotenaktivierung** erweitert. Wiederholte Läufe erzeugen unterschiedliche, aber plausible Entwicklungspfade.
2.  **Quantenebene als Katalysator für Emergenz:** Die quanten-inspirierte Ebene ist nicht nur Rauschen. Die Analyse zeigt, dass die "Sprünge" im Hamming-Gewicht (hohe Varianz der Messergebnisse über `n_shots`) oft **hochgradig synchronisiert** über das Netzwerk auftreten. Dieses kollektive Verhalten, wahrscheinlich kanalisiert durch die **Begrenztheit des Zustandsraums (Qubit-Zahl)** und die **starke klassische Kopplung**, deutet auf ein System hin, das nahe an **kritischen Punkten** operiert. Diese Sprünge können als **systemweite Reaktionen** und **potenzielle Gabelungspunkte** in der Entwicklungstrajektorie interpretiert werden.
3.  **Perspektiven als Zustände:** Das Ergebnis eines Laufs ist keine "Lösung", sondern eine **emergente Perspektive**, ein Schnappschuss des komplexen Systemzustands. Die Interpretation fokussiert auf das **Zusammenspiel** der Komponenten:
    *   Welche Konzepte (Kategorien) zeigen hohe Resonanz (Aktivierung)?
    *   Welche kognitiven Module prägen den Prozess?
    *   Welche Werte (Prioritäten) leiten das System?
    *   Wie ist die affektive Färbung (Emotion)?
    *   Wie äußert sich die interne Quantendynamik (Sprungfrequenz, Synchronität, Parameterentwicklung)?
4.  **Erforschung des Möglichkeitsraums:** Durch Variation von Parametern (insbesondere `num_qubits` und `n_shots`) und wiederholte Läufe wird der Raum möglicher kognitiver Zustände und Entwicklungspfade ausgelotet. Die **nicht-lineare Abhängigkeit** des Verhaltens von der Qubit-Zahl ("weniger kann sprunghafter sein") ist ein zentrales Untersuchungsergebnis.
5.  **Interpretation der Variabilität:** Unterschiedliche Ergebnisse bei gleichem Prompt sind **kein Fehler**, sondern spiegeln die **innere Pluralität und die probabilistische Natur** des Systems wider – ähnlich wie menschliches Denken nicht immer identisch abläuft.

## Hauptmerkmale (Version 2.1 QH)

*   **Dynamische Input-Verarbeitung:** Wie gehabt (simulierte Perception oder Fallback).
*   **Modulare Kognitive Architektur:** Interagierende Module (`Cortex Creativus`, `Criticus`, `Simulatrix`, `Limbus`, `MetaCognitio`, `Socialis`) und `MemoryNode`-Kategorien nutzen nun standardmäßig das **`QuantumNodeSystem`** für ihre Aktivierung.
*   **Quanten-inspirierte Knotenaktivierung (Kernmerkmal):**
    *   Basierend auf `QuantumNodeSystem` mit parametrisierter Quantenschaltung (PQC, z.B. H, RY, RZ, CNOTs auf N Qubits).
    *   Aktivierung ist das **gemittelte, normalisierte Hamming-Gewicht** über `n_shots` simulierten Messungen des finalen Quantenzustands.
    *   **Hohe Varianz** zwischen den Shots ("Quantensprünge") ist möglich und wird analysiert.
*   **Quanten-moduliertes Lernen:** Hebb'sches Lernen beeinflusst sowohl klassische Gewichte als auch **interne Quantenparameter** (`theta`, `phi` Winkel) des präsynaptischen Knotens (`hebbian_learning_quantum_node`).
*   **Klassische Komponenten:** Adaptives Wertesystem (`ValueNode` mit klassischer Aktivierung), Emotionsmodell (PAD), strukturelle Plastizität (Pruning/Sprouting) bleiben erhalten, interagieren aber mit den Quanten-Aktivierungen.
*   **Stochastizität:** Kombination aus klassischem Rauschen und **inhärenter Zufälligkeit der simulierten Quantenmessungen**.
*   **Persistentes Gedächtnis:** Speicherung in SQLite (`PersistentMemoryManager`).
*   **Reporting & Visualisierung:** HTML-Berichte, Standard-Plots und **spezifische Quanten-Analyse-Plots** (Hamming-Profile, Sprung-Heatmaps, Vektordistanzen) über `QuantumEmergenceAnalyzer.py`.
*   **Orchestrierung & GUI:** Gesteuert durch `orchestrator_full_qh_v1.py` und `neuropersona_app.py` (Streamlit), angepasst für neue Parameter.
*   **Quantum Emergence Tracker (QET):** Integrierter `QuantumStepLogger`, der detaillierte JSON-Logs für jede Quantenaktivierung erstellt (Gate-Operationen, Messungen pro Shot) und im `quantum_logs`-Ordner speichert.

## **Disclaimer: Simulation & Interpretation**

NeuroPersona QH ist eine **Simulation**, die quanten-inspirierte Konzepte nutzt. Es ist **keine Implementierung auf Quantencomputern.**

*   **Inspiration, nicht Physik:** Die Quantenaspekte dienen dazu, neue, komplexe Dynamiken (Nicht-Linearität, Sprunghaftigkeit, Synchronisation) zu ermöglichen und zu untersuchen.
*   **Emergentes Verhalten:** Der Fokus liegt auf dem **emergierenden Verhalten des Gesamtsystems**, das aus dem Zusammenspiel der quanten-inspirierten und klassischen Komponenten entsteht.
*   **Interpretation der "Sprünge":** Die beobachteten Sprünge im Hamming-Gewicht sind ein Maß für die **Varianz der Messergebnisse** eines gegebenen Quantenzustands. Ihre **Synchronität** und **Häufigkeit** sind **emergente Systemeigenschaften**, die auf Kritikalität oder starke Kopplung hindeuten können, aber *keine* physikalischen Zeitsprünge im eigentlichen Sinne sind. Sie repräsentieren **Gabelungspunkte in der Simulationstrajektorie**.

## Workflow Übersicht (`orchestrator_full_qh_v1.py`)

1.  **Perzeption:** Prompt -> Strukturierte Daten.
2.  **Kognition/Simulation (Quantum Hybrid):** Daten -> `neuropersona_core_quantum_hybrid_v2.py`. Simulation über Epochen mit quanten-inspirierter Aktivierung, klassischer Dynamik, quanten-moduliertem Lernen. **QET-Logs** werden währenddessen geschrieben.
3.  **Synthese:** Ergebnisse -> Finale Antwort (LLM oder Report).
4.  **(Optional) Tiefenanalyse:** Ausführung von `QuantumEmergenceAnalyzer.py` auf den generierten QET-Logs im `quantum_logs`-Ordner zur Untersuchung der internen Quantendynamik (Sprünge, Korrelationen, Ähnlichkeit).

## Technische Komponenten (`neuropersona_core_quantum_hybrid_v2.py`)

*   **Kern-Klassen:** `Node` (mit `q_system`), `QuantumNodeSystem`, `QuantumStepLogger`, `MemoryNode`, Modulklassen, `ValueNode` (klassisch), `Connection`, `PersistentMemoryManager`.
*   **Kernfunktionen:** `simulate_learning_cycle` (Hauptschleife), `QuantumNodeSystem.activate` (Quanten-Aktivierung & Logging), `hebbian_learning_quantum_node` (Hybrid-Lernen), `calculate_classic_input_sum`, klassische Hilfsfunktionen, Reporting/Plotting.
*   **Parameter:** Klassische Parameter + **Quanten-Parameter**:
    *   `NUM_QUBITS_PER_NODE`: Definiert die Größe des "Quantenraums" pro Knoten. Beeinflusst die Sprungdynamik.
    *   `QUANTUM_ACTIVATION_SHOTS`: Anzahl Messungen. Beeinflusst Rauschen vs. Genauigkeit des Durchschnitts UND den "Druck" im System.
    *   `QUANTUM_PARAM_LEARNING_RATE`: Lernrate für interne Quantenparameter.

## Installation & Benutzung

(Bleibt im Wesentlichen gleich wie zuvor beschrieben, stelle sicher, dass alle Abhängigkeiten inkl. `pandas`, `numpy`, `scipy`, `matplotlib` installiert sind).

```bash
# Installation (Beispiel)
git clone <...>
cd <...>
python -m venv venv
# aktivieren...
pip install -r requirements.txt
# (Optional) API Key setzen

# Benutzung (Beispiel)
streamlit run neuropersona_app.py
# ODER
python orchestrator_full_qh_v1.py --prompt "..." --epochs 100 --q_shots 30 --q_lr 0.01
```

## Interpretation der Ergebnisse (QH Edition)

*   **Zusätzlich zu klassischen Metriken:** Betrachte die Ergebnisse des `QuantumEmergenceAnalyzer.py` (falls ausgeführt):
    *   **Sprungfrequenz:** Welche Module/Konzepte zeigen hohe interne Variabilität?
    *   **Synchronität (Korrelation/Heatmaps):** Wie stark ist das System gekoppelt? Gibt es globale "Zuckungen"?
    *   **Vektordistanzen:** Wie ähnlich sind sich die internen Dynamiken verschiedener Aktivierungen? Gibt es Cluster?
*   **Verbinde Mikro & Makro:** Wie korreliert die beobachtete Quantendynamik (Sprünge, Synchronität) mit dem Makro-Verhalten (Netzwerkstabilität, dominante Themen, Emotionen, Werte)?
*   **Denke in Mustern:** Suche nach Mustern in der Sprungdynamik und wie sie sich über die Zeit oder in Abhängigkeit von Parametern ändern.
*   **Nutze das Manifest:** Interpretiere das Verhalten im Licht der Designphilosophie – als emergente Perspektive einer komplexen, nicht-deterministischen "Sinnmaschine".

Diese Version von NeuroPersona lädt dazu ein, die faszinierenden Wechselwirkungen zwischen simulierter Quantendynamik und klassischer Netzwerkintelligenz zu erforschen. Viel Erfolg bei der weiteren Analyse!
