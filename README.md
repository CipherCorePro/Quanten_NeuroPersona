# NeuroPersona (Quantum Node Edition - Experimental)

**Version:** 2.0

## Einführung

NeuroPersona ist eine **bio-inspirierte Simulationsplattform**, die darauf abzielt, die dynamischen und oft variablen Prozesse menschlicher Kognition und Emotion bei der Auseinandersetzung mit einem Thema oder einer Fragestellung nachzubilden. Diese Version führt **experimentelle quanten-inspirierte Knoten (Quantum Nodes)** ein, die eine neue Ebene der dynamischen Aktivierung und des Lernens hinzufügen.

Anstatt eine einzige, deterministische Antwort zu liefern, erforscht NeuroPersona weiterhin **verschiedene plausible "Denkpfade" oder "Perspektiven"**. Jeder Simulationslauf stellt eine einzigartige Momentaufnahme eines möglichen kognitiven/affektiven Zustands dar, der nun auch durch simulierte quanten-ähnliche Mechanismen auf Knotenebene beeinflusst wird.

Das System modelliert interagierende kognitive Module (Kreativität, Kritik, Simulation, etc.), ein adaptives Wertesystem (klassisch), einen dynamischen emotionalen Zustand (PAD-Modell, klassisch beeinflusst) und neuronale Plastizität (strukturell und aktivitätsabhängig, mit quanten-moduliertem Feedback).

## Kernphilosophie: Simulation statt Vorhersage (Quantum-Erweitert)

Der grundlegende Ansatz bleibt erhalten, wird aber durch die Quanten-Knoten erweitert:

1.  **Variabilität als Feature:** Das System ist inhärent **nicht-deterministisch**. Wiederholte Läufe mit demselben Input führen zu unterschiedlichen Endzuständen. Dies liegt an Faktoren wie zufälliger Initialisierung, stochastischem Rauschen, Emotionsdynamik, pfadabhängigem Lernen UND nun zusätzlich an der **probabilistischen Natur der Quanten-Knoten-Aktivierung** (simulierte Messungen) und der **Dynamik interner Knotenparameter**.
2.  **Emergente Perspektiven:** Jeder Simulationslauf bleibt ein einzigartiger "Gedankengang". Das Ergebnis ist eine **simulierte, plausible Perspektive**, deren Entstehung nun auch von den quanten-inspirierten Aktivierungsdynamiken beeinflusst wird.
3.  **Interpretation des Zustands:** Das Ziel ist weiterhin, den **finalen kognitiven und affektiven Zustand** *innerhalb eines Laufs* zu verstehen:
    *   Welche Werte dominieren (klassisch)?
    *   Welche kognitiven Module sind besonders aktiv (quanten-aktiviert)?
    *   Wie ist die emotionale Grundstimmung (klassisch)?
    *   Welche internen Quantenparameter der Knoten sind signifikant geworden?
    *   Ist dieser Zustand in sich kohärent (z.B. hohe quantenbasierte Kreativitätsaktivierung korreliert mit hohem klassischem Innovationswert)?
    *   Auch scheinbare "Inkonsistenzen" bleiben valide Ergebnisse, die komplexe kognitive Zustände repräsentieren.
4.  **Erforschung des Möglichkeitsraums:** Durch Simulation verschiedener Läufe wird der *Raum möglicher kognitiver Reaktionen* ausgelotet, der nun durch die quanten-inspirierte Dynamik potenziell erweitert oder anders strukturiert ist.
5.  **Quantum Layer Interpretation:** Die Analyse berücksichtigt nun auch den Einfluss der simulierten Quantendynamik (interne Knotenparameter, Messeffekte durch `n_shots`) auf die beobachteten Knotenaktivierungen und das Netzwerkverhalten.

## Hauptmerkmale (Version 2.0)

*   **Dynamische Input-Verarbeitung:** Nutzt (simulierte) Perception Unit oder Fallback.
*   **Modulare Kognitive Architektur:** Simuliert interagierende Module (wie gehabt), deren Knoten nun **quanten-inspirierte Aktivierung** nutzen können (Ausnahme: `ValueNode`).
    *   `CortexCreativus`, `CortexCriticus`, `SimulatrixNeuralis`, `LimbusAffektus`, `MetaCognitio`, `CortexSocialis`: Nutzen `QuantumNodeSystem` für Aktivierung.
    *   `MemoryNode`: Nutzen ebenfalls `QuantumNodeSystem`.
*   **Quanten-inspirierte Knotenaktivierung (NEU):**
    *   Jeder aktivierte Knoten (außer `ValueNode`, Input-Nodes) verwendet eine `QuantumNodeSystem`-Instanz.
    *   Aktivierung basiert auf einer parametrisierten Quantenschaltung (PQC, z.B. H-RY-RZ auf einem Qubit).
    *   Die finale Aktivierung (0.0-1.0) ist die **geschätzte Wahrscheinlichkeit**, den Zustand |1> zu messen, basierend auf `n_shots` simulierten Messungen.
*   **Quanten-moduliertes Lernen (NEU):**
    *   Die Hebb'sche Lernregel aktualisiert nicht nur das **klassische Verbindungsgewicht**, sondern gibt auch Feedback an die **internen Parameter (z.B. Rotationswinkel) des präsynaptischen Quanten-Knotens**, wodurch dessen zukünftige Aktivierungsempfindlichkeit beeinflusst wird (`hebbian_learning_quantum_node`).
*   **Adaptives Wertesystem:** Interne Werte (`ValueNode`) bleiben **klassisch aktiviert** (0.0-1.0), beeinflussen das Verhalten und werden dynamisch angepasst.
*   **Neuronale Plastizität:** Simulation von strukturellen Änderungen (Pruning/Sprouting klassischer Verbindungen) und aktivitätsabhängigem Lernen (klassische Gewichte + Q-Param-Feedback).
*   **Stochastizität:** Gezielter Einsatz von Zufallselementen (Initialisierung, Emotionen) plus die **inhärente Stochastizität der simulierten Quantenmessungen**.
*   **Persistentes Gedächtnis:** Langfristige Speicherung (inkl. Q-Params) über SQLite (`PersistentMemoryManager`).
*   **Reporting & Visualisierung:** Generiert detaillierte HTML-Berichte und Plots, einschließlich der **Entwicklung interner Quantenparameter**.
*   **Orchestrierung:** Ein angepasstes `orchestrator_full_qh_v1.py` Skript steuert den Workflow.
*   **GUI:** Ein angepasstes `neuropersona_app.py` (Streamlit) Frontend ist verfügbar.

## **Disclaimer: Experimentelle Natur**

Diese Version von NeuroPersona ist **HOCH EXPERIMENTELL**. Die "Quantum Nodes" sind eine *Simulation* von quanten-inspirierten Prinzipien auf klassischer Hardware mittels Bibliotheken wie NumPy. Es handelt sich **nicht** um eine Implementierung auf echter Quantenhardware.

*   **Simulation, keine Realität:** Die Quantenmechanik wird stark vereinfacht und dient als Inspiration für neue Aktivierungs- und Lerndynamiken.
*   **Fokus auf Dynamik:** Der Nutzen liegt in der Erforschung neuartiger, potenziell komplexerer Netzwerkdynamiken, nicht in der exakten Abbildung quantenphysikalischer Prozesse.
*   **Interpretation:** Ergebnisse sollten als Resultate einer komplexen, stochastischen Simulation interpretiert werden, deren Verhalten durch die quanten-inspirierten Regeln beeinflusst wird. Vergleiche mit klassischen neuronalen Netzen oder echter Quanten-KI sind mit Vorsicht zu genießen.

## Workflow Übersicht (`orchestrator_full_qh_v1.py`)

1.  **Perzeption:** Nutzer-Prompt -> Strukturierte Daten (simuliertes CSV/DataFrame) (`gemini_perception_unit.py` oder Fallback).
2.  **Kognition/Simulation (Quantum):** Daten -> `neuropersona_core_quantum_hybrid_v2.py`. Netzwerk wird initialisiert. Über Epochen hinweg interagieren quanten-basierte Aktivierungen, klassische Wert-/Emotionsupdates, quanten-moduliertes Hebb'sches Lernen und Plastizität.
3.  **Synthese (Optional):** Simulationsergebnisse (Bericht, Daten) -> Finale, kontextualisierte Antwort über externe LLM-API (z.B. Gemini) oder direkter Report.

## Technische Komponenten (`neuropersona_core_quantum_hybrid_v2.py`)

*   **Kern-Klassen:**
    *   `Node`: Basisklasse, enthält optional `QuantumNodeSystem`.
    *   `QuantumNodeSystem`: Simuliert PQC und Messung für einen Knoten.
    *   `MemoryNode`, spezialisierte Modulklassen (erben von `Node`, nutzen oft `QuantumNodeSystem`).
    *   `ValueNode`: Erbt von `Node`, aber verwendet **klassische Aktivierung**.
    *   `Connection`: Stellt klassische Verbindungen mit Gewichten dar.
    *   `PersistentMemoryManager`.
*   **Kernfunktionen:**
    *   `simulate_learning_cycle`: Hauptsimulationsschleife (angepasst für Quanten-Aktivierung/-Lernen).
    *   `QuantumNodeSystem.activate`: Berechnet quanten-basierte Aktivierungswahrscheinlichkeit.
    *   `QuantumNodeSystem.update_internal_params`: Aktualisiert interne Qubit-Parameter.
    *   `hebbian_learning_quantum_node`: Aktualisiert klass. Gewicht UND Q-Params.
    *   `calculate_classic_input_sum`: Berechnet den klassischen Input für Quantenknoten.
    *   Klassische Funktionen (angepasst, um Quanten-Aktivierungen als Input zu nutzen): `calculate_value_adjustment`, `update_emotion_state`, `apply_reinforcement`, `prune_connections`, `sprout_connections`, etc.
    *   `generate_final_report`, `create_html_report`, Plotting-Funktionen (inkl. Q-Param-Plot).
*   **Parameter:** Klassische Parameter (`DEFAULT_LEARNING_RATE`, `DEFAULT_DECAY_RATE`, etc.) + **Neue Quanten-Parameter**:
    *   `QUANTUM_ACTIVATION_SHOTS`: Anzahl simulierter Messungen pro Aktivierung (Tradeoff Genauigkeit/Geschwindigkeit).
    *   `QUANTUM_PARAM_LEARNING_RATE`: Lernrate für die internen Parameter der Quanten-Knoten.

## Installation

1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd <repository-ordner> # z.B. NeuroPersona_Quantum
    ```
2.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # MacOS/Linux
    source venv/bin/activate
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    # Benötigt mindestens: pandas, numpy, matplotlib
    # Optional für volle Funktionalität: networkx, tqdm, google-generativeai, streamlit
    ```
4.  **API Key (Optional):** Für `orchestrator_full_qh_v1.py` mit finaler Gemini-Antwort oder Perception Unit:
    ```bash
    # Setze GEMINI_API_KEY als Umgebungsvariable (siehe vorherige README für Beispiele)
    export GEMINI_API_KEY='DEIN_API_KEY' # Beispiel Linux/MacOS
    ```

## Benutzung

Primär über die Streamlit GUI oder den Orchestrator:

*   **GUI starten:**
    ```bash
    streamlit run neuropersona_app.py
    ```
    Die GUI ermöglicht die Eingabe des Prompts, Anpassung der wichtigsten (klassischen und quanten-) Parameter und startet den im Orchestrator definierten Workflow.

*   **Orchestrator direkt aufrufen (Beispiel mit Parametern):**
    ```bash
    python orchestrator_full_qh_v1.py --prompt "Meine Analysefrage..." --epochs 50 --q_shots 10 --q_lr 0.02 --plots
    # Lässt man --prompt weg, fragt das Skript interaktiv.
    # --help zeigt alle Optionen an.
    ```

## Interpretation der Ergebnisse (Quantum Node Edition)

Denke an die **Kernphilosophie** und den **Disclaimer**:

*   **Fokus auf den Einzellauf:** Interpretiere den generierten HTML-Report und die Plots für *diesen spezifischen* Lauf.
*   **Quanten-Aktivierung verstehen:** Die Aktivierung eines Quanten-Knotens (0.0-1.0) ist die *geschätzte Wahrscheinlichkeit*, dass der Knoten bei einer Messung im Zustand |1> gefunden wird (basierend auf `n_shots`). Ein Wert von 0.8 bedeutet, er war in 80% der simulierten Messungen "aktiv". Dies ist inhärent probabilistisch.
*   **Interne Q-Parameter:** Der Plot `plot_act_weights.png` zeigt nun auch die Entwicklung des ersten internen Quantenparameters (z.B. Winkel Theta) für dynamische Knoten. Änderungen hier reflektieren das "interne Lernen" des Knotens und beeinflussen seine Aktivierungsempfindlichkeit.
*   **Analysiere den Zustand:** Wie verhalten sich dominante Kategorien (Quanten-aktiviert), Modulaktivitäten (Quanten-aktiviert), Werte (klassisch) und Emotionen (klassisch) *zueinander*? Ist das resultierende Profil intern kohärent im Sinne der Simulationslogik?
*   **Vergleiche nicht starr:** Erwarte *noch weniger* identische Ergebnisse bei erneuten Läufen als in der klassischen Version, wegen der zusätzlichen Stochastizität der Quantenmessungen. Beobachte die *Bandbreite* möglicher Zustände.
*   **Sättigung (Werte bei 1.0):** Gilt weiterhin für klassische Werte. Quanten-Aktivierungen *können* nahe 1.0 liegen, bedeuten aber "sehr hohe Wahrscheinlichkeit für |1>", nicht unbedingt eine harte Sättigung im klassischen Sinne.
*   **"Inkonsistenzen" als Ergebnis:** Bleiben valide und interpretierbare Resultate komplexer Dynamiken.

## Wichtige Parameter (Konstanten in `neuropersona_core_quantum_hybrid_v2.py` oder über GUI/CLI)

*   `DEFAULT_EPOCHS`: Anzahl Simulationszyklen.
*   `DEFAULT_LEARNING_RATE`: Basis-Lernrate für **klassische Gewichtsänderungen**.
*   `DEFAULT_DECAY_RATE`: Zerfallsrate für **klassische Gewichte**.
*   `VALUE_UPDATE_RATE`, `EMOTION_UPDATE_RATE`: Steuern klassische Wert/Emotions-Dynamik.
*   `PRUNING_THRESHOLD`, `SPROUTING_THRESHOLD`: Steuern klassische strukturelle Plastizität.
*   **`QUANTUM_ACTIVATION_SHOTS` (NEU):** Anzahl Messungen pro Quanten-Aktivierung. Mehr Shots -> genauere Wahrscheinlichkeitsschätzung, aber langsamer. Weniger Shots -> schneller, aber mehr Rauschen/Varianz in der Aktivierung.
*   **`QUANTUM_PARAM_LEARNING_RATE` (NEU):** Lernrate für die **internen Quantenparameter** der Knoten. Steuert, wie schnell sich Knoten intern anpassen.

Das Anpassen dieser Parameter beeinflusst die Dynamik, Stabilität und das emergente Verhalten des Systems. Tuning ist hier noch experimenteller als bei der klassischen Version.

---

## Eine Analogie für Nicht-Wissenschaftler: Wie NeuroPersona (Quantum Edition) "denkt"

Stellen Sie sich wieder einen **Menschen** vor, den Sie fragen: "Sollten wir in eine riskante Technologie investieren?". Wir wissen, die Antwort kann je nach Tag/Stimmung variieren (optimistisch, vorsichtig, analytisch).

Die klassische NeuroPersona-Version hat dies durch sich ändernde "Stimmungen" (Emotionen) und "Prioritäten" (Werte) simuliert.

Die **Quantum Node Edition** fügt eine weitere Ebene hinzu, inspiriert von der Quantenwelt:

*   **Unsichere "Meinungsstärke":** Stellen Sie sich vor, jeder "Gedanke" (Knoten) im Gehirn ist nicht einfach "an" oder "aus", sondern hat eine *Wahrscheinlichkeit*, aktiv zu sein. Manchmal ist ein Gedanke mit 80% Wahrscheinlichkeit präsent, manchmal nur mit 30%. NeuroPersona simuliert dies jetzt durch eine Art "Quanten-Würfeln" (`n_shots` Messungen). Das Ergebnis ist nicht immer gleich, selbst bei gleichem Input!
*   **Lernende Gedanken:** Nicht nur die *Verbindungen* zwischen Gedanken lernen, sondern der "Gedanke" selbst kann lernen, wie "sensibel" er auf bestimmte Inputs reagiert (Anpassung der internen Quantenparameter). Ein Gedanke über "Risiko" könnte mit der Zeit "sensibler" werden, wenn er oft zusammen mit negativen Erfahrungen aktiviert wird.
*   **Mehr Zufall, mehr Möglichkeiten:** Diese quanten-inspirierten Effekte erhöhen die Zufälligkeit und Variabilität im System. Das bedeutet, dass die Bandbreite der möglichen "Antworten" oder "Perspektiven", die die Simulation erzeugt, potenziell noch größer und überraschender sein kann.

**Wenn also die NeuroPersona Quantum Edition bei wiederholten Läufen unterschiedliche Ergebnisse liefert, ist das noch stärker als zuvor ein *gewolltes Feature*.** Es simuliert noch dynamischere, fluktuierendere Denkprozesse. Das Ziel bleibt, die **Bandbreite möglicher, plausibler kognitiver Reaktionen** zu verstehen, die nun durch diese zusätzliche, quanten-inspirierte Ebene beeinflusst werden. Es ist ein experimenteller Ansatz, um noch komplexere kognitive Dynamiken zu erforschen.
