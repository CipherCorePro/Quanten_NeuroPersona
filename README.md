# NeuroPersona QN - Ein Experimenteller Workflow zur Ideenanalyse
![image](https://github.com/user-attachments/assets/b8b36e59-c54a-4733-a678-a7d50fb2c180)
**(Wichtiger Hinweis: Dies ist ein experimentelles Projekt! Es dient der Forschung und dem Ausprobieren neuer Ideen. Die Ergebnisse sind nicht garantiert korrekt oder stabil.)**

## Was ist das überhaupt? 🤔

Stell dir vor, du könntest einem Computer nicht nur sagen, *was* er tun soll, sondern ihm auch beibringen, über ein komplexes Thema *nachzudenken* – ähnlich wie ein menschliches Gehirn, das verschiedene Aspekte beleuchtet, Ideen entwickelt, sie bewertet und sogar eine Art "Bauchgefühl" (Emotion) dazu hat.

Genau das versucht dieses Projekt **NeuroPersona QN** (QN steht für Quantum Nodes = Quanten-Knoten) zu simulieren:

*   Es nimmt eine Frage oder ein Thema (z.B. "Zukunft der KI in der Medizin").
*   Es zerlegt dieses Thema in kleinere Fragen und Antworten.
*   Ein simuliertes "Gehirn" (das NeuroPersona-Netzwerk) verarbeitet diese Informationen über mehrere Runden (Epochen).
*   Dieses "Gehirn" hat verschiedene spezialisierte "Bereiche" (Module), die für Kreativität, Kritik, Emotionen usw. zuständig sind.
*   Das Besondere hier: Die "Nervenzellen" (Knoten) in diesem Gehirn nutzen eine **Quanten-inspirierte** Methode, um Informationen sehr flexibel zu verarbeiten (mehr dazu unten).
*   Am Ende liefert das System einen Analysebericht und versucht, eine zusammenfassende Antwort auf deine ursprüngliche Frage zu geben.

**Kurz gesagt:** Es ist ein Software-Experiment, das versucht, komplexe Denk- und Analyseprozesse auf eine neuartige Weise nachzubilden.

## Wie funktioniert es (im Überblick)? ⚙️

Der ganze Prozess läuft in mehreren Schritten ab, die von verschiedenen Skripten gesteuert werden:

1.  **Thema verstehen (Optional: Mit KI-Hilfe):**
    *   Das Skript `gemini_perception_unit.py` nimmt dein eingegebenes Thema.
    *   **Wenn** du einen Zugang zu Googles KI "Gemini" hast (über einen API-Schlüssel), fragt es die KI, passende Unterthemen, Fragen und Antworten zu generieren.
    *   **Wenn nicht**, erfindet das Skript selbst ein paar einfache Fragen und Antworten zum Thema.
    *   *Ergebnis:* Eine strukturierte Liste von Diskussionspunkten für das simulierte Gehirn.

2.  **Das "Nachdenken" (Die Kernsimulation):**
    *   Das Skript `neuropersona_core_quantum_node_v2_multi_qubit.py` ist das **Herzstück**.
    *   Es baut ein Netzwerk aus vielen verbundenen Knoten (wie Nervenzellen) auf.
    *   Einige Knoten repräsentieren Kategorien (aus Schritt 1), andere sind spezielle "Denkmodule" (Kreativität, Kritik, Emotion, Strategie, Soziales Bewusstsein, Werte).
    *   Die "Quanten-Knoten" verarbeiten eingehende Signale auf eine spezielle, flexible Weise.
    *   Das Netzwerk durchläuft viele Zyklen ("Epochen"). In jedem Zyklus:
        *   Signale fließen durch das Netzwerk.
        *   Knoten werden basierend auf den Signalen und ihrer internen Quanten-Logik "aktiviert".
        *   Verbindungen zwischen aktiven Knoten werden stärker (Lernen).
        *   Schwache Verbindungen werden schwächer oder entfernt (Vergessen/Optimieren).
        *   Die "Denkmodule" werden aktiv (z.B. Ideen generieren, bewerten, Emotionen anpassen).
    *   *Ergebnis:* Ein detaillierter Bericht über den Zustand des Netzwerks am Ende und wie es sich entwickelt hat.

3.  **Alles zusammenfügen (Der Dirigent):**
    *   Das Skript `orchestrator_full_qh_v1.py` steuert den gesamten Ablauf.
    *   Es ruft zuerst die "Perception Unit" auf, um die Daten zu bekommen.
    *   Dann startet es die Kernsimulation ("NeuroPersona Core") mit den Daten.
    *   Zum Schluss (optional, wenn Gemini verfügbar ist) fasst es die Ergebnisse der Simulation zusammen und versucht, eine menschenlesbare Antwort auf deine ursprüngliche Frage zu formulieren, die den "Charakter" der Simulation widerspiegelt.

4.  **Die Benutzeroberfläche (Das Cockpit):**
    *   Das Skript `neuropersona_app.py` startet eine einfache Webseite in deinem Browser (mit Streamlit).
    *   Hier kannst du dein Thema eingeben, einige Simulationseinstellungen (wie Dauer, Lerngeschwindigkeit) anpassen und den Startknopf drücken.
    *   Die App ruft dann den "Dirigenten" (Orchestrator) auf und zeigt dir am Ende das Ergebnis an.

## Was ist das Besondere an den "Quanten-Knoten"? ✨

Okay, hier wird es ein bisschen abstrakt, aber keine Sorge, du brauchst kein Physikstudium!

*   **Normale Computerbits:** Sind entweder 0 oder 1 (An oder Aus).
*   **Quantenbits (Qubits):** Können durch Quantenphysik gleichzeitig 0 *und* 1 sein (und alles dazwischen). Das erlaubt viel komplexere Berechnungen.
*   **Unsere "Quanten-Knoten":** Sie sind *inspiriert* von dieser Idee. Sie nutzen *keine echte* Quantenhardware, sondern simulieren mathematisch eine Art "flexiblen Zustand". Jeder Knoten hier hat mehrere simulierte Qubits (z.B. 4 oder 10).
*   **Der Zweck:** Diese komplexere interne Logik soll den Knoten erlauben, auf Signale nuancierter und weniger vorhersehbar zu reagieren als einfache An/Aus-Schalter. Es ist ein Versuch, die "Unschärfe" oder "Mehrdeutigkeit" im menschlichen Denken nachzubilden.
*   **Das "Messen":** Um eine klare Aktivität (einen Wert zwischen 0 und 1) zu bekommen, muss der simulierte Quantenzustand "gemessen" werden. Das Ergebnis ist wahrscheinlichkeitsbasiert (wie in der echten Quantenmechanik). Wir machen das mehrmals ("Shots") und nehmen den Durchschnitt, um eine stabilere Aktivierung zu erhalten.
*   **Lernen:** Auch die internen "Quanten-Parameter" der Knoten können sich durch Lernen leicht verändern.

**Wichtig:** Es ist eine **Simulation und Inspiration**, kein echter Quantencomputer! Aber es ist ein spannendes Experiment, um zu sehen, ob dieser Ansatz zu interessanteren oder "lebensechteren" Simulationsergebnissen führt. Die `quantum_logs` speichern übrigens sehr detailliert, was in diesen Knoten passiert (nur für Experten interessant).

## Voraussetzungen 📋

*   **Python:** Du brauchst Python 3 (idealerweise 3.9 oder neuer) auf deinem Computer.
*   **Bibliotheken:** Einige zusätzliche Python-Pakete werden benötigt. Du installierst sie über die Kommandozeile mit `pip`. Die wichtigsten sind:
    *   `pandas` (für Datenstrukturen)
    *   `numpy` (für Zahlenberechnungen)
    *   `streamlit` (für die Web-Oberfläche)
    *   `matplotlib` (zum Erstellen von Diagrammen/Plots)
    *   `networkx` (optional, für Netzwerk-Diagramme)
    *   `google-generativeai` (optional, nur wenn du Gemini nutzen willst)
    *   `tqdm` (optional, für Fortschrittsbalken in der Konsole)
*   **Google Gemini API Key (Optional):**
    *   Wenn du möchtest, dass das System die bestmöglichen Eingabedaten generiert und am Ende eine hochwertige Zusammenfassung schreibt, brauchst du einen API-Schlüssel von Google AI Studio.
    *   Dieser Schlüssel muss als Umgebungsvariable namens `GEMINI_API_KEY` gesetzt werden. Wie das geht, hängt von deinem Betriebssystem ab (suche nach "Umgebungsvariable setzen [dein Betriebssystem]").
    *   **Ohne Schlüssel funktioniert das Programm auch**, aber die Ergebnisse sind einfacher (es nutzt dann eingebaute Fallbacks).

## Installation 📦

1.  Stelle sicher, dass du Python und pip installiert hast.
2.  Öffne eine Kommandozeile (Terminal, Eingabeaufforderung).
3.  Navigiere in das Verzeichnis, in dem du die heruntergeladenen `.py`-Dateien gespeichert hast.
4.  Installiere die benötigten Bibliotheken mit diesem Befehl:
    ```bash
    pip install pandas numpy streamlit matplotlib networkx google-generativeai tqdm
    ```
    *(Wenn du Gemini nicht nutzen willst, kannst du `google-generativeai` weglassen.)*

## Konfiguration (Wichtig!) ⚙️

Bevor du startest, überprüfe **unbedingt** diese Punkte:

1.  **Gemini API Key (Optional):** Wenn du Gemini nutzen willst, setze die Umgebungsvariable `GEMINI_API_KEY` mit deinem Schlüssel.
2.  **Dateinamen prüfen!**
    *   Öffne die Datei `neuropersona_app.py`. Finde die Zeile `ORCHESTRATOR_MODULE = "orchestrator_full_qh_v1"` und stelle sicher, dass der Name (`orchestrator_full_qh_v1`) **exakt** dem Dateinamen deines Orchestrator-Skripts entspricht (ohne `.py`). Passe ihn bei Bedarf an!
    *   Öffne die Datei `orchestrator_full_qh_v1.py`. Finde die Zeile `NEUROPERSONA_CORE_MODULE = "neuropersona_core_quantum_hybrid_v2"` und stelle sicher, dass der Name (`neuropersona_core_quantum_hybrid_v2`) **exakt** dem Dateinamen deines Core-Simulations-Skripts entspricht (ohne `.py`). Passe ihn bei Bedarf an!

    **Wenn diese Namen nicht stimmen, kann die App den Orchestrator nicht finden, oder der Orchestrator kann die Simulation nicht starten!**

## Wie starte ich es? ▶️

Es gibt zwei Hauptwege:

1.  **Mit der Benutzeroberfläche (Empfohlen für Einsteiger):**
    *   Öffne eine Kommandozeile im Projektverzeichnis.
    *   Gib ein: `streamlit run neuropersona_app.py`
    *   Dein Webbrowser sollte sich öffnen und die Oberfläche anzeigen.
    *   Gib dein Thema ein, passe ggf. die Parameter an (oder lass die Standardwerte) und klicke auf "Starte NeuroPersona Analyse".
    *   Warte, bis die Simulation fertig ist (das kann je nach Einstellungen und Thema eine Weile dauern!). Das Ergebnis erscheint dann auf der Seite.

2.  **Direkt über die Kommandozeile (Für Fortgeschrittene):**
    *   Öffne eine Kommandozeile im Projektverzeichnis.
    *   Gib ein: `python orchestrator_full_qh_v1.py`
    *   Du wirst nach einem Thema gefragt. Gib es ein und drücke Enter.
    *   Die Simulation läuft im Hintergrund, und du siehst Statusmeldungen. Am Ende wird die finale Antwort ausgegeben.
    *   Du kannst auch Parameter direkt mitgeben, z.B.:
        ```bash
        # Beispiel: Weniger Epochen, höhere Quanten-Lernrate, ohne Plots
        python orchestrator_full_qh_v1.py --epochs 15 --q_lr 0.015 --no-plots --prompt "Mein Thema hier"
        ```
        (Mit `--prompt` wird die interaktive Frage übersprungen.)

## Was kann ich erwarten? 📊

Wenn die Simulation läuft, siehst du Statusmeldungen in der Konsole oder im Statusbereich der Web-App.

Am Ende erhältst du (je nach Einstellungen):

*   **Eine finale Textantwort:** Diese versucht, die Analyseergebnisse zusammenzufassen und auf deine Frage einzugehen, angepasst an den "Charakter" der Simulation. Wenn du einen Programmierauftrag gestellt hast, kann hier auch Python-Code enthalten sein, der von der "Odin"-Persona erstellt wurde.
*   **Einen HTML-Bericht:** Eine Datei namens `neuropersona_report_MQ...html` wird im Projektverzeichnis gespeichert. Sie enthält eine detailliertere Zusammenfassung und (falls aktiviert) die generierten Diagramme.
*   **Diagramme (Plots):** Wenn die Option "Plots generieren" aktiviert ist, werden mehrere `.png`-Diagramme in einem Unterordner namens `plots_quantum_node_...` gespeichert. Sie zeigen z.B.:
    *   Wie sich die Aktivierungen der Knoten über die Zeit ändern.
    *   Wie sich die Verbindungsgewichte entwickeln.
    *   Wie sich die "Emotionen" und "Werte" im Netzwerk verändern.
    *   Statistiken zur Netzwerkstruktur.
    *   Eine Visualisierung des Netzwerks selbst (nur wenn `networkx` installiert ist).
*   **Quanten-Logs (Für Experten):** Im Ordner `quantum_logs` werden sehr detaillierte JSON-Dateien gespeichert, die jeden Schritt innerhalb der Quanten-Knoten-Aktivierung protokollieren. Das ist meist nur für die Fehlersuche oder tiefe Analyse relevant.
*   **Gespeicherter Zustand (Optional):** Wenn "Speichern" aktiviert ist, wird der Endzustand des Netzwerks (inkl. der gelernten Quanten-Parameter) in einer `.json`-Datei gespeichert (z.B. `neuropersona_quantum_node_mq4_state_v2.json`), damit du eine Simulation später fortsetzen ("Laden") kannst.

## Wichtige Hinweise & Haftungsausschluss ⚠️

*   **EXPERIMENTELL:** Dieses Projekt ist hoch experimentell. Die "Quanten"-Aspekte sind eine Simulation und Inspiration, keine exakte Abbildung der Quantenphysik.
*   **Keine Garantien:** Die Ergebnisse der Simulation (Berichte, Antworten, Diagramme) sind nicht notwendigerweise korrekt, vollständig oder wissenschaftlich fundiert. Sie spiegeln nur den Zustand des simulierten Systems wider.
*   **Rechenintensiv:** Besonders die Quanten-Knoten-Simulation kann je nach Anzahl der Epochen, Knoten und "Shots" lange dauern und viel Rechenleistung benötigen.
*   **Fehler möglich:** Wie bei jeder komplexen Software können Fehler auftreten.
*   **Keine Haftung:** Die Nutzung dieser Software erfolgt auf eigene Gefahr. Es wird keine Haftung für Ergebnisse, Fehler oder mögliche Schäden übernommen.
