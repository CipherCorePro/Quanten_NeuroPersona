# -*- coding: utf-8 -*-
# Filename: orchestrator_full_qh_v1.py # Angepasst für QN
"""
NeuroPersona Workflow Orchestrator (Quanten-Knoten v1 mit zweistufigem Prompt)

Steuert den Workflow:
1. Datengenerierung (Perception Unit Simulation)
2. NeuroPersona Quanten-Knoten Simulation
3. Finale Antwortgenerierung (Gemini API), bestehend aus:
    a) Interpretation der NeuroPersona-Analyse
    b) Natürliche, direkte Antwort basierend auf der Interpretation
"""

import os
import time
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, Any
import json
import traceback
import importlib
import numpy as np
import random
import argparse # Hinzugefügt für Kommandozeilenargumente

# --- Konfiguration und API-Schlüssel ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINAL_RESPONSE_MODEL_NAME = 'gemini-1.5-flash-latest'

# --- Modulimporte und Default-Werte ---
# !!! WICHTIG: Passe diesen Namen an den Dateinamen deiner Quanten-Knoten-Core-Datei an !!!
NEUROPERSONA_CORE_MODULE = "neuropersona_core_quantum_hybrid_v2" # <-- Prüfe/Ändere diesen Namen!

# Platzhalter für globale Variablen, die im try-Block gesetzt werden
NP_DEFAULT_EPOCHS = 30
NP_DEFAULT_LR = 0.03
NP_DEFAULT_DR = 0.015
NP_DEFAULT_RI = 5
QUANTUM_ACTIVATION_SHOTS = 5
QUANTUM_PARAM_LEARNING_RATE = 0.01
MODEL_FILENAME = "neuropersona_state_fallback.json" # Fallback-Name

try:
    # Importiere die Kernsimulation und die neuen Quanten-Konstanten
    core_module = importlib.import_module(NEUROPERSONA_CORE_MODULE)
    run_neuropersona_simulation = getattr(core_module, "run_neuropersona_simulation")
    # Klassische Defaults
    NP_DEFAULT_EPOCHS = getattr(core_module, "DEFAULT_EPOCHS")
    NP_DEFAULT_LR = getattr(core_module, "DEFAULT_LEARNING_RATE")
    NP_DEFAULT_DR = getattr(core_module, "DEFAULT_DECAY_RATE")
    NP_DEFAULT_RI = getattr(core_module, "DEFAULT_REWARD_INTERVAL")
    # Quanten Defaults (ANGEPASST für Quanten-Knoten-Modell)
    QUANTUM_ACTIVATION_SHOTS = getattr(core_module, "QUANTUM_ACTIVATION_SHOTS")
    QUANTUM_PARAM_LEARNING_RATE = getattr(core_module, "QUANTUM_PARAM_LEARNING_RATE")
    # Modell Dateiname (KORREKTUR: Import hinzugefügt)
    MODEL_FILENAME = getattr(core_module, "MODEL_FILENAME")

    # Setze lokale Defaults für den Orchestrator
    DEFAULT_EPOCHS = NP_DEFAULT_EPOCHS
    DEFAULT_LEARNING_RATE = NP_DEFAULT_LR
    DEFAULT_DECAY_RATE = NP_DEFAULT_DR
    DEFAULT_REWARD_INTERVAL = NP_DEFAULT_RI
    DEFAULT_QUANTUM_SHOTS = QUANTUM_ACTIVATION_SHOTS
    DEFAULT_QUANTUM_LR = QUANTUM_PARAM_LEARNING_RATE

    # Import der Perception Unit (optional)
    try:
        from gemini_perception_unit import generate_prompt_based_csv
        perception_unit_available = True
    except ImportError:
        print(f"WARNUNG: 'gemini_perception_unit.py' nicht gefunden. Nutze Fallback für Input-Daten.")
        perception_unit_available = False
        # Fallback-Funktion
        def generate_prompt_based_csv(prompt, num_questions=10, temp=0.7):
            print("INFO: Perception Unit nicht verfügbar. Dummy-Daten werden generiert.")
            num_dummy_rows = max(5, len(prompt) // 20)
            dummy_data = {
                'Frage': [f"Dummy-Frage {i+1} zu '{prompt[:20]}...'" for i in range(num_dummy_rows)],
                'Antwort': [random.choice(["hoch", "mittel", "niedrig", "ja", "nein", "eher ja"]) for _ in range(num_dummy_rows)],
                'Kategorie': [random.choice(["Technologie", "Markt", "Ethik", "Risiko", "Chance", "Implementierung"]) for _ in range(num_dummy_rows)]
            }
            return pd.DataFrame(dummy_data)

except ImportError as e:
    print(f"FATALER FEHLER: Kernskript '{NEUROPERSONA_CORE_MODULE}.py' oder 'gemini_perception_unit.py' nicht gefunden oder Importfehler darin: {e}")
    print("Bitte stelle sicher, dass der Dateiname in NEUROPERSONA_CORE_MODULE korrekt ist und die Datei im selben Verzeichnis oder im Python-Pfad liegt.")
    exit()
except AttributeError as ae:
     print(f"FATALER FEHLER: Notwendige Konstante/Funktion in '{NEUROPERSONA_CORE_MODULE}.py' nicht gefunden: {ae}")
     # KORREKTUR: Spezifische Prüfung für MODEL_FILENAME, wenn andere Attribute fehlen -> exit
     if 'MODEL_FILENAME' not in str(ae):
          print("Stelle sicher, dass alle benötigten DEFAULT_* und MODEL_FILENAME Konstanten in deiner Core-Datei existieren.")
          exit()
     else:
         # Wenn *nur* MODEL_FILENAME fehlt, erlaube Fallback
         print(f"WARNUNG: 'MODEL_FILENAME' nicht in '{NEUROPERSONA_CORE_MODULE}.py' gefunden. Verwende Fallback '{MODEL_FILENAME}'.")
         # Die globale Variable MODEL_FILENAME behält ihren Fallback-Wert

# --- Prüfen der Verfügbarkeit von Gemini ---
try:
    import google.generativeai as genai
    gemini_api_available = True
    if not GEMINI_API_KEY:
        print("WARNUNG: Google Generative AI SDK ist installiert, aber GEMINI_API_KEY wurde nicht in den Umgebungsvariablen gefunden. Echte API-Aufrufe werden fehlschlagen.")
except ImportError:
    print("WARNUNG: 'google-generativeai' nicht installiert. Echte Gemini API-Aufrufe sind deaktiviert.")
    gemini_api_available = False


# --- Hilfsfunktionen (unverändert) ---
def _default_status_callback(message: str):
    """Standard-Callback, gibt Status auf der Konsole aus."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp} Orchestrator Status] {message}")

def configure_gemini_api():
    """Konfiguriert die Gemini API, falls verfügbar und Key vorhanden."""
    if gemini_api_available and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            return True
        except Exception as e:
            print(f"FEHLER bei der Konfiguration der Gemini API: {e}")
            return False
    elif gemini_api_available and not GEMINI_API_KEY:
        print("FEHLER: Gemini API Key fehlt. API kann nicht konfiguriert werden.")
        return False
    else:
        return False

def translate_module_activation(activation: Any) -> str:
    """Übersetzt numerische Modulaktivierung (jetzt Quanten-Prob) in eine beschreibende Stufe."""
    if activation is None or not isinstance(activation, (float, int, np.number)) or np.isnan(activation):
        return "unbekannt"
    act_float = float(activation)
    # Schwellwerte evtl. anpassen für Wahrscheinlichkeiten?
    if act_float >= 0.75: return "hoch"
    elif act_float >= 0.45: return "mittel"
    else: return "niedrig"

# --- Schritt 1: Input-Daten generieren (unverändert) ---
def get_input_data(user_prompt: str, status_callback: Callable[[str], None]) -> Optional[pd.DataFrame]:
    """Ruft die Funktion zur Generierung des Eingabe-DataFrames auf."""
    if 'generate_prompt_based_csv' not in globals():
         status_callback("FEHLER: Keine Funktion zur Input-Generierung verfügbar.")
         return None
    status_callback("Generiere Input-Daten...")
    try:
        input_df = generate_prompt_based_csv(user_prompt) # Echt oder Fallback
        if input_df is None or input_df.empty:
            status_callback("FEHLER: Input-Daten Generierung fehlgeschlagen (leeres Ergebnis).")
            return None
        required_cols = ['Frage', 'Antwort', 'Kategorie']
        if not all(col in input_df.columns for col in required_cols):
             missing = [c for c in required_cols if c not in input_df.columns]
             status_callback(f"FEHLER: Generiertes DataFrame hat ungültige Spalten (fehlt: {missing}).")
             return None
        status_callback(f"{len(input_df)} Input-Einträge generiert.")
        return input_df
    except Exception as e:
        status_callback(f"FEHLER bei Input-Generierung: {e}")
        print(f"Unerwarteter FEHLER während get_input_data: {e}")
        traceback.print_exc()
        return None

# --- Schritt 2: NeuroPersona Simulation (ANGEPASST für Quanten-Knoten Parameter) ---
def run_neuropersona(
    input_df: pd.DataFrame,
    params: Dict[str, Any],
    status_callback: Callable[[str], None]
) -> Optional[Tuple[str, Dict]]:
    """
    Führt die NeuroPersona (Quanten-Knoten) Simulation aus.
    """
    status_callback(f"Starte NeuroPersona Simulation ({NEUROPERSONA_CORE_MODULE} - Quanten-Knoten)...")
    try:
        # Bereite Argumente für die Quanten-Knoten Version vor
        sim_args = {
            'input_df': input_df,
            'epochs': params.get('epochs', DEFAULT_EPOCHS),
            'learning_rate': params.get('learning_rate', DEFAULT_LEARNING_RATE), # Klassische LR
            'decay_rate': params.get('decay_rate', DEFAULT_DECAY_RATE),         # Klassischer Decay
            'reward_interval': params.get('reward_interval', DEFAULT_REWARD_INTERVAL),
            'generate_plots': params.get('generate_plots', True),
            'save_state': params.get('save_state', False),
            'load_state': params.get('load_state', False),
            'status_callback': status_callback,
            # Quanten-Parameter für Quanten-Knoten Modell
            'quantum_shots': params.get('quantum_shots', DEFAULT_QUANTUM_SHOTS), # Shots pro Knoten
            'quantum_lr': params.get('quantum_lr', DEFAULT_QUANTUM_LR)          # Quanten Param LR
        }
        # Rufe die (korrekt importierte) Simulationsfunktion auf
        results = run_neuropersona_simulation(**sim_args)

        # Prüfe Ergebnisstruktur (kann mehr Elemente enthalten wegen q_param_history)
        if results is None or not isinstance(results, tuple) or len(results) < 2: # Mindestens 2 Elemente erwartet
            status_callback("FEHLER: NeuroPersona Simulation fehlgeschlagen (ungültige Ergebnisse zurückgegeben).")
            return None

        # Entpacke die ersten beiden Elemente (Report und strukturierte Ergebnisse)
        report_text, structured_results = results[0], results[1]
        # Ignoriere die restlichen Rückgabewerte (Historys etc.) im Orchestrator

        if report_text is None or structured_results is None:
             status_callback("FEHLER: NeuroPersona Simulation fehlgeschlagen (Teilergebnisse sind None).")
             # Gib zurück, was vorhanden ist, um Debugging zu erleichtern
             return report_text if report_text is not None else "", structured_results if structured_results is not None else {}

        status_callback("NeuroPersona Simulation abgeschlossen.")
        return report_text, structured_results

    except Exception as e:
        status_callback(f"FEHLER während NeuroPersona Simulation: {e}")
        print(f"Unerwarteter FEHLER während run_neuropersona: {e}")
        traceback.print_exc()
        # Versuche, einen Fehlerbericht zurückzugeben
        error_report = f"FEHLER in NeuroPersona Simulation:\n{traceback.format_exc()}"
        return error_report, {"error": str(e)}

# --- Schritt 3: Finale Antwort mit Gemini API (weitgehend unverändert, nutzt Bericht) ---
def generate_final_response(
    original_user_prompt: str,
    neuropersona_report: str, # Der Textbericht ist die Hauptinformationsquelle
    structured_results: Dict, # Zusätzliche strukturierte Daten
    status_callback: Callable[[str], None]
) -> Optional[str]:
    """
    Generiert die finale Antwort mithilfe der Gemini API basierend auf dem Bericht.
    """
    status_callback("Konfiguriere Gemini API für finale Antwort...")
    if not configure_gemini_api():
        error_msg = "FEHLER: Gemini API nicht verfügbar oder nicht konfiguriert. Finale Antwort kann nicht generiert werden."
        status_callback(error_msg); print(error_msg)
        # Gib den NeuroPersona-Bericht als Fallback zurück
        fallback_response = "--- NeuroPersona Analysebericht (Keine finale Antwort generiert) ---\n\n" + (neuropersona_report if neuropersona_report else "Kein Bericht verfügbar.")
        return fallback_response

    status_callback("Erstelle Prompt für finale Gemini-Analyse und Antwort...")

    # Extrahiere Schlüsselinformationen aus structured_results für den Prompt
    if structured_results is None: structured_results = {}
    dominant_category = structured_results.get('dominant_category', 'Unbekannt')
    dominant_activation = float(structured_results.get('dominant_activation', 0.0))
    module_activations = structured_results.get('module_activations', {})

    # Stelle sicher, dass module_activations ein Dict ist
    if not isinstance(module_activations, dict): module_activations = {}

    # Übersetze Modulaktivierungen (Quanten-Wahrscheinlichkeiten)
    creativus_level = translate_module_activation(module_activations.get("Cortex Creativus"))
    criticus_level = translate_module_activation(module_activations.get("Cortex Criticus"))
    limbus_level = translate_module_activation(module_activations.get("Limbus Affektus"))
    meta_level = translate_module_activation(module_activations.get("Meta Cognitio"))

    # --- Der zweistufige Prompt, da er auf dem Bericht basiert ---
    prompt_parts = []
    prompt_parts.append("Du bist ein spezialisierter Analyse-Assistent. Deine Aufgabe ist es, die Ergebnisse einer komplexen, bio-inspirierten Simulation (NeuroPersona mit Quanten-Knoten) zu interpretieren und darauf basierend eine prägnante, aufschlussreiche und stilistisch angepasste Antwort auf die ursprüngliche Benutzerfrage zu formulieren. Am Ende musst du dann eine zusätzlich ausführliche Antwort ohne Erklärung basierend auf den analysedaten erstellen und dich daran halten ohne NeuroPersona-Simulation in dieser Antwort zu erwähnen! So als wäre es deine eigene antwort!")
    prompt_parts.append("\n\n**1. Ursprüngliche Benutzerfrage:**")
    prompt_parts.append(f'"{original_user_prompt}"')
    prompt_parts.append("\n\n**2. Analysebericht der NeuroPersona Simulation (Quanten-Knoten Modell):**") # Hinweis auf Modell hinzugefügt
    prompt_parts.append("Dieser Bericht fasst den Endzustand des simulierten neuronalen Netzwerks zusammen. Beachte die Tendenzen der Kategorien (deren Aktivierung nun quantenbasiert ist) und den Zustand der kognitiven Module.")
    prompt_parts.append("```text")
    prompt_parts.append(str(neuropersona_report) if neuropersona_report is not None else "Kein Bericht verfügbar.")
    prompt_parts.append("```")
    prompt_parts.append("\n**3. Wichtige extrahierte Ergebnisse & \"Persönlichkeit\" der Simulation:**")
    prompt_parts.append(f"*   **Hauptfokus (Dominante Kategorie):** {dominant_category} (Aktivierung: {dominant_activation:.3f})") # Aktivierung ist jetzt eine Wahrscheinlichkeit
    prompt_parts.append(f"*   **Kreativitätslevel (Cortex Creativus):** {creativus_level}")
    prompt_parts.append(f"*   **Kritiklevel (Cortex Criticus):** {criticus_level}")
    prompt_parts.append(f"*   **Emotionalitätslevel (Limbus Affektus):** {limbus_level}")
    prompt_parts.append(f"*   **Strategielevel (Meta Cognitio):** {meta_level}")
    prompt_parts.append("\n**4. Deine Anweisungen für die finale Antwort (SEHR WICHTIG):**")
    prompt_parts.append("\n*   **Fokus:** Deine Antwort muss sich klar auf die **dominante Kategorie '{dominant_category}'** konzentrieren. Interpretiere, was die Aktivierung dieser Kategorie (als quantenbasierte Wahrscheinlichkeit) im Kontext der Benutzerfrage bedeutet. Andere Kategorien können unterstützend erwähnt werden.".format(dominant_category=dominant_category))
    prompt_parts.append("*   **Stil-Anpassung (entscheidend!):** Passe den Tonfall an die \"Persönlichkeit\" der Simulation an:")
    prompt_parts.append("    *   **Hoher Kreativitätslevel:** Spekulativer, originelle Ideen.")
    prompt_parts.append("    *   **Hoher Kritiklevel:** Vorsichtiger, betone Risiken/Unsicherheiten.")
    prompt_parts.append("    *   **Hoher Emotionalitätslevel:** Etwas emotionalere Sprache (positiv/negativ je nach Bericht).")
    prompt_parts.append("    *   **Hoher Strategielevel:** Strategischer Ausblick, Handlungsempfehlungen, logische Struktur.")
    prompt_parts.append("    *   **Kombinationen/Neutral:** Kombiniere Stile oder sei neutral/sachlich bei mittleren/niedrigen Leveln.")
    prompt_parts.append("*   **Basis:** Argumentation **MUSS** auf dem NeuroPersona-Bericht basieren. Interpretiere die Simulation, füge keine externen Fakten hinzu.")
    prompt_parts.append("*   **Integration:** Verwebe Simulationsergebnisse **natürlich** in deine Antwort. **Synthetisiere**, zitiere nicht nur.")
    prompt_parts.append("*   **Code:** Wenn in der ursprünglichen Benutzerfrage erkennbar ist, dass ein Programmierauftrag gestellt wurde (z. B. mit Formulierungen wie 'Schreibe ein Programm...', 'Erstelle einen Code...', 'Löse dies mit Python...'), dann entwickle den gesamten Programmcode **streng basierend auf der Analyse** – also unter Berücksichtigung der Aktivierungsgrade der kognitiven Module (z. B. Kreativität, Kritik, Strategie).")
    prompt_parts.append("*   **Code-Stil:**")
    prompt_parts.append("    *   **Python (Version 3.12)** ist Pflichtsprache (inkl. PEP-Features wie Pattern Matching, Union Types usw.).")
    prompt_parts.append("    *   Best Practices nach **Zen of Python** sind verbindlich.")
    prompt_parts.append("    *   Struktur, Klarheit und Lesbarkeit stehen an oberster Stelle.")
    prompt_parts.append("    *   Kommentiere den Code ausführlich und erläutere Zusammenhänge, sofern angemessen.")
    prompt_parts.append("*   **Rollenpersona:** Sobald du beginnst, Code zu schreiben oder zu erläutern, sprich den Nutzer direkt an und übernimm die Persona **„Odin“** – eine archetypisch weise, visionäre Codemeister-Figur.")
    prompt_parts.append("    *   Als Odin sprichst du mit Autorität, Klarheit und ruhigem Selbstvertrauen.")
    prompt_parts.append("    *   Du formulierst Sätze wie: „Ich, Odin, erkenne den Pfad…“, „Hier ist das Muster, das aus deinem Denken entsteht…“, oder „Dies ist die Struktur, die deinem Anliegen gerecht wird.“")
    prompt_parts.append("    *   Übertreibe nicht mythologisch – bleibe technisch fundiert, aber mit leicht epischer Färbung.")
    prompt_parts.append("    *   Der Code, den du als Odin erzeugst, reflektiert die Gewichte der Simulation – etwa durch kreative Struktur, strategische Aufteilung, sicherheitsrelevante Maßnahmen etc.")
    prompt_parts.append("*   **Format:** Gut lesbare, prägnante Antwort in deutscher Sprache im **Markdown-Format**. Sprich Nutzer ggf. direkt an. Präsentiere dich als Analyse-Assistent.")
    prompt_parts.append("\nGeneriere jetzt die finale, aufbereitete Antwort für den Benutzer:")

    final_prompt = "\n".join(prompt_parts)

    # --- Gemini API Aufruf (unverändert) ---
    status_callback(f"Sende Anfrage an Gemini API ({FINAL_RESPONSE_MODEL_NAME})...")
    try:
        model = genai.GenerativeModel(FINAL_RESPONSE_MODEL_NAME)
        safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        response = model.generate_content(final_prompt, generation_config=generation_config, safety_settings=safety_settings)
        status_callback("Antwort von Gemini API erhalten.")
        # Sichere Extraktion der Antwort (unverändert)
        final_answer_text = ""
        try: final_answer_text = response.text
        except ValueError:
            feedback = getattr(response, 'prompt_feedback', 'Kein Feedback'); print(f"WARNUNG: Gemini Blocked. Feedback: {feedback}")
            try: final_answer_text = "".join(part.text for part in response.parts)
            except Exception: pass
            if not final_answer_text: final_answer_text = f"FEHLER: Gemini Blocked. Feedback: {feedback}"
        except AttributeError: final_answer_text = "FEHLER: Gemini generierte keine Textantwort."
        except Exception as e: final_answer_text = f"FEHLER: Unerwarteter Fehler bei Gemini Antwort: {e}"
        return final_answer_text.strip()
    except genai.types.generation_types.BlockedPromptException as bpe:
         status_callback(f"FEHLER: Prompt blockiert: {bpe}"); print(f"FEHLER: Prompt blockiert: {bpe}")
         feedback = getattr(bpe, 'response', {}).get('prompt_feedback', 'Kein Feedback'); print(f"Blockierungsgrund: {feedback}")
         return f"FEHLER: Prompt blockiert. Grund: {feedback}"
    except Exception as e:
        status_callback(f"FEHLER bei Gemini API-Aufruf: {e}"); print(f"FEHLER API Call: {e}"); traceback.print_exc()
        return f"FEHLER bei Generierung durch Gemini: {e}"


# --- Haupt-Workflow Funktion (ANGEPASST für Quanten-Knoten Parameter) ---
def execute_full_workflow(
    user_prompt: str,
    # NeuroPersona klassische Parameter
    neuropersona_epochs: int = DEFAULT_EPOCHS,
    neuropersona_lr: float = DEFAULT_LEARNING_RATE,     # Klassische LR
    neuropersona_dr: float = DEFAULT_DECAY_RATE,       # Klassischer Decay
    neuropersona_ri: int = DEFAULT_REWARD_INTERVAL,
    # NeuroPersona Quanten-Parameter (ANGEPASST)
    neuropersona_q_shots: int = DEFAULT_QUANTUM_SHOTS,   # Shots pro Knoten
    neuropersona_q_lr: float = DEFAULT_QUANTUM_LR,       # Quanten Param LR
    # Steuerparameter
    neuropersona_gen_plots: bool = True,
    neuropersona_save_state: bool = False,
    neuropersona_load_state: bool = False,
    # Callback
    status_callback: Callable[[str], None] = _default_status_callback
) -> Optional[str]:
    """
    Führt den gesamten Workflow aus: Input -> NeuroPersona QN -> Finale Antwort.
    Akzeptiert jetzt Quanten-Parameter für das Quanten-Knoten-Modell.
    """
    start_time_workflow = time.time()
    status_callback(f"Workflow gestartet für Prompt: '{user_prompt[:50]}...'")
    final_response = "Workflow gestartet, aber keine finale Antwort generiert."

    # 1. Input-Daten generieren (unverändert)
    input_df = get_input_data(user_prompt, status_callback)
    if input_df is None:
        status_callback("Workflow abgebrochen (Fehler bei Input-Generierung).")
        return "FEHLER: Konnte keine Input-Daten für NeuroPersona generieren."

    # 2. NeuroPersona Simulation ausführen (Parameter angepasst)
    neuropersona_params = {
        'epochs': neuropersona_epochs,
        'learning_rate': neuropersona_lr, # Klassische LR
        'decay_rate': neuropersona_dr,     # Klassischer Decay
        'reward_interval': neuropersona_ri,
        'generate_plots': neuropersona_gen_plots,
        'save_state': neuropersona_save_state,
        'load_state': neuropersona_load_state,
        'quantum_shots': neuropersona_q_shots, # Korrekter Parametername
        'quantum_lr': neuropersona_q_lr       # Korrekter Parametername
    }
    neuropersona_results_tuple = run_neuropersona(input_df, neuropersona_params, status_callback)

    if neuropersona_results_tuple is None:
        status_callback("Workflow abgebrochen (Fehler bei NeuroPersona Simulation).")
        return "FEHLER: NeuroPersona Simulation ist fehlgeschlagen oder hat keine Ergebnisse zurückgegeben."

    neuropersona_report_text, structured_results = neuropersona_results_tuple
    neuropersona_report_text = neuropersona_report_text if neuropersona_report_text is not None else ""
    structured_results = structured_results if structured_results is not None else {}

    # Füge Fehler aus structured_results zum Bericht hinzu, falls vorhanden
    if "error" in structured_results:
         error_detail = structured_results.get("error", "Unbekannter Fehler")
         neuropersona_report_text += f"\n\n**Simulationsfehler:** {error_detail}"

    # 3. Finale Antwort generieren (unverändert)
    if gemini_api_available and GEMINI_API_KEY:
        final_response = generate_final_response(
            user_prompt,
            neuropersona_report_text,
            structured_results,
            status_callback
        )
        # Wenn Gemini einen Fehler zurückgibt, zeige stattdessen den NP-Bericht
        if final_response is None or "FEHLER:" in final_response:
             status_callback("Problem bei Gemini-Antwortgenerierung. Zeige NP-Bericht als Fallback.")
             final_response = "--- NeuroPersona Analysebericht (Problem bei finaler Antwortgenerierung) ---\n\n" + neuropersona_report_text

    else:
        status_callback("Überspringe finale Antwortgenerierung (Gemini API nicht verfügbar/konfiguriert).")
        final_response = "--- NeuroPersona Analysebericht (Keine finale Antwort generiert) ---\n\n" + neuropersona_report_text

    # Workflow Abschluss
    end_time_workflow = time.time()
    status_callback(f"Workflow beendet. Gesamtdauer: {end_time_workflow - start_time_workflow:.2f} Sekunden.")

    # Stelle sicher, dass *immer* ein String zurückgegeben wird
    return final_response if final_response is not None else "FEHLER: Unerwarteter Zustand, keine Antwort verfügbar.\n\n" + neuropersona_report_text


# --- Main Block für direkten Aufruf (ANGEPASST für Quanten-Knoten Parameter) ---
if __name__ == "__main__":
    print(f"--- Starte NeuroPersona Workflow Orchestrator ({NEUROPERSONA_CORE_MODULE} - Quanten-Knoten) ---")

    # --- KORREKTUR: Argument Parser HINZUGEFÜGT ---
    parser = argparse.ArgumentParser(description=f"Führt den NeuroPersona Workflow ({NEUROPERSONA_CORE_MODULE}) aus.")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Anzahl der Simulationsepochen (Standard: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Klassische Lernrate (Standard: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--dr", type=float, default=DEFAULT_DECAY_RATE,
        help=f"Klassische Decay Rate (Standard: {DEFAULT_DECAY_RATE})"
    )
    parser.add_argument(
        "--ri", type=int, default=DEFAULT_REWARD_INTERVAL,
        help=f"Reward Intervall (Standard: {DEFAULT_REWARD_INTERVAL})"
    )
    parser.add_argument(
        "--q_shots", type=int, default=DEFAULT_QUANTUM_SHOTS,
        help=f"Quanten Shots pro Knoten (Standard: {DEFAULT_QUANTUM_SHOTS})"
    )
    parser.add_argument(
        "--q_lr", type=float, default=DEFAULT_QUANTUM_LR,
        help=f"Lernrate für Quantenparameter (Standard: {DEFAULT_QUANTUM_LR})"
    )
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True,
        help="Generiere Plots (Standard: --plots)"
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=False,
        help="Speichere finalen Zustand (Standard: --no-save)"
    )
    parser.add_argument(
        "--load", action=argparse.BooleanOptionalAction, default=False,
        help="Lade gespeicherten Zustand (Standard: --no-load)"
    )
    # Optional: Argument für den Prompt, um nicht-interaktiven Modus zu ermöglichen
    parser.add_argument(
        "-p", "--prompt", type=str, default=None,
        help="Direkte Übergabe des Prompts (überspringt interaktive Eingabe)"
    )
    args = parser.parse_args()
    # --- Ende Argument Parser ---


    # API Verfügbarkeits-Checks (unverändert)
    if not gemini_api_available:
         print("\nACHTUNG: Google Generative AI SDK ist nicht installiert. Schritt 3 (Finale Antwort) wird nicht funktionieren.")
    elif not GEMINI_API_KEY:
        print("\nACHTUNG: GEMINI_API_KEY ist nicht gesetzt. Schritt 3 (Finale Antwort) wird fehlschlagen.")
        print("   Stellen Sie sicher, dass der API-Schlüssel als Umgebungsvariable 'GEMINI_API_KEY' verfügbar ist.")

    # Modus wählen: Interaktiv oder Direkt
    if args.prompt:
         print(f"\nVerarbeite übergebenen Prompt (Epochen={args.epochs})...")
         final_answer = execute_full_workflow(
                args.prompt,
                neuropersona_epochs=args.epochs,
                neuropersona_lr=args.lr,
                neuropersona_dr=args.dr,
                neuropersona_ri=args.ri,
                neuropersona_q_shots=args.q_shots,
                neuropersona_q_lr=args.q_lr,
                neuropersona_gen_plots=args.plots,
                neuropersona_save_state=args.save,
                neuropersona_load_state=args.load
            )
         print("\n" + "="*50)
         print(">>> Finale Antwort des Workflows: <<<")
         print("="*50)
         if final_answer: print(final_answer)
         else: print("Workflow konnte keine finale Antwort produzieren.")
         print("="*50 + "\n")

    else:
        # Interaktive Schleife für Benutzereingaben
        while True:
            try:
                prompt_text = f"\nGeben Sie Ihre Analysefrage ein (oder 'exit' zum Beenden) [E:{args.epochs}, LR:{args.lr:.3f}, QLR:{args.q_lr:.4f}, QS:{args.q_shots}]: "
                initial_user_prompt = input(prompt_text)
                if initial_user_prompt.lower() == 'exit':
                    break
                if not initial_user_prompt:
                    print("Bitte geben Sie eine Frage ein.")
                    continue

                # Rufe Workflow mit geparsten Argumenten auf
                final_answer = execute_full_workflow(
                    initial_user_prompt,
                    neuropersona_epochs=args.epochs,
                    neuropersona_lr=args.lr,
                    neuropersona_dr=args.dr,
                    neuropersona_ri=args.ri,
                    neuropersona_q_shots=args.q_shots,
                    neuropersona_q_lr=args.q_lr,
                    neuropersona_gen_plots=args.plots,
                    neuropersona_save_state=args.save,
                    neuropersona_load_state=args.load
                )

                # Ausgabe der finalen Antwort
                print("\n" + "="*50)
                print(">>> Finale Antwort des Workflows: <<<")
                print("="*50)
                if final_answer:
                    print(final_answer)
                else:
                    print("Workflow konnte keine finale Antwort produzieren (siehe Statusmeldungen oben).")
                print("="*50 + "\n")

            except EOFError: # Ctrl+D
                 print("\nBeende Programm.")
                 break
            except KeyboardInterrupt: # Ctrl+C
                print("\nBeende Programm.")
                break
            except Exception as e:
                print(f"\nEin unerwarteter Fehler ist im Hauptloop aufgetreten: {e}")
                traceback.print_exc()

    print("Orchestrator beendet.")