# -*- coding: utf-8 -*-
# Filename: neuropersona_app.py
"""
Streamlit Frontend für den NeuroPersona Workflow (Quanten-Knoten Edition).
"""

import streamlit as st
import os
import time
import traceback
import pandas as pd
import importlib
import numpy as np # Importieren für np.number Check

# --- WICHTIG: Importiere die notwendigen Komponenten aus deinem Orchestrator ---
# Stelle sicher, dass der Orchestrator-Dateiname korrekt ist
# und dass der Code darin importierbar ist (kein Code außerhalb von if __name__ == "__main__" ausführen)
# KORREKTUR: Passe den Namen an dein Orchestrator-Skript an!
ORCHESTRATOR_MODULE = "orchestrator_full_qh_v1" # <-- DEIN ORCHESTRATOR-SKRIPTNAME (ohne .py)
CORE_MODULE_USED = "Unbekannt" # Wird unten aus dem Orchestrator geholt

# Fallback-Werte, falls Import fehlschlägt
DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_DECAY_RATE, DEFAULT_REWARD_INTERVAL = 30, 0.03, 0.015, 5
DEFAULT_QUANTUM_SHOTS, DEFAULT_QUANTUM_LR = 5, 0.01
GEMINI_API_KEY, gemini_api_available = None, False
MODEL_FILENAME = "neuropersona_state_fallback_ui.json" # UI Fallback
orchestrator_imported = False

try:
    orchestrator = importlib.import_module(ORCHESTRATOR_MODULE)
    execute_full_workflow = getattr(orchestrator, "execute_full_workflow")
    # Hole die Default-Werte für die UI direkt aus dem Orchestrator-Modul
    DEFAULT_EPOCHS = getattr(orchestrator, "DEFAULT_EPOCHS")
    DEFAULT_LEARNING_RATE = getattr(orchestrator, "DEFAULT_LEARNING_RATE")
    DEFAULT_DECAY_RATE = getattr(orchestrator, "DEFAULT_DECAY_RATE")
    DEFAULT_REWARD_INTERVAL = getattr(orchestrator, "DEFAULT_REWARD_INTERVAL")
    DEFAULT_QUANTUM_SHOTS = getattr(orchestrator, "DEFAULT_QUANTUM_SHOTS")
    DEFAULT_QUANTUM_LR = getattr(orchestrator, "DEFAULT_QUANTUM_LR")
    GEMINI_API_KEY = getattr(orchestrator, "GEMINI_API_KEY")
    gemini_api_available = getattr(orchestrator, "gemini_api_available")
    CORE_MODULE_USED = getattr(orchestrator, "NEUROPERSONA_CORE_MODULE") # Welches Core-Modul wird verwendet?
    # KORREKTUR: Importiere MODEL_FILENAME aus dem Orchestrator
    MODEL_FILENAME = getattr(orchestrator, "MODEL_FILENAME")

    orchestrator_imported = True
    print(f"Orchestrator '{ORCHESTRATOR_MODULE}' und Defaults erfolgreich importiert.")
    print(f"Verwendetes Core-Modul laut Orchestrator: '{CORE_MODULE_USED}'")
    print(f"Verwendeter Model Filename: '{MODEL_FILENAME}'")

except ImportError as e:
    st.error(f"Fehler beim Importieren des Orchestrator-Skripts ('{ORCHESTRATOR_MODULE}.py'): {e}")
    st.error("Stellen Sie sicher, dass die Datei im selben Verzeichnis liegt und keine Syntaxfehler enthält.")
    # Fallback-Werte werden verwendet
except AttributeError as e:
    st.error(f"Fehler: Ein erwarteter Parameter/Funktion wurde im Orchestrator ('{ORCHESTRATOR_MODULE}.py') nicht gefunden: {e}")
    st.error("Bitte überprüfen Sie die Orchestrator-Datei.")
    # KORREKTUR: Überprüfe, ob MODEL_FILENAME fehlt, ansonsten Fallback-Werte verwenden
    if 'MODEL_FILENAME' in str(e):
        st.warning(f"WARNUNG: 'MODEL_FILENAME' nicht im Orchestrator gefunden. Verwende Fallback für Hilfetext.")
        # MODEL_FILENAME behält seinen UI-Fallback-Wert
    else:
        orchestrator_imported = False # Setze auf False, wenn andere Attribute fehlen


# --- Streamlit UI Aufbau ---
st.set_page_config(page_title="NeuroPersona QN", layout="wide")
st.title(f"🧠 NeuroPersona Workflow (Quanten-Knoten)")
st.caption(f"Verwendetes Core-Modul: `{CORE_MODULE_USED}` (experimentell)")

# --- Sidebar für Parameter ---
st.sidebar.header("⚙️ Simulationsparameter")

epochs = st.sidebar.number_input(
    "Epochen:",
    min_value=1,
    max_value=10000, # Erhöhtes Maximum
    value=DEFAULT_EPOCHS,
    step=10,
    help="Anzahl der Simulationszyklen."
)

st.sidebar.subheader("Klassische Parameter")
learning_rate = st.sidebar.slider(
    "Klass. Lernrate (Gewichte):",
    min_value=0.001,
    max_value=0.5,
    value=float(DEFAULT_LEARNING_RATE), # Stelle sicher, dass es float ist
    step=0.001,
    format="%.3f",
    help="Lernrate für klassische Verbindungsgewichte."
)
decay_rate = st.sidebar.slider(
    "Klass. Decay Rate (Gewichte):",
    min_value=0.0,
    max_value=0.1,
    value=float(DEFAULT_DECAY_RATE),
    step=0.001,
    format="%.3f",
    help="Zerfallsrate für klassische Gewichte pro Epoche."
)
reward_interval = st.sidebar.number_input(
    "Reward Intervall:",
    min_value=1,
    value=DEFAULT_REWARD_INTERVAL,
    step=1,
    help="Nach wie vielen Epochen Verstärkungslernen angewendet wird."
)

st.sidebar.subheader("Quanten Parameter")
q_shots = st.sidebar.number_input(
    "Q Shots / Knoten:",
    min_value=1,
    max_value=100,
    value=DEFAULT_QUANTUM_SHOTS,
    step=1,
    help="Anzahl Quantenmessungen pro Knotenaktivierung (beeinflusst Genauigkeit vs. Geschwindigkeit)."
)
q_lr = st.sidebar.slider(
    "Q Param Lernrate:",
    min_value=0.0001,
    max_value=0.5,
    value=float(DEFAULT_QUANTUM_LR),
    step=0.001,
    format="%.4f",
    help="Lernrate für die internen Quantenparameter der Knoten."
)

st.sidebar.subheader("Steuerung")
gen_plots = st.sidebar.checkbox("Plots generieren", value=True, help="Speichert Analyseplots im `plots_...` Ordner.")
# KORREKTUR: Verwende die importierte/Fallback MODEL_FILENAME Variable
save_state = st.sidebar.checkbox("Finalen Zustand speichern", value=False, help=f"Speichert den Netzwerkzustand in `{MODEL_FILENAME}`.")
load_state = st.sidebar.checkbox("Zustand laden", value=False, help=f"Versucht, Netzwerkzustand aus `{MODEL_FILENAME}` zu laden (überschreibt Init).")

# --- Hauptbereich ---
st.markdown("---")
user_prompt = st.text_area(
    "❓ Geben Sie Ihre Analysefrage / Ihr Thema ein:",
    height=150,
    value="Wie kann die Menschheit gleichzeitig exponentielles technologisches Wachstum (insbesondere in KI und Biotechnologie) fördern UND globale existenzielle Risiken (durch dieselben Technologien und andere Faktoren wie Klimawandel oder Pandemien) minimieren? Analysiere die inhärenten Spannungen, potenziellen Synergien und notwendigen Paradigmenwechsel."
)

start_button = st.button("🚀 Starte NeuroPersona Analyse")
st.markdown("---")

# --- Status und Ergebnis Anzeige ---
status_container = st.container() # Container für Statusmeldungen
result_placeholder = st.empty() # Platzhalter für das finale Ergebnis

# Initialisiere Session State für Ergebnis
if 'final_answer' not in st.session_state:
    st.session_state.final_answer = None
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = None

# --- Logik beim Button Klick ---
if start_button and orchestrator_imported:
    if not user_prompt:
        st.warning("Bitte geben Sie eine Frage oder ein Thema ein.")
    else:
        st.session_state.final_answer = None # Reset altes Ergebnis
        result_placeholder.empty() # Leere alten Platzhalter

        # Sammle Parameter
        current_params = {
            'user_prompt': user_prompt,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'decay_rate': decay_rate,
            'reward_interval': reward_interval,
            'q_shots': q_shots,
            'q_lr': q_lr,
            'gen_plots': gen_plots,
            'save_state': save_state,
            'load_state': load_state,
        }
        st.session_state.last_run_params = current_params # Speichere Parameter für Anzeige

        try:
            # Verwende st.status für einen Block, der den Fortschritt anzeigt
            with st.status(f"Führe NeuroPersona Workflow für '{user_prompt[:30]}...' aus", expanded=True) as status:

                # Definieren eine Callback-Funktion, die st.status aktualisiert
                def update_status_widget(message):
                    status.write(f"> {message}") # Schreibe Status ins Widget

                st.info("Workflow gestartet...")
                update_status_widget("Initialisiere...")

                # ---- EIGENTLICHER WORKFLOW AUFRUF ----
                final_answer_text = execute_full_workflow(
                    user_prompt=user_prompt,
                    neuropersona_epochs=epochs,
                    neuropersona_lr=learning_rate,
                    neuropersona_dr=decay_rate,
                    neuropersona_ri=reward_interval,
                    neuropersona_q_shots=q_shots,
                    neuropersona_q_lr=q_lr,
                    neuropersona_gen_plots=gen_plots,
                    neuropersona_save_state=save_state,
                    neuropersona_load_state=load_state,
                    status_callback=update_status_widget # Übergib den Status-Widget-Callback
                )
                # ---------------------------------------

                st.session_state.final_answer = final_answer_text # Speichere Ergebnis
                status.update(label="Workflow abgeschlossen!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Ein unerwarteter Fehler ist während des Workflows aufgetreten: {e}")
            st.error(traceback.format_exc())
            st.session_state.final_answer = f"FEHLER: {e}\n\n{traceback.format_exc()}" # Zeige Fehler als Ergebnis an

elif start_button and not orchestrator_imported:
    st.error("Orchestrator konnte nicht importiert werden. Ausführung nicht möglich.")

# --- Ergebnis anzeigen (wird nach dem Lauf bei der nächsten Interaktion/Rerun angezeigt) ---
if st.session_state.final_answer:
    st.markdown("---")
    st.subheader("📋 Finale Antwort des Workflows")
    # Zeige Parameter des letzten Laufs an
    if st.session_state.last_run_params:
         params_str = ", ".join(f"{k}={v}" for k, v in st.session_state.last_run_params.items() if k != 'user_prompt')
         with st.expander("Verwendete Parameter für diese Antwort"):
             st.json(st.session_state.last_run_params) # Zeigt Parameter als JSON

    # Zeige die Antwort (kann Markdown enthalten)
    result_placeholder.markdown(st.session_state.final_answer)


# --- Optional: API Key Warnung in Sidebar ---
st.sidebar.markdown("---")
if orchestrator_imported:
    if not GEMINI_API_KEY and gemini_api_available:
        st.sidebar.warning("⚠️ Gemini API Key fehlt. Finale Interpretation wird übersprungen.")
    elif not gemini_api_available:
        st.sidebar.warning("⚠️ `google-generativeai` fehlt. Finale Interpretation deaktiviert.")
    elif GEMINI_API_KEY and gemini_api_available:
         st.sidebar.success("✅ Gemini API konfiguriert.")

st.sidebar.caption(f"NeuroPersona Core: {CORE_MODULE_USED}")