# -*- coding: utf-8 -*-
# Gemini Perception Unit – Hybrid-Modus (mit Fallback)
# ----------------------------------------------------
import pandas as pd
import random
import os
import re
import traceback

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

# Fallback-Kategorien (zur Notfallnutzung)
DEFAULT_CATEGORIES = [
    "Grundlagen", "Chancen", "Risiken", "Auswirkungen", "Ethik", "Anwendung",
    "Vergleich", "Kosten", "Markt", "Prognose", "Regulierung", "Fähigkeiten",
    "Wahrnehmung", "Innovation", "Technologie", "Wirtschaft", "Zukunft", "Gesellschaft"
]

# --- Gemini Prompt für intelligente Themenextraktion ---
GEMINI_SUBTOPIC_PROMPT = """
Extrahiere aus folgendem Thema sinnvolle Subthemen und generiere für jedes dieser Subthemen mindestens 2 konkrete Frage-Antwort-Paare. Ordne jede Frage in eine passende semantische Kategorie ein.

**Hauptthema:** "{theme}"

Erwünschtes Format: JSON-Liste von Objekten mit den Feldern:
- "Frage"
- "Antwort"
- "Kategorie"
"""

# --- Gemini-gestützte Subthemenanalyse ---
def query_gemini_for_semantic_structuring(prompt: str) -> list[dict]:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        raise RuntimeError("Gemini API nicht verfügbar.")

    final_prompt = GEMINI_SUBTOPIC_PROMPT.format(theme=prompt)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(final_prompt)
        text = response.text.strip()

        # Versuche JSON zu extrahieren (auch aus Markdown)
        json_text = re.search(r"\[.*\]", text, re.DOTALL)
        if json_text:
            data = eval(json_text.group())  # Sicherstellen, dass das wirklich JSON-ähnlich ist
        else:
            raise ValueError("Gemini-Antwort enthält kein erkennbares JSON.")

        # Validieren
        if not isinstance(data, list):
            raise ValueError("Erwartet wurde eine Liste von Frage-Antwort-Kategorien.")
        for entry in data:
            if not all(k in entry for k in ("Frage", "Antwort", "Kategorie")):
                raise ValueError("Eintrag unvollständig: " + str(entry))

        return data

    except Exception as e:
        print(f"[Gemini Fallback] Fehler bei Verarbeitung: {e}")
        traceback.print_exc()
        raise RuntimeError("Gemini konnte die Datenstruktur nicht liefern.")

# --- Fallback: Heuristische Simulation ---
def simulate_fallback_generation(prompt: str, min_entries: int = 12) -> list[dict]:
    keywords = re.findall(r'\b\w{4,}\b', prompt)
    theme = keywords[0] if keywords else "das Thema"

    question_templates = [
        f"Was sind die Hauptmerkmale von {theme}?",
        f"Welche Chancen bietet {theme}?",
        f"Welche Risiken sind mit {theme} verbunden?",
        f"Gibt es ethische Bedenken bei {theme}?",
        f"Was sind typische Anwendungsfälle für {theme}?",
        f"Wie beeinflusst {theme} die Gesellschaft?",
        f"Welche Kosten verursacht {theme}?",
        f"Wie steht es um die Regulierung von {theme}?",
    ]
    answer_templates = [
        "Eine zentrale Antwort ist ...",
        "Typische Herausforderungen sind ...",
        "Die Chancen liegen in ...",
        "Ein Risiko ist beispielsweise ...",
        "In der Praxis zeigt sich ...",
        "Gesellschaftlich wird diskutiert ...",
        "Kosten können stark variieren ...",
    ]

    results = []
    for _ in range(min_entries):
        q = random.choice(question_templates)
        a = random.choice(answer_templates)
        c = random.choice(DEFAULT_CATEGORIES)
        results.append({"Frage": q, "Antwort": a, "Kategorie": c})

    return results

# --- Umwandlung in DataFrame ---
def parse_to_dataframe(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    if not all(col in df.columns for col in ["Frage", "Antwort", "Kategorie"]):
        raise ValueError("Unvollständige Spalten in DataFrame.")
    return df

# --- Hauptfunktion ---
def generate_prompt_based_csv(user_prompt: str, min_entries: int = 12) -> pd.DataFrame:
    if not user_prompt or not user_prompt.strip():
        raise ValueError("Leerer Prompt.")

    print(f"[GeminiPerception] Analysiere: '{user_prompt}'")

    try:
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            print("[GeminiPerception] Versuche intelligente Generierung über Gemini...")
            structured_data = query_gemini_for_semantic_structuring(user_prompt)
        else:
            raise RuntimeError("Gemini nicht verfügbar.")
    except Exception:
        print("[GeminiPerception] Fallback wird verwendet (heuristische Simulation).")
        structured_data = simulate_fallback_generation(user_prompt, min_entries)

    return parse_to_dataframe(structured_data)

# --- Testlauf ---
if __name__ == "__main__":
    prompt = "Zukunft der künstlichen Intelligenz in der Medizin"
    df = generate_prompt_based_csv(prompt)
    print(df.head(10).to_string())
