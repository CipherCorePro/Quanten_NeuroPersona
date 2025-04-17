# -*- coding: utf-8 -*-
# Filename: neuropersona_core_quantum_node_v2_multi_qubit.py # <--- NEUER DATEINAME & VERSION
# Description: NeuroPersona Core mit Multi-Qubit Quantenknoten (10 Qubits pro Knoten)
#              und quanten-moduliertem Hebbian Learning. HOCH EXPERIMENTELL.
#              **MIT** integriertem Quantum Emergence Tracker (QET) Logging.

# --- Imports ---
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg') # Use 'Agg' if Tkinter is not available or needed
import matplotlib.pyplot as plt
import math
from collections import Counter, deque
import json
import importlib
import sqlite3
import os
import time
import threading
import traceback
from typing import Optional, Callable, List, Tuple, Dict, Any
import uuid # Import für Logger
from datetime import datetime # Import für Logger

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
entry_widgets = {} # For GUI elements

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warnung: networkx nicht gefunden. Netzwerk-Graph-Plot wird nicht erstellt.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warnung: tqdm nicht gefunden. Fortschrittsbalken nicht verfügbar.")
    # Define a dummy tqdm if not available
    def tqdm(iterable, *args, **kwargs): return iterable

# ########################################################################
# # --- Quanten-Komponenten (Fokus auf Multi-Qubit Knoten) ---
# ########################################################################

# === Basis-Gates (Single Qubit) ===
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex) # Projector |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=complex) # Projector |1><1|

def _ry(theta: float) -> np.ndarray:
    """Creates an RY gate matrix."""
    cos_t = np.cos(theta / 2)
    sin_t = np.sin(theta / 2)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

def _rz(phi: float) -> np.ndarray:
    """Creates an RZ gate matrix."""
    exp_m = np.exp(-1j * phi / 2)
    exp_p = np.exp(1j * phi / 2)
    return np.array([[exp_m, 0], [0, exp_p]], dtype=complex)

# === Multi-Qubit Gate Helpers ===
def _apply_gate(state_vector: np.ndarray, gate: np.ndarray, target_qubit: int, num_qubits: int) -> np.ndarray:
    """
    Applies a single-qubit gate to a target qubit in a multi-qubit system
    using Kronecker products.

    Args:
        state_vector (np.ndarray): The current state vector of the system.
        gate (np.ndarray): The 2x2 single-qubit gate matrix.
        target_qubit (int): The index (0 to num_qubits-1) of the target qubit.
        num_qubits (int): The total number of qubits in the system.

    Returns:
        np.ndarray: The updated state vector after applying the gate.

    Raises:
        ValueError: If gate dimensions, target qubit index, or state vector size are invalid.
    """
    if gate.shape != (2, 2):
        raise ValueError("Gate must be a 2x2 matrix.")
    if not (0 <= target_qubit < num_qubits):
        raise ValueError(f"Target qubit {target_qubit} is out of range [0, {num_qubits-1}]")
    expected_len = 2**num_qubits
    if len(state_vector) != expected_len:
         raise ValueError(f"State vector length {len(state_vector)} does not match {num_qubits} qubits (expected {expected_len})")

    # Build the full operator matrix for the entire system
    op_list = [I] * num_qubits # Start with identity on all qubits
    op_list[target_qubit] = gate # Place the actual gate at the target position

    # Compute the Kronecker product of all operators
    full_matrix = op_list[0]
    for i in range(1, num_qubits):
        full_matrix = np.kron(full_matrix, op_list[i])

    # Apply the full matrix to the state vector
    new_state = np.dot(full_matrix, state_vector)
    return new_state

def _apply_cnot(state_vector: np.ndarray, control_qubit: int, target_qubit: int, num_qubits: int) -> np.ndarray:
    """
    Applies a CNOT gate to control and target qubits in a multi-qubit system.

    Args:
        state_vector (np.ndarray): The current state vector.
        control_qubit (int): Index of the control qubit.
        target_qubit (int): Index of the target qubit.
        num_qubits (int): Total number of qubits.

    Returns:
        np.ndarray: The updated state vector.

    Raises:
        ValueError: If qubit indices are invalid, identical, or state vector size is wrong.
    """
    if not (0 <= control_qubit < num_qubits and 0 <= target_qubit < num_qubits):
        raise ValueError("Control or target qubit index out of range.")
    if control_qubit == target_qubit:
        raise ValueError("Control and target qubits must be different.")
    expected_len = 2**num_qubits
    if len(state_vector) != expected_len:
         raise ValueError(f"State vector length {len(state_vector)} does not match {num_qubits} qubits (expected {expected_len})")

    # Build the CNOT matrix for the full system using projectors:
    # CNOT = |0><0|_c (x) I_t + |1><1|_c (x) X_t
    op_list_p0 = [I] * num_qubits
    op_list_p1 = [I] * num_qubits

    op_list_p0[control_qubit] = P0 # |0><0| projector on control qubit
    op_list_p1[control_qubit] = P1 # |1><1| projector on control qubit
    op_list_p1[target_qubit] = X  # Pauli-X on target qubit only if control is |1>

    # Calculate the Kronecker products for both terms
    term0_matrix = op_list_p0[0]
    term1_matrix = op_list_p1[0]
    for i in range(1, num_qubits):
        term0_matrix = np.kron(term0_matrix, op_list_p0[i])
        term1_matrix = np.kron(term1_matrix, op_list_p1[i])

    # The full CNOT matrix is the sum of the two terms
    cnot_matrix = term0_matrix + term1_matrix

    # Apply the matrix to the state vector
    new_state = np.dot(cnot_matrix, state_vector)
    return new_state

# === NEU/KORRIGIERT: Quantum Emergence Tracker (QET) Logger Klasse ===
class QuantumStepLogger:
    """
    Ein Live-Logger, der jeden Quantenschritt (Gate-Anwendung, Messung)
    innerhalb einer QuantumNodeSystem-Aktivierung dokumentiert.
    Erzeugt eine detaillierte Liste von Ereignissen, die als JSON gespeichert wird.
    """
    def __init__(self, log_id: Optional[str] = None):
        """
        Initialisiert den Logger.

        Args:
            log_id (Optional[str]): Eine eindeutige ID für diesen Log-Lauf.
                                    Wenn None, wird eine UUID generiert.
        """
        self.log_id: str = log_id or str(uuid.uuid4())
        self.entries: List[Dict[str, Any]] = []

    def log_gate(self, shot: int, op_index: int, op_type: str, qubits: Tuple[int, ...], params: Tuple[float, ...] = ()):
        """
        Protokolliert die Anwendung eines Quantengates.

        Args:
            shot (int): Der Index des aktuellen Simulations-Shots (beginnend bei 0).
            op_index (int): Der Index der Operation innerhalb der PQC-Sequenz für diesen Shot.
            op_type (str): Der Typ des Gates (z.B. 'H', 'RY', 'CNOT').
            qubits (Tuple[int, ...]): Ein Tupel der Qubit-Indizes, auf die das Gate wirkt
                                    (z.B. (target,) für Single-Qubit, (control, target) für CNOT).
            params (Tuple[float, ...]): Ein Tupel von Parametern für das Gate (z.B. Rotationswinkel).
                                        Standardmäßig leer.
        """
        # Stelle sicher, dass Parameter Floats sind (oder leer) und endlich
        safe_params = tuple(float(p) if isinstance(p, (int, float, np.number)) and np.isfinite(p) else 0.0 for p in params)

        self.entries.append({
            "timestamp": datetime.utcnow().isoformat() + "Z", # Füge 'Z' für UTC hinzu
            "type": "gate",
            "shot": int(shot), # Sicherstellen, dass es int ist
            "op_index": int(op_index), # Sicherstellen, dass es int ist
            "op_type": str(op_type), # Sicherstellen, dass es str ist
            "qubits": list(int(q) for q in qubits), # Konvertiere zu Liste von ints für JSON Kompatibilität
            "params": list(safe_params)  # Konvertiere zu Liste für JSON Kompatibilität
        })

    # --- KORRIGIERTE log_measurement Methode ---
    def log_measurement(self, shot: int, state_index: Any, hamming_weight: Any, num_qubits_for_binary: int):
        """
        Protokolliert das Ergebnis einer Quantenmessung am Ende eines Shots.
        Stellt sicher, dass die Typen korrekt sind und behandelt Fehler.

        Args:
            shot (int): Der Index des Simulations-Shots.
            state_index (Any): Der gemessene Zustand als Dezimalzahl (sollte int sein).
            hamming_weight (Any): Das Hamming-Gewicht des gemessenen Zustands (sollte int sein).
            num_qubits_for_binary (int): Die Anzahl der Qubits, um die Binärdarstellung korrekt aufzufüllen.
        """
        try:
            # Konvertiere explizit zu int, bevor sie verwendet werden
            state_idx_int = int(state_index)
            hamming_w_int = int(hamming_weight)
            num_q_int = int(num_qubits_for_binary)

            # Generiere Binärstring mit führenden Nullen passend zur Qubit-Zahl
            binary_representation = format(state_idx_int, f'0{num_q_int}b')

            # Erstelle den Log-Eintrag
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z", # Füge 'Z' für UTC hinzu
                "type": "measurement",
                "shot": int(shot),
                "result_index": state_idx_int,
                "binary": binary_representation,
                "hamming_weight": hamming_w_int
            }
            self.entries.append(log_entry)
        except (ValueError, TypeError) as e:
             # Fange potenzielle Fehler bei der Typumwandlung oder Formatierung ab
             print(f"FEHLER beim Erstellen des Measurement-Log-Eintrags (Shot {shot}): {e}. state_index='{state_index}', hamming_weight='{hamming_weight}'")
             # Füge einen Fehler-Eintrag hinzu, um das Problem im Log zu sehen
             self.entries.append({
                 "timestamp": datetime.utcnow().isoformat() + "Z",
                 "type": "logging_error",
                 "shot": int(shot),
                 "error_message": f"Failed to log measurement: {e}",
                 "original_data": {
                     "state_index": repr(state_index), # repr() für Debugging
                     "hamming_weight": repr(hamming_weight),
                     "num_qubits": num_qubits_for_binary
                 }
             })
        except Exception as e:
             # Fange alle anderen unerwarteten Fehler ab
             print(f"UNERWARTETER FEHLER beim Loggen der Messung (Shot {shot}): {e}")
             traceback.print_exc()
             self.entries.append({
                 "timestamp": datetime.utcnow().isoformat() + "Z",
                 "type": "logging_error",
                 "shot": int(shot),
                 "error_message": f"Unexpected error logging measurement: {e}",
                 "original_data": {
                     "state_index": repr(state_index),
                     "hamming_weight": repr(hamming_weight),
                     "num_qubits": num_qubits_for_binary
                 }
             })

    # --- KORRIGIERTE save Methode ---
    def save(self, filename: Optional[str] = None):
        """
        Speichert alle protokollierten Einträge in einer JSON-Datei.
        Stellt sicher, dass die Datei korrekt geschrieben und geschlossen wird,
        auch bei Fehlern während des Schreibvorgangs.

        Args:
            filename (Optional[str]): Der gewünschte Dateiname. Wenn None, wird ein
                                      Standardname mit der Log-ID generiert.
        """
        filename = filename or f"quantum_log_{self.log_id}.json"
        log_dir = "quantum_logs"
        try:
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(log_dir, filename)
        except OSError as e:
             print(f"FEHLER beim Erstellen des Log-Verzeichnisses '{log_dir}': {e}")
             return

        # Versuche, die Datei zu schreiben
        file_handle = None
        try:
            # Öffne die Datei zum Schreiben
            file_handle = open(filepath, 'w', encoding='utf-8')

            # Schreibe das öffnende '[' manuell, um Flexibilität zu haben
            file_handle.write('[\n')

            # Schreibe jeden Eintrag einzeln, um Fehler früher zu erkennen
            num_entries = len(self.entries)
            for i, entry in enumerate(self.entries):
                try:
                    # Schreibe den Eintrag als JSON, füge Einrückung hinzu
                    json.dump(entry, file_handle, indent=2, ensure_ascii=False)
                    # Füge ein Komma hinzu, wenn es nicht der letzte Eintrag ist
                    if i < num_entries - 1:
                        file_handle.write(',\n')
                    else:
                        file_handle.write('\n') # Letzter Eintrag, nur Zeilenumbruch
                except (TypeError, ValueError) as entry_err:
                     # Fehler beim Serialisieren eines spezifischen Eintrags
                     print(f"FEHLER beim Serialisieren des Log-Eintrags {i} in '{filepath}': {entry_err}")
                     print(f"--> Fehlerhafter Eintrag (repr): {repr(entry)}")
                     # Schreibe einen Fehlerplatzhalter in die Datei
                     error_entry = {
                         "type": "serialization_error",
                         "index": i,
                         "error_message": str(entry_err),
                         "problematic_entry_repr": repr(entry)
                     }
                     # Versuche, den Fehlerplatzhalter zu schreiben
                     try:
                         json.dump(error_entry, file_handle, indent=2, ensure_ascii=False)
                         if i < num_entries - 1: file_handle.write(',\n')
                         else: file_handle.write('\n')
                     except Exception as write_err_err:
                         print(f"KONNTE FEHLERPLATZHALTER nicht schreiben: {write_err_err}")
                         file_handle.write(f'{{"error": "Failed to write error placeholder for entry {i}"}}')
                         if i < num_entries - 1: file_handle.write(',\n')
                         else: file_handle.write('\n')


            # Schreibe das schließende ']'
            file_handle.write(']')

            # Erzwinge das Schreiben des Puffers auf das Betriebssystem
            file_handle.flush()
            # Erzwinge das Schreiben vom Betriebssystem auf die Festplatte
            os.fsync(file_handle.fileno())

            # Optional: Erfolgsmeldung nach erfolgreichem Schreiben und Synchronisieren
            # print(f"[QET Logger] Log erfolgreich geschrieben und synchronisiert: {filepath}")

        except IOError as e:
            print(f"FEHLER beim Schreiben/Speichern des QET Logs '{filepath}': {e}")
            traceback.print_exc()
        except Exception as e:
             # Fange andere unerwartete Fehler beim Schreiben ab
             print(f"UNERWARTETER FEHLER beim Speichern des QET Logs '{filepath}': {e}")
             traceback.print_exc()
        finally:
            # Stelle sicher, dass die Datei IMMER geschlossen wird, auch wenn Fehler auftraten
            if file_handle is not None:
                try:
                    file_handle.close()
                    # print(f"[QET Logger] Datei geschlossen: {filepath}") # Debugging
                except Exception as close_err:
                    print(f"FEHLER beim Schließen der Log-Datei '{filepath}': {close_err}")



# === The Quantum System for ONE Node (MULTI-QUBIT) ===
class QuantumNodeSystem:
    """
    Simulates the quantum-based activation for a single NeuroPersona node
    using a multi-qubit system (e.g., 10 qubits).
    Utilizes a Parametrized Quantum Circuit (PQC) with rotations and entanglement.
    INTEGRIERT den QuantumStepLogger zur detaillierten Protokollierung.
    """
    def __init__(self, num_qubits: int = 10, initial_params: Optional[np.ndarray] = None):
        """
        Initializes the quantum system for a node.

        Args:
            num_qubits (int): The number of qubits dedicated to this node.
            initial_params (Optional[np.ndarray]): Starting values for the internal
                                                    parameters (expects shape (num_qubits * 2,)).
        """
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive.")
        self.num_qubits = num_qubits
        # Each qubit has 2 parameters: one for RY (theta), one for RZ (phi)
        self.num_params = num_qubits * 2

        if initial_params is None:
            # Initialize parameters randomly, e.g., angles between 0 and pi
            self.params = np.random.rand(self.num_params) * np.pi
        elif isinstance(initial_params, np.ndarray) and initial_params.shape == (self.num_params,):
             # Ensure parameters are finite upon initialization
             if not np.all(np.isfinite(initial_params)):
                  raise ValueError("Initial parameters contain non-finite values.")
             self.params = np.array(initial_params, dtype=float)
        else:
            raise ValueError(f"Shape of initial_params ({initial_params.shape if isinstance(initial_params, np.ndarray) else type(initial_params)})"
                             f" does not match required num_params ({self.num_params}) for {self.num_qubits} qubits")

        # Initialize the internal state vector to |00...0>
        self.state_vector_size = 2**self.num_qubits
        self.state_vector = np.zeros(self.state_vector_size, dtype=complex)
        self.state_vector[0] = 1.0 + 0j

    def _build_pqc_ops(self, input_strength: float) -> List[Tuple]:
        """
        Defines the sequence of quantum operations (gates) for the PQC of this node.
        Structure: H on all -> RY(theta_i) on all -> RZ(phi_i) on all -> CNOT chain.
            - theta_i is influenced by input_strength * params[2*i].
            - phi_i is determined by params[2*i + 1].

        Args:
            input_strength (float): The classical weighted input sum from other nodes.

        Returns:
            List[Tuple]: A list of operations, e.g., ('H', target_q), ('RY', target_q, angle), ('CNOT', control_q, target_q).
        """
        ops = []
        # Scale input to a reasonable range, e.g., [-1, 1] using tanh
        scaled_input = np.tanh(input_strength)

        # Layer 1: Hadamard gate on all qubits to create superposition
        for i in range(self.num_qubits):
            ops.append(('H', i))

        # Layer 2: Parameterized RY rotations based on input and parameters
        for i in range(self.num_qubits):
            # Stelle sicher, dass das Ergebnis endlich ist
            theta = scaled_input * self.params[2 * i] # Parameter index 0, 2, 4, ...
            if not np.isfinite(theta): theta = 0.0 # Fallback auf 0 bei ungültigem Winkel
            ops.append(('RY', i, theta))

        # Layer 3: Parameterized RZ rotations based on parameters
        for i in range(self.num_qubits):
            phi = self.params[2 * i + 1] # Parameter index 1, 3, 5, ...
            if not np.isfinite(phi): phi = 0.0 # Fallback
            ops.append(('RZ', i, phi))

        # Layer 4: Entangling layer using a chain of CNOT gates
        # Connects qubit i to qubit (i+1) mod N
        if self.num_qubits > 1:
            for i in range(self.num_qubits):
                control_q = i
                target_q = (i + 1) % self.num_qubits
                ops.append(('CNOT', control_q, target_q))

        # Additional layers could be added here for more complex circuits
        return ops

    # --- KORRIGIERTE activate Methode ---
    def activate(self, input_strength: float, n_shots: int = 3, node_label: Optional[str] = None) -> float:
        """
        Executes the multi-qubit PQC and returns the node's activation level.
        Activation is defined as the average Hamming weight (number of |1>s)
        across all measurements, normalized by the number of qubits.
        Logs detailed steps using QuantumStepLogger. Ensures logger is saved even on error.

        Args:
            input_strength (float): The classical weighted input sum.
            n_shots (int): The number of simulated measurements to estimate the activation.
            node_label (Optional[str]): The label of the node being activated, used for log filename.

        Returns:
            float: The estimated activation level (a probability between 0.0 and 1.0).
                   Also saves a detailed JSON log file for this activation.
        """
        if not np.isfinite(input_strength):
            print(f"WARNUNG: Ungültige input_strength ({input_strength}) für Node {node_label or 'unbekannt'}. Setze auf 0.0.")
            input_strength = 0.0

        # --- QET Logging: Initialisierung ---
        activation_log_id = f"{node_label or 'unknown_node'}_act_nq{self.num_qubits}_{time.strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:4]}"
        logger = QuantumStepLogger(log_id=activation_log_id)

        pqc_ops = self._build_pqc_ops(input_strength)
        total_hamming_weight = 0
        activation_prob = 0.0 # Default value

        # --- Gesamte Simulation in try/finally Block, um Logger-Speichern sicherzustellen ---
        try:
            for shot in range(n_shots):
                current_state = np.zeros(self.state_vector_size, dtype=complex)
                current_state[0] = 1.0

                for op_index, op in enumerate(pqc_ops):
                    op_type = op[0]
                    gate_applied_successfully = False # Flag um Gate Anwendung zu tracken
                    try:
                        # --- QET Gate Logging ---
                        if op_type in ['H', 'RY', 'RZ']:
                            target_q = op[1]
                            angle = float(op[2]) if len(op) > 2 and isinstance(op[2], (float, int, np.number)) and np.isfinite(op[2]) else 0.0
                            gate_params = (angle,) if op_type in ['RY', 'RZ'] else ()
                            logger.log_gate(shot, op_index, op_type, (target_q,), gate_params)
                        elif op_type == 'CNOT':
                            control_q, target_q = op[1], op[2]
                            logger.log_gate(shot, op_index, op_type, (control_q, target_q))

                        # --- Eigentliche Gate-Anwendung ---
                        if op_type == 'H':
                            target_q = op[1]
                            current_state = _apply_gate(current_state, H, target_q, self.num_qubits)
                        elif op_type == 'RY':
                            target_q, angle = op[1], op[2]
                             # Stelle sicher, dass der Winkel endlich ist für die RY-Matrix
                            if not np.isfinite(angle): angle = 0.0
                            current_state = _apply_gate(current_state, _ry(angle), target_q, self.num_qubits)
                        elif op_type == 'RZ':
                            target_q, angle = op[1], op[2]
                             # Stelle sicher, dass der Winkel endlich ist für die RZ-Matrix
                            if not np.isfinite(angle): angle = 0.0
                            current_state = _apply_gate(current_state, _rz(angle), target_q, self.num_qubits)
                        elif op_type == 'CNOT':
                            control_q, target_q = op[1], op[2]
                            current_state = _apply_cnot(current_state, control_q, target_q, self.num_qubits)

                        gate_applied_successfully = True # Markiere als erfolgreich

                        # Prüfe *nach* Gate-Anwendung auf NaNs/Infs
                        if not np.all(np.isfinite(current_state)):
                             print(f"FEHLER: Nicht-finite Werte im State Vector nach Gate {op_type} (Index {op_index}, Shot {shot}) Node {node_label or 'unbekannt'}!")
                             current_state.fill(0.0)
                             current_state[0] = 1.0
                             gate_applied_successfully = False # Markiere als nicht erfolgreich
                             break # Breche die Gate-Schleife für diesen Shot ab

                    except (ValueError, IndexError, TypeError, Exception) as e: # Breiterer Exception-Fang
                        print(f"FEHLER bei Gate-Anwendung: Op={op} (Index {op_index}, Shot {shot}) Node {node_label or 'unbekannt'}. State Shape={current_state.shape}. Error: {e}")
                        traceback.print_exc() # Gebe Traceback aus für Debugging
                        current_state.fill(0.0)
                        current_state[0] = 1.0
                        gate_applied_successfully = False # Markiere als nicht erfolgreich
                        break # Breche die Gate-Schleife für diesen Shot ab

                # --- Nach Gate-Anwendung (pro Shot) ---
                if not gate_applied_successfully: # Wenn ein Gate im Shot fehlgeschlagen ist, überspringe Messung
                    continue

                probabilities = np.abs(current_state)**2
                prob_sum = np.sum(probabilities)

                if not np.isfinite(prob_sum) or not np.all(np.isfinite(probabilities)):
                     print(f"FEHLER: Nicht-finite Wahrscheinlichkeiten (Sum={prob_sum}) vor Messung (Shot {shot}, Node {node_label or 'unbekannt'}). Setze auf |0...0>.")
                     probabilities.fill(0.0)
                     probabilities[0] = 1.0
                     prob_sum = 1.0 # Korrigiere Summe für Normalisierung

                if not np.isclose(prob_sum, 1.0, atol=1e-7):
                     if prob_sum <= 1e-9:
                         probabilities.fill(0.0)
                         probabilities[0] = 1.0
                     else:
                         probabilities = np.maximum(0, probabilities) / np.sum(np.maximum(0, probabilities))
                # Erneute Prüfung nach Normalisierung
                probabilities = np.maximum(0, probabilities)
                probabilities /= np.sum(probabilities)

                try:
                     measured_state_index = np.random.choice(range(self.state_vector_size), p=probabilities)
                     # KORRIGIERT: Explizite int Konvertierung vor Hamming-Berechnung
                     state_idx_int = int(measured_state_index)
                     binary_string = format(state_idx_int, f'0{self.num_qubits}b')
                     hamming_weight = binary_string.count('1')
                     total_hamming_weight += hamming_weight
                     logger.log_measurement(shot, state_idx_int, hamming_weight, self.num_qubits) # Logge korrekte int-Werte
                except (ValueError, TypeError) as e:
                     print(f"FEHLER bei Messung (Shot {shot}, Node {node_label or 'unbekannt'}): Probs sum={np.sum(probabilities):.6f}. Error: {e}. Using argmax.")
                     measured_state_index = np.argmax(probabilities)
                     # KORRIGIERT: Explizite int Konvertierung im Fallback
                     state_idx_int = int(measured_state_index)
                     binary_string = format(state_idx_int, f'0{self.num_qubits}b')
                     hamming_weight = binary_string.count('1')
                     total_hamming_weight += hamming_weight
                     logger.log_measurement(shot, state_idx_int, hamming_weight, self.num_qubits) # Logge korrekte int-Werte

            # --- Nach allen Shots ---
            if n_shots > 0 and self.num_qubits > 0:
                 activation_prob = float(np.clip(total_hamming_weight / (n_shots * self.num_qubits), 0.0, 1.0))
            else:
                 activation_prob = 0.0

        except Exception as activation_error:
             print(f"FATAL ERROR during QuantumNodeSystem.activate loop for node {node_label or 'unknown'}: {activation_error}")
             traceback.print_exc()
             activation_prob = 0.0 # Setze Aktivierung auf 0 im Fehlerfall

        finally:
            # --- QET Logging: Speichern (IMMER ausführen) ---
            log_filename = f"quantum_log_{activation_log_id}.json"
            # Speichere den Logger *bevor* die Funktion verlassen wird
            logger.save(log_filename)

        # Finale Prüfung auf Endlichkeit der Aktivierungswahrscheinlichkeit
        if not np.isfinite(activation_prob):
             print(f"WARNUNG: Endgültige activation_prob ist nicht endlich ({activation_prob}) für Node {node_label or 'unknown'}. Setze auf 0.0.")
             activation_prob = 0.0

        return activation_prob

    # Die Methoden update_internal_params, get_params, set_params bleiben wie im vorherigen Codeblock
    def update_internal_params(self, delta_params: np.ndarray):
        """
        Updates the internal parameters (angles) of the quantum system based on learning feedback.
        Also applies clipping to keep angles within a reasonable range (e.g., [0, 2*pi]).

        Args:
            delta_params (np.ndarray): An array containing the changes to be applied to each parameter.
                                      Must have the same shape as self.params.
        """
        if delta_params.shape != self.params.shape:
             print(f"WARNING: Shape mismatch during parameter update ({delta_params.shape} vs {self.params.shape}). Update skipped.")
             return
        if not np.all(np.isfinite(delta_params)):
             print(f"WARNING: Non-finite values detected in delta_params ({delta_params}). Update skipped.")
             return

        new_params = self.params + delta_params
        if not np.all(np.isfinite(new_params)):
            print(f"WARNING: Non-finite values resulted from parameter update. Attempting recovery.")
            self.params = np.where(np.isfinite(new_params), new_params, self.params)
        else:
            self.params = new_params

        self.params = np.clip(self.params, 0, 2 * np.pi)

    def get_params(self) -> np.ndarray:
        """Returns a copy of the current internal parameters."""
        safe_params = self.params.copy()
        if not np.all(np.isfinite(safe_params)):
            print(f"WARNUNG: get_params() für Node {getattr(self, 'label', 'unbekannt')} enthält nicht-finite Werte. Korrigiere...")
            safe_params = np.nan_to_num(safe_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
        return safe_params


    def set_params(self, params: np.ndarray):
        """
        Sets the internal parameters directly.

        Args:
            params (np.ndarray): The new parameter array. Must match self.num_params shape.
        """
        if isinstance(params, np.ndarray) and params.shape == self.params.shape:
            if np.all(np.isfinite(params)):
                self.params = np.clip(params, 0, 2 * np.pi)
            else:
                print(f"WARNING: Attempted to set non-finite parameters. Parameters not changed.")
        else:
            print(f"WARNING: Shape mismatch when setting parameters ({params.shape if isinstance(params, np.ndarray) else type(params)} vs {self.params.shape}). Parameters not changed.")

# --- Global Instances ---
CURRENT_EMOTION_STATE: Dict[str, float] = {}
persistent_memory_manager: Optional['PersistentMemoryManager'] = None
activation_history: Dict[str, deque] = {} # Stores activation history for nodes

# --- Constants ---
NUM_QUBITS_PER_NODE = 4 # Number of qubits per quantum node
MODEL_FILENAME = f"neuropersona_quantum_node_mq{NUM_QUBITS_PER_NODE}_state_v2.json" # State file
SETTINGS_FILENAME = f"neuropersona_gui_settings_mq{NUM_QUBITS_PER_NODE}_v2.json" # GUI settings file
PLOTS_FOLDER = f"plots_quantum_node_mq{NUM_QUBITS_PER_NODE}_v2" # Folder for generated plots
PERSISTENT_MEMORY_DB = f"neuropersona_longterm_memory_mq{NUM_QUBITS_PER_NODE}_v2.db" # Long-term memory DB

# Simulation Parameters (tuned for multi-qubit complexity)
DEFAULT_EPOCHS = 20 # Reduced number of epochs due to increased computation time
DEFAULT_LEARNING_RATE = 0.02 # Base learning rate for classical weights
DEFAULT_DECAY_RATE = 0.01 # Decay rate for classical weights
DEFAULT_REWARD_INTERVAL = 5 # How often reinforcement is applied
DEFAULT_ACTIVATION_THRESHOLD_PROMOTION = 0.7 # Threshold for memory promotion (based on normalized Hamming weight)
DEFAULT_HISTORY_LENGTH_MAP_PROMOTION = {"short_term": 5, "mid_term": 15} # Epochs needed in history for promotion
DEFAULT_MODULE_CATEGORY_WEIGHT = 0.15 # Base weight influence between modules/categories
HISTORY_MAXLEN = 100 # Max length of history deques (reduced for memory)
QUANTUM_ACTIVATION_SHOTS = 3 # REDUCED number of measurement shots per activation for performance!
QUANTUM_PARAM_LEARNING_RATE = 0.005 # Learning rate for internal quantum parameters (potentially smaller due to more parameters)

# Emotion (PAD) Model Parameters
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}
EMOTION_UPDATE_RATE = 0.03 # How quickly emotions change towards target
EMOTION_VOLATILITY = 0.02 # Amount of random noise in emotion updates
EMOTION_DECAY_TO_NEUTRAL = 0.05 # Tendency to return to neutral emotion

# Value Node Parameters
DEFAULT_VALUES = {"Innovation": 0.1, "Sicherheit": 0.1, "Effizienz": 0.1, "Ethik": 0.1, "Neugier": 0.1}
VALUE_UPDATE_RATE = 0.1 # How quickly values adapt
VALUE_INFLUENCE_FACTOR = 0.1 # How much values influence signal processing

# Reinforcement Learning Parameters
REINFORCEMENT_PLEASURE_THRESHOLD = 0.3 # Pleasure level above which reward occurs
REINFORCEMENT_CRITIC_THRESHOLD = 0.7 # Critic score above which reward occurs
REINFORCEMENT_FACTOR = 0.01 # Base factor for reinforcement weight changes

# Persistent Memory Parameters
PERSISTENT_MEMORY_TABLE = "core_memories"
PERSISTENT_REFLECTION_TABLE = "reflection_log"
MEMORY_RELEVANCE_THRESHOLD = 0.6 # Activation threshold for consolidating long-term memory
MEMORY_CONSOLIDATION_INTERVAL = 10 # How often memory consolidation runs

# Structural Plasticity Parameters
STRUCTURAL_PLASTICITY_INTERVAL = 6 # How often pruning/sprouting occurs
PRUNING_THRESHOLD = 0.015 # Weight threshold below which connections are pruned
SPROUTING_THRESHOLD = 0.75 # Activation threshold above which nodes might sprout new connections
SPROUTING_NEW_WEIGHT_MEAN = 0.025 # Mean for new connection weights
MAX_CONNECTIONS_PER_NODE = 50 # Maximum outgoing connections per node (reduced)
NODE_PRUNING_ENABLED = False # Whether to prune entire inactive nodes (use with caution)
NODE_INACTIVITY_THRESHOLD_EPOCHS = 30 # Epochs of inactivity before node pruning check
NODE_INACTIVITY_ACTIVATION = 0.1 # Activation threshold for inactivity (normalized Hamming weight)

# Meta-Cognition Parameters
REFLECTION_LOG_MAX_LEN = 150 # Max length of the reflection log
STAGNATION_DETECTION_WINDOW = 6 # Epoch window to detect stagnation
STAGNATION_THRESHOLD = 0.008 # Change threshold for detecting stagnation
OSCILLATION_DETECTION_WINDOW = 8 # Epoch window to detect oscillations
OSCILLATION_THRESHOLD_STD = 0.3 # Standard deviation threshold for oscillation detection

# --- Helper Functions ---
def random_neuron_type() -> str:
    """Assigns a random neuron type based on predefined probabilities."""
    r = random.random()
    if r < 0.7: return "excitatory"
    elif r < 0.95: return "inhibitory"
    else: return "interneuron"

def calculate_dynamic_learning_rate(base_lr: float, emotion_state: Dict[str, float], meta_cognitive_state: Dict[str, Any]) -> float:
    """Calculates a learning rate modulated by emotion and meta-cognition."""
    arousal = emotion_state.get('arousal', 0.0)
    pleasure = emotion_state.get('pleasure', 0.0)
    # Emotional modulation: Higher arousal/pleasure slightly increase LR
    emo_factor = 1.0 + (arousal * 0.3) + (pleasure * 0.2)
    # Meta-cognitive modulation: Strategy can boost or reduce LR
    meta_factor = meta_cognitive_state.get('lr_boost', 1.0)
    dynamic_lr = base_lr * emo_factor * meta_factor
    # Clip the dynamic LR to prevent extreme values
    return float(np.clip(dynamic_lr, 0.0005, 0.5))

def _default_status_callback(message: str):
    """Default callback function to print status messages."""
    print(f"[Status] {message}")

# --- HTML Report Generation ---
def create_html_report(final_summary: str, final_recommendation: str, interpretation_log: list, important_categories: list, structured_results: dict, plots_folder: str = PLOTS_FOLDER, output_html: str = "neuropersona_report.html") -> None:
    """Generates an HTML report summarizing the simulation results."""
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    # Find available plot files and order them
    try:
        all_files = [f for f in os.listdir(plots_folder) if f.endswith(".png")]
    except FileNotFoundError:
        plots = []
    else:
        # Define a preferred order for plots
        plot_order = ["plot_act_weights.png", "plot_dynamics.png", "plot_modules.png", "plot_emo_values.png", "plot_structure_stats.png", "plot_network_graph.png"]
        plots_in_order = [p for p in plot_order if p in all_files]
        other_plots = sorted([f for f in all_files if f not in plot_order])
        plots = plots_in_order + other_plots

    # Determine color based on the final recommendation
    recommendation_color = {
        "Empfehlung": "#28a745", "Empfehlung (moderat)": "#90ee90",
        "Abwarten": "#ffc107", "Abwarten (Instabil/Schwach)": "#ffe066",
        "Abraten": "#dc3545", "Abraten (moderat)": "#f08080"
    }.get(final_recommendation, "#6c757d") # Default color

    # Extract final states for the report
    final_emotion = structured_results.get('emotion_state', {})
    final_values = structured_results.get('value_node_activations', {})
    reflections = structured_results.get('reflection_summary', [])
    exec_time = structured_results.get('execution_time_seconds', None)
    stability = structured_results.get('stability_assessment', 'N/A')

    try:
        with open(output_html, "w", encoding="utf-8") as f:
            # HTML Head and CSS Styles
            f.write(f"""<!DOCTYPE html><html lang='de'>
<head>
    <meta charset='UTF-8'>
    <title>NeuroPersona Bericht ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; }}
        h1, h2, h3 {{ color: #343a40; border-bottom: 1px solid #dee2e6; padding-bottom: 5px; }}
        .report-container {{ max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.1); }}
        .prognosis {{ background: {recommendation_color}; color: white; padding: 15px; border-radius: 8px; font-size: 1.15em; margin-bottom: 20px; text-align: center; }}
        details {{ margin-top: 15px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; }}
        summary {{ font-weight: 700; cursor: pointer; color: #0056b3; }}
        summary:hover {{ color: #003d80; }}
        img {{ max-width: 95%; height: auto; margin-top: 10px; border: 1px solid #dee2e6; border-radius: 5px; display: block; margin-left: auto; margin-right: auto; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; background: #e9ecef; padding: 10px; border-radius: 4px; font-size: 0.9em; }}
        .footer {{ margin-top: 40px; text-align: center; font-size: .85em; color: #adb5bd; }}
        .epoch-entry {{ border-bottom: 1px dashed #eee; padding-bottom: 5px; margin-bottom: 5px; font-size: 0.9em; }}
        .epoch-entry:last-child {{ border-bottom: none; }}
        .category-list li {{ margin-bottom: 3px; }}
    </style>
</head>
<body><div class='report-container'>""")

            # Report Title (reflects multi-qubit version)
            f.write(f"<h1>NeuroPersona Analysebericht ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2 - EXPERIMENTELL)</h1>")
            # Overall Assessment Box
            f.write(f"<div class='prognosis'>📈 <b>Gesamteinschätzung: {final_recommendation}</b></div>")

            # Core Points & Network State Section
            f.write("<details open><summary>📋 Kernpunkte & Netzwerkzustand</summary>")
            dom_cat = structured_results.get('dominant_category', 'N/A')
            dom_act = structured_results.get('dominant_activation', 0.0)
            f_dom_cat = structured_results.get('frequent_dominant_category', 'N/A')
            f.write(f"<p><b>Finale Dominanz:</b> {dom_cat} (Aktivierung: {dom_act:.3f})<br>")
            f.write(f"<b>Häufigste Dominanz über Zeit:</b> {f_dom_cat}<br>")
            f.write(f"<b>Stabilitätsbewertung:</b> {stability}</p>")
            f.write(f"<h3>Wichtigste Kategorien (Final)</h3>")
            if important_categories:
                f.write("<ul class='category-list'>")
                for cat, imp in important_categories:
                    f.write(f"<li><b>{cat}</b>: {imp}</li>")
                f.write("</ul>")
            else:
                f.write("<p>Keine hervorstechenden Kategorien identifiziert.</p>")
            f.write("<br><i>Textuelle Zusammenfassung:</i><pre>" + final_summary + "</pre></details>")

            # Cognitive & Emotional State Section
            f.write("<details><summary>🧠 Kognitiver & Emotionaler Zustand (Final)</summary>")
            f.write("<h3>Emotionen (PAD):</h3><pre>" + json.dumps(final_emotion, indent=2) + "</pre>")
            f.write("<h3>Werte:</h3><pre>" + json.dumps(final_values, indent=2) + "</pre></details>")

            # Meta-Cognitive Reflections Section (if available)
            if reflections:
                f.write("<details><summary>🤔 Meta-Kognitive Reflexionen (Letzte 5)</summary>")
                for entry in reflections[:5]: # Show only the last 5 reflections
                    msg = entry.get('message','(Keine Nachricht)')
                    data_str = json.dumps(entry.get('data',{})) if entry.get('data') else ""
                    f.write(f"<p><b>Epoche {entry.get('epoch','?')}:</b> {msg} {f'<small><i>({data_str})</i></small>' if data_str else ''}</p>")
                f.write("</details>")

            # Analysis History Section (if available)
            if interpretation_log:
                f.write("<details><summary>📈 Analyseverlauf (Letzte 10 Epochen)</summary>")
                log_subset = interpretation_log[-10:] # Show only the last 10 epochs
                for entry in reversed(log_subset):
                    epoch = entry.get('epoch', '-')
                    dom = entry.get('dominant_category', '-')
                    act = entry.get('dominant_activation', 0.0)
                    emo_p = entry.get('emotion_state', {}).get('pleasure', 0.0)
                    val_i = entry.get('value_node_activations', {}).get('Innovation', 0.0)
                    # Ensure activation is a valid float for formatting
                    act_val = float(act) if isinstance(act, (float, int, np.number)) and not np.isnan(act) else 0.0
                    act_str = f"{act_val:.2f}"
                    f.write(f"<div class='epoch-entry'><b>E{epoch}:</b> Dominanz: {dom} ({act_str}), Pleasure: {emo_p:.2f}, Innovation: {val_i:.2f}</div>")
                if len(interpretation_log) > 10:
                    f.write("<p><i>... (ältere Epochen werden nicht angezeigt)</i></p>")
                f.write("</details>")

            # Visualizations Section (if available)
            if plots:
                f.write("<details open><summary>🖼️ Visualisierungen</summary>")
                for plot_rel_path in plots:
                    plot_filename = os.path.basename(plot_rel_path)
                    # Generate a title from the filename
                    plot_title = plot_filename.replace('.png','').replace('plot_','').replace('_',' ').title()
                    f.write(f"<p style='text-align:center; font-weight:bold; margin-top:15px;'>{plot_title}</p>")
                    # Embed the image using the relative path within the PLOTS_FOLDER
                    f.write(f"<img src='{plots_folder}/{plot_filename}' alt='{plot_filename}'><br>")
                f.write("</details>")
            else:
                f.write("<details><summary>🖼️ Visualisierungen</summary><p>Keine Plots wurden generiert oder gefunden.</p></details>")

            # Footer
            f.write("<div class='footer'>")
            if exec_time:
                f.write(f"Analyse-Dauer: {exec_time:.2f} Sekunden. ")
            f.write(f"Erstellt mit NeuroPersona ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2) am {time.strftime('%d.%m.%Y %H:%M:%S')}")
            f.write("</div>")

            # Close HTML tags
            f.write("</div></body></html>")

        print(f"✅ HTML-Report ({NUM_QUBITS_PER_NODE}-Qubit) erfolgreich erstellt: {output_html}")
    except IOError as e:
        print(f"FEHLER beim Schreiben des HTML-Reports '{output_html}': {e}")
    except Exception as e:
        print(f"Unbekannter FEHLER bei der HTML-Report-Erstellung: {e}")
        traceback.print_exc()

# --- Network Helper Functions ---
def convert_text_answer_to_numeric(answer_text: str) -> float:
    """Converts common textual answers to a numeric value between 0.0 and 1.0."""
    if not isinstance(answer_text, str): return 0.5 # Default for non-string input
    text = answer_text.strip().lower()
    # Map specific phrases first
    mapping = {
        "sehr hoch": 0.95, "hoch": 0.8, "eher hoch": 0.7, "mittel": 0.5, "eher niedrig": 0.35,
        "niedrig": 0.3, "gering": 0.2, "sehr gering": 0.05,
        "ja": 0.9, "eher ja": 0.7, "nein": 0.1, "eher nein": 0.3,
        "positiv": 0.85, "eher positiv": 0.65, "negativ": 0.15, "eher negativ": 0.35,
        "neutral": 0.5, "unsicher": 0.4, "sicher": 0.9, "unbekannt": 0.5,
        "stimme voll zu": 1.0, "stimme zu": 0.8, "stimme eher zu": 0.6,
        "lehne ab": 0.2, "lehne voll ab": 0.0, "lehne eher ab": 0.4,
        "trifft voll zu": 1.0, "trifft zu": 0.8, "trifft eher zu": 0.6,
        "trifft nicht zu": 0.2, "trifft gar nicht zu": 0.0, "trifft eher nicht zu": 0.4
    }
    if text in mapping:
        return mapping[text]

    # General keyword matching as fallback
    if "hoch" in text or "stark" in text or "viel" in text: return 0.8
    if "gering" in text or "niedrig" in text or "wenig" in text: return 0.2
    if "mittel" in text or "moderat" in text: return 0.5
    if "positiv" in text or "chance" in text or "gut" in text: return 0.85
    if "negativ" in text or "risiko" in text or "schlecht" in text: return 0.15
    if "ja" in text or "zustimm" in text or "trifft zu" in text: return 0.9
    if "nein" in text or "ablehn" in text or "trifft nicht zu" in text: return 0.1

    # Default if no match found
    return 0.5

def preprocess_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the input DataFrame: checks columns, cleans text, normalizes answers."""
    print("Starte Datenvorverarbeitung...")
    if not isinstance(input_data, pd.DataFrame):
        print("FEHLER: Eingabedaten sind kein Pandas DataFrame.")
        return pd.DataFrame(columns=['Frage', 'Antwort', 'Kategorie', 'normalized_answer'])

    required_cols = ['Frage', 'Antwort', 'Kategorie']
    if not all(col in input_data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in input_data.columns]
        print(f"FEHLER: Erforderliche Spalten fehlen im DataFrame: {missing}.")
        return pd.DataFrame(columns=required_cols + ['normalized_answer'])

    data = input_data.copy()
    # Clean 'Frage' and 'Kategorie' columns
    data['Frage'] = data['Frage'].astype(str).str.strip()
    # Ensure 'Kategorie' is string, handle missing/empty values
    data['Kategorie'] = data['Kategorie'].astype(str).str.strip().replace('', 'Unkategorisiert').fillna('Unkategorisiert')

    # Apply normalization function to 'Antwort'
    data['normalized_answer'] = data['Antwort'].apply(convert_text_answer_to_numeric)

    print(f"Datenvorverarbeitung abgeschlossen. {len(data)} Zeilen verarbeitet. "
          f"{data['Kategorie'].nunique()} einzigartige Kategorien gefunden.")
    return data

def social_influence(nodes_list: List['Node'], social_network: Dict[str, float], influence_factor: float = 0.05):
    """Applies social influence by slightly reinforcing connections targeting influenced nodes."""
    applied_count = 0
    nodes_with_labels = [node for node in nodes_list if hasattr(node, 'label')]
    if not nodes_with_labels: return # No nodes to influence

    for target_node in nodes_with_labels:
        if target_node.label not in social_network: continue # Node not in social context

        social_impact = social_network.get(target_node.label, 0.0) * influence_factor
        if social_impact <= 0.001: continue # Negligible impact

        # Find incoming connections to this target_node
        for source_node in nodes_with_labels:
            if hasattr(source_node, 'connections'):
                source_activation = float(getattr(source_node, 'activation', 0.0)) # Use current activation
                for conn in source_node.connections:
                    # Check if the connection points to the target node
                    if hasattr(conn.target_node, 'label') and conn.target_node.label == target_node.label:
                        # Reinforce the connection weight based on social impact and source activation
                        reinforcement = social_impact * source_activation
                        conn.weight = float(np.clip(float(conn.weight) + reinforcement, 0.0, 1.0)) # Ensure float and clip
                        applied_count += 1
    # Optional: print(f"[Social] Applied influence to {applied_count} connections.")

def decay_weights(nodes_list: List['Node'], decay_rate: float = 0.002, forgetting_curve: float = 0.98):
    """Applies a decay factor to all connection weights in the network."""
    # Calculate the combined decay factor (must be between 0 and 1)
    decay_factor = float(np.clip((1.0 - decay_rate) * forgetting_curve, 0.0, 1.0))
    if decay_factor > 0.999: return # Avoid unnecessary computation if decay is negligible

    decayed_count = 0
    for node in nodes_list:
        if hasattr(node, 'connections'):
            for conn in node.connections:
                if hasattr(conn, 'weight'):
                    conn.weight = float(conn.weight) * decay_factor # Apply decay
                    decayed_count += 1
    # Optional: print(f"[Decay] Decayed {decayed_count} connection weights.")

# --- Hebbian Learning (MODIFIED for Multi-Qubit Feedback) ---
def hebbian_learning_quantum_node(node_a: 'Node', connection: 'Connection',
                                  learning_rate_classical: float = 0.1,
                                  learning_rate_quantum: float = QUANTUM_PARAM_LEARNING_RATE,
                                  weight_limit: float = 1.0, reg_factor: float = 0.001):
    """
    Modified Hebbian learning rule for quantum nodes.
    - Updates the classical connection weight.
    - Provides feedback to the internal quantum parameters of the *presynaptic* node (node_a).
    - Includes Long-Term Depression (LTD) and weight regularization.
    """
    node_b = connection.target_node
    # Ensure both nodes and their activations are valid
    if not node_b or not hasattr(node_a, 'activation') or not hasattr(node_b, 'activation'):
        return

    # Get activations (floats representing normalized Hamming weight)
    act_a = float(node_a.activation)
    act_b = float(node_b.activation)

    # Hebbian Condition (Potentiation): Pre- and Post-synaptic nodes are co-active
    # Thresholds might need adjustment based on typical activation ranges
    if act_a > 0.55 and act_b > 0.55:
        # 1. Classical Weight Update (strengthen connection)
        delta_weight_classical = learning_rate_classical * act_a * act_b
        connection.weight = float(np.clip(float(connection.weight) + delta_weight_classical, 0.0, weight_limit))

        # 2. Quantum Parameter Feedback (Potentiation for presynaptic node)
        # If node_a has a quantum system, adjust its parameters to make it slightly
        # more likely to activate in the future under similar input.
        if hasattr(node_a, 'q_system') and isinstance(node_a.q_system, QuantumNodeSystem):
            q_system_a = node_a.q_system
            # Create a delta array for parameters, initialized to zero
            param_delta = np.zeros_like(q_system_a.get_params())
            if len(param_delta) > 0:
                # Modify the RY parameters (indices 0, 2, 4, ...)
                # Increase theta slightly -> increases probability of rotating towards |1>
                ry_indices = range(0, q_system_a.num_params, 2)
                param_delta[list(ry_indices)] = learning_rate_quantum * act_a * act_b * 0.5 # Apply scaled delta
            q_system_a.update_internal_params(param_delta)

    # Long-Term Depression (LTD): Pre-synaptic fires, Post-synaptic does not strongly
    elif act_a > 0.55 and act_b < 0.3:
        # 1. Classical Weight Update (weaken connection)
        delta_weight_ltd = -0.1 * learning_rate_classical * act_a # Small reduction
        connection.weight = float(np.clip(float(connection.weight) + delta_weight_ltd, 0.0, weight_limit))

        # 2. Quantum Parameter Feedback (Depression for presynaptic node)
        # Make node_a slightly less sensitive / less likely to activate
        if hasattr(node_a, 'q_system') and isinstance(node_a.q_system, QuantumNodeSystem):
            q_system_a = node_a.q_system
            param_delta = np.zeros_like(q_system_a.get_params())
            if len(param_delta) > 0:
                # Decrease the RY parameters (theta) slightly
                ry_indices = range(0, q_system_a.num_params, 2)
                param_delta[list(ry_indices)] = -0.1 * learning_rate_quantum * act_a * 0.5 # Apply scaled negative delta
            q_system_a.update_internal_params(param_delta)

    # 3. Regularization (applied always to prevent weights from growing indefinitely)
    # Simple L1 or L2 regularization can be used. Here, a simple decay based on weight magnitude.
    connection.weight = float(np.clip(float(connection.weight) - reg_factor * float(connection.weight), 0.0, weight_limit))


# --- Structural Plasticity Functions ---
def prune_connections(nodes_list: List['Node'], threshold: float = PRUNING_THRESHOLD) -> int:
    """Removes connections with weights below the specified threshold."""
    pruned_count = 0
    for node in nodes_list:
        if hasattr(node, 'connections'):
            original_count = len(node.connections)
            # Keep only connections where target exists and weight is above threshold
            node.connections = [
                conn for conn in node.connections
                if conn.target_node and hasattr(conn, 'weight') and float(conn.weight) >= threshold
            ]
            pruned_count += original_count - len(node.connections)
    # if pruned_count > 0: print(f"[Plasticity] Pruned {pruned_count} weak connections.")
    return pruned_count

def sprout_connections(nodes_list: List['Node'], activation_history_local: Dict[str, deque],
                         threshold: float = SPROUTING_THRESHOLD, max_conns: int = MAX_CONNECTIONS_PER_NODE,
                         new_weight_mean: float = SPROUTING_NEW_WEIGHT_MEAN) -> int:
    """Creates new connections between highly co-active nodes that are not yet connected."""
    sprouted_count = 0
    # Identify valid nodes with labels and activation history
    valid_nodes = [node for node in nodes_list if hasattr(node, 'label')]
    node_map = {node.label: node for node in valid_nodes}

    # Get the latest activation for nodes with history
    last_activations = {}
    for label, history in activation_history_local.items():
        if history and label in node_map:
            # Safely get the last activation value, ensuring it's a float
            last_act_raw = history[-1]
            if isinstance(last_act_raw, (float, np.number)) and not np.isnan(last_act_raw):
                last_activations[label] = float(last_act_raw)

    # Find nodes that were recently active above the sprouting threshold
    active_nodes_labels = [label for label, act in last_activations.items() if act > threshold]

    if len(active_nodes_labels) < 2: return 0 # Need at least two active nodes to form a connection

    random.shuffle(active_nodes_labels)
    # Limit the number of potential new connections per epoch to avoid excessive growth
    max_sprouts_per_epoch = max(1, int(len(active_nodes_labels) * 0.1)) # Sprout up to 10% of active nodes
    attempted_sprouts = 0

    for i, label1 in enumerate(active_nodes_labels):
        if sprouted_count >= max_sprouts_per_epoch: break # Limit sprouts per epoch

        node1 = node_map.get(label1)
        # Check if node1 exists, has connections attribute, and hasn't reached max connections
        if not node1 or not hasattr(node1, 'connections') or len(node1.connections) >= max_conns:
            continue

        # Consider other active nodes as potential partners
        potential_partners_labels = active_nodes_labels[i+1:] + active_nodes_labels[:i] # Cycle through partners
        random.shuffle(potential_partners_labels)

        for label2 in potential_partners_labels:
             # Limit search attempts to prevent potential infinite loops in dense graphs
            if attempted_sprouts > len(active_nodes_labels) * 3: break
            attempted_sprouts += 1

            node2 = node_map.get(label2)
            # Skip if node2 is invalid, same as node1, or has max connections
            if not node2 or (node1 == node2): continue
            if not hasattr(node2, 'connections') or len(node2.connections) >= max_conns: continue

            # Check if a connection already exists in either direction
            conn_exists_12 = any(hasattr(conn.target_node, 'label') and conn.target_node.label == label2
                                for conn in node1.connections if conn.target_node)
            conn_exists_21 = any(hasattr(conn.target_node, 'label') and conn.target_node.label == label1
                                for conn in node2.connections if conn.target_node)

            # If no connection exists, create a new one
            if not conn_exists_12 and not conn_exists_21:
                # Generate a small initial weight, ensuring it's positive
                new_weight = max(0.001, np.random.normal(new_weight_mean, new_weight_mean / 2.0))
                node1.add_connection(node2, weight=float(new_weight))
                sprouted_count += 1
                # Move to the next source node once a connection is sprouted for node1
                break
        # Outer loop break check
        if attempted_sprouts > len(active_nodes_labels) * 3: break

    # if sprouted_count > 0: print(f"[Plasticity] Sprouted {sprouted_count} new connections.")
    return sprouted_count

def prune_inactive_nodes(nodes_list: List['Node'], activation_history_local: Dict[str, deque],
                           current_epoch: int, threshold_epochs: int = NODE_INACTIVITY_THRESHOLD_EPOCHS,
                           activation_threshold: float = NODE_INACTIVITY_ACTIVATION,
                           enabled: bool = NODE_PRUNING_ENABLED) -> Tuple[List['Node'], int]:
    """
    Removes nodes that have been inactive for a specified number of epochs.
    Protected node types (modules, values) are not pruned. Also removes incoming connections to pruned nodes.
    """
    if not enabled:
        return nodes_list, 0 # Return original list if pruning is disabled

    nodes_to_remove_labels = set()
    # Define classes of nodes that should not be pruned automatically
    protected_classes = (CortexCreativus, SimulatrixNeuralis, CortexCriticus,
                         LimbusAffektus, MetaCognitio, CortexSocialis, ValueNode)

    # Filter for nodes that are candidates for pruning (not protected, have label)
    candidate_nodes = [
        node for node in nodes_list
        if isinstance(node, Node) and hasattr(node, 'label')
        and not isinstance(node, protected_classes)
        and not node.label.startswith("Q_") # Don't prune input nodes
    ]

    # 1. Identify inactive nodes among candidates
    for node in candidate_nodes:
        history = activation_history_local.get(node.label)
        # Check if sufficient history exists
        if not history or len(history) < threshold_epochs:
            continue

        # Check if the node was recently inactive
        # Get the last 'threshold_epochs' activations, ensuring they are valid floats
        recent_history_raw = list(history)[-threshold_epochs:]
        recent_history_numeric = [act for act in recent_history_raw
                                  if isinstance(act, (float, np.number)) and not np.isnan(act)]

        # If there's valid recent history and all activations are below threshold, mark for removal
        if recent_history_numeric and all(act < activation_threshold for act in recent_history_numeric):
            nodes_to_remove_labels.add(node.label)

    # If no nodes are marked for removal, return the original list
    if not nodes_to_remove_labels:
        return nodes_list, 0

    # 2. Create the new list of nodes, excluding those marked for removal
    original_node_count = len(nodes_list)
    new_nodes_list = [node for node in nodes_list if not (hasattr(node, 'label') and node.label in nodes_to_remove_labels)]
    nodes_removed_count = original_node_count - len(new_nodes_list)

    # 3. Remove incoming connections pointing to the deleted nodes from the *remaining* nodes
    connections_removed_count = 0
    for node in new_nodes_list: # Iterate over the nodes that *remain*
        if hasattr(node, 'connections'):
            original_conn_count = len(node.connections)
            # Keep only connections whose target is *not* in the removal set
            node.connections = [
                conn for conn in node.connections
                if conn.target_node and hasattr(conn.target_node, 'label')
                and conn.target_node.label not in nodes_to_remove_labels
            ]
            connections_removed_count += original_conn_count - len(node.connections)

    # Log the pruning action if nodes were removed
    if nodes_removed_count > 0:
         print(f"[Plasticity E{current_epoch}] Pruned {nodes_removed_count} inactive nodes "
               f"({', '.join(list(nodes_to_remove_labels))}) and {connections_removed_count} related incoming connections.")

    # Optional: Add a check here for remaining 'zombie' connections (connections with target=None)
    # for node in new_nodes_list:
    #    if hasattr(node, 'connections'):
    #        zombies = [i for i, conn in enumerate(node.connections) if conn.target_node is None]
    #        if zombies: print(f"WARNING: Zombie connections found in node {node.label} at indices {zombies} AFTER pruning.")


    return new_nodes_list, nodes_removed_count


# --- Core Network Structure Classes ---
class Connection:
    """Represents a directed connection between two nodes with a weight."""
    def __init__(self, target_node: 'Node', weight: Optional[float] = None):
        self.target_node: Optional['Node'] = target_node
        # Initialize weight randomly if not provided, ensuring it's a float
        raw_weight = weight if weight is not None else random.uniform(0.05, 0.3)
        self.weight: float = float(np.clip(raw_weight, 0.0, 1.0)) # Store weight as float, clipped [0, 1]

    def __repr__(self) -> str:
        target_label = getattr(self.target_node, 'label', 'None')
        return f"<Connection to:{target_label} W:{self.weight:.3f}>"

class Node:
    """Base class for a node in the NeuroPersona network."""
    def __init__(self, label: str, neuron_type: str = "excitatory", num_qubits: int = NUM_QUBITS_PER_NODE):
        """
        Initializes a Node.

        Args:
            label (str): A unique identifier for the node.
            neuron_type (str): Type of neuron ('excitatory', 'inhibitory', 'interneuron').
            num_qubits (int): The number of qubits for the node's quantum system.
                              Set to 0 or less to disable quantum system for this node type.
        """
        self.label: str = label
        self.neuron_type: str = neuron_type
        self.connections: List[Connection] = []
        # Activation is the probability (normalized Hamming weight) from quantum measurement
        self.activation: float = 0.0
        # activation_sum stores the classical weighted input sum for the *next* quantum activation
        self.activation_sum: float = 0.0
        self.activation_history: deque = deque(maxlen=HISTORY_MAXLEN)

        # Each node gets its own quantum system instance if num_qubits > 0
        self.q_system: Optional[QuantumNodeSystem] = None
        if num_qubits > 0:
            try:
                self.q_system = QuantumNodeSystem(num_qubits=num_qubits)
            except ValueError as e:
                print(f"Error initializing QuantumNodeSystem for Node {self.label}: {e}")
                self.q_system = None # Fallback if initialization fails
        #else: node has no quantum system (e.g., ValueNode, InputNode)

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None):
        """Adds a connection from this node to a target node."""
        # Prevent self-connections or connections to None
        if target_node is self or target_node is None:
            # print(f"DEBUG: Skipped invalid connection attempt from {self.label} to {target_node}")
            return
        # Avoid duplicate connections to the same target
        if not any(conn.target_node == target_node for conn in self.connections):
            self.connections.append(Connection(target_node, weight))

    def __repr__(self) -> str:
        """Provides a string representation of the node."""
        act_str = f"{self.activation:.3f}"
        q_info = ""
        if self.q_system:
            q_info = f" Qubits:{self.q_system.num_qubits}"
            # Optionally show first param: f" P0:{self.q_system.params[0]:.2f}"
        conn_count = len(self.connections)
        return f"<{type(self).__name__} {self.label} Act:{act_str}{q_info} Conns:{conn_count}>"

class MemoryNode(Node):
    """Represents a node holding a category or concept, capable of memory promotion."""
    def __init__(self, label: str, memory_type: str = "short_term", neuron_type: str = "excitatory"):
        # Initialize with the standard number of qubits for quantum nodes
        super().__init__(label, neuron_type=neuron_type, num_qubits=NUM_QUBITS_PER_NODE)
        self.memory_type: str = memory_type # "short_term", "mid_term", "long_term"
        self.retention_times: Dict[str, int] = {"short_term": 5, "mid_term": 20, "long_term": 100}
        self.retention_time: int = self.retention_times.get(memory_type, 20)
        self.time_in_memory: int = 0 # Epochs spent in the current memory type
        self.history_length_maps = DEFAULT_HISTORY_LENGTH_MAP_PROMOTION.copy()

    def promote(self, activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD_PROMOTION,
                history_length_map: Optional[Dict[str, int]] = None):
        """Promotes the memory node to the next stage (short->mid, mid->long) if consistently active."""
        if history_length_map is None:
            history_length_map = self.history_length_maps

        required_length = history_length_map.get(self.memory_type)

        # Cannot promote if already long-term, or not enough history exists
        if self.memory_type == "long_term" or required_length is None or len(self.activation_history) < required_length:
            if self.memory_type != "long_term": self.time_in_memory += 1
            return

        # Check average activation over the required history window
        recent_history = list(self.activation_history)[-required_length:]
        # Filter out non-numeric entries if any (should ideally not happen)
        valid_recent_activations = [act for act in recent_history if isinstance(act, (float, np.number)) and not np.isnan(act)]

        if not valid_recent_activations:
            avg_recent_activation = 0.0 # Handle case where history is empty or invalid
        else:
            avg_recent_activation = np.mean(valid_recent_activations)

        # Promote if average activation exceeds the threshold
        if avg_recent_activation > activation_threshold:
            original_type = self.memory_type
            if self.memory_type == "short_term":
                self.memory_type = "mid_term"
            elif self.memory_type == "mid_term":
                self.memory_type = "long_term"

            # If promotion occurred, reset timer and update retention time
            if original_type != self.memory_type:
                self.time_in_memory = 0
                self.retention_time = self.retention_times.get(self.memory_type, self.retention_time)
                # print(f"[Memory] Node '{self.label}' promoted from {original_type} to {self.memory_type}.")
        else:
            # If not promoted, increment time spent in current memory type
            if self.memory_type != "long_term":
                self.time_in_memory += 1

    def __repr__(self) -> str:
        """Custom representation for MemoryNode."""
        base_repr = super().__repr__()
        # Remove closing '>' add memory type info, and re-add '>'
        return base_repr[:-1] + f" MemType:{self.memory_type}>"

class ValueNode(Node):
    """Represents a core value. Uses classical activation, NO quantum system."""
    def __init__(self, label: str, initial_value: float = 0.5):
        # Initialize as a Node but explicitly disable the quantum system by setting num_qubits=0
        super().__init__(label, neuron_type="excitatory", num_qubits=0)
        # Activation represents the strength of the value (classical)
        self.activation = float(np.clip(initial_value, 0.0, 1.0))

    def update_value(self, adjustment: float):
        """Updates the value activation based on network dynamics."""
        current_value = float(self.activation)
        self.activation = float(np.clip(current_value + float(adjustment), 0.0, 1.0))

    # Override __repr__ as q_system is None
    def __repr__(self) -> str:
        """Custom representation for ValueNode."""
        act_str = f"{self.activation:.3f}"
        conn_count = len(self.connections)
        return f"<{type(self).__name__} {self.label} Value:{act_str} Conns:{conn_count}>"


# --- Persistent Memory Management ---
class PersistentMemoryManager:
    """Handles storage and retrieval of long-term memories and reflections in an SQLite database."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PersistentMemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = PERSISTENT_MEMORY_DB):
        if self._initialized: return
        self.db_path: str = db_path
        self.table_name: str = PERSISTENT_MEMORY_TABLE
        self.reflection_table_name: str = PERSISTENT_REFLECTION_TABLE
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._initialize_db()
        self._initialized = True
        print(f"PersistentMemoryManager initialized with database: {self.db_path}")

    def _get_connection(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """Establishes or returns the existing database connection."""
        if self.conn is None or self.cursor is None:
            try:
                # Connect with timeout and disable thread checking for potential background use
                self.conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
                self.conn.execute("PRAGMA journal_mode=WAL;") # Improve concurrency
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                print(f"FATAL DATABASE ERROR: Could not connect to {self.db_path}: {e}")
                self.conn , self.cursor = None, None
                raise # Re-raise the exception as this is critical
        return self.conn, self.cursor

    def _close_connection(self):
        """Commits changes and closes the database connection."""
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
            except sqlite3.Error as e:
                print(f"ERROR closing database connection: {e}")
            finally:
                self.conn, self.cursor = None, None

    def _execute_query(self, query: str, params: tuple = (), fetch_one: bool = False,
                       fetch_all: bool = False, commit: bool = False) -> Any:
        """Executes a given SQL query with parameters and handles results/commits."""
        result = None
        conn = None # Avoid using self.conn directly in case of recursion/re-entry issues
        cursor = None
        try:
            conn, cursor = self._get_connection()
            if cursor:
                 cursor.execute(query, params)
                 if commit: conn.commit() # Commit immediately if requested
                 if fetch_one: result = cursor.fetchone()
                 elif fetch_all: result = cursor.fetchall()
                 # Note: Connection is not closed here; managed by higher-level methods or `close()`
        except sqlite3.Error as e:
            print(f"DATABASE QUERY ERROR: {e}\nQuery: {query}\nParams: {params}")
            # Don't close here, let the calling function handle potential retries or closing
            # self._close_connection() # Avoid closing prematurely
        except Exception as e:
            print(f"UNEXPECTED ERROR during DB query execution: {e}")
            # self._close_connection() # Avoid closing prematurely
        return result

    def _initialize_db(self):
        """Creates the necessary tables if they don't exist."""
        try:
            # Core memories table
            mem_table_sql = f'''CREATE TABLE IF NOT EXISTS {self.table_name} (
                                id INTEGER PRIMARY KEY,
                                memory_key TEXT UNIQUE NOT NULL,
                                memory_content TEXT NOT NULL,
                                relevance REAL DEFAULT 0.5,
                                last_accessed REAL DEFAULT 0,
                                created_at REAL DEFAULT {time.time()}
                            )'''
            self._execute_query(mem_table_sql, commit=True)
            # Index for faster lookups by key
            mem_index_sql = f'CREATE INDEX IF NOT EXISTS idx_memory_key ON {self.table_name}(memory_key)'
            self._execute_query(mem_index_sql, commit=True)

            # Reflection log table
            ref_table_sql = f'''CREATE TABLE IF NOT EXISTS {self.reflection_table_name} (
                                log_id INTEGER PRIMARY KEY,
                                epoch INTEGER,
                                timestamp REAL,
                                message TEXT,
                                log_data TEXT
                            )'''
            self._execute_query(ref_table_sql, commit=True)
            print("Database tables checked/initialized successfully.")
        except Exception as e:
            print(f"CRITICAL DATABASE ERROR during initialization: {e}")
            # Initialization is critical, re-raise or handle appropriately
            raise
        finally:
            self._close_connection() # Close after initialization attempt

    def store_memory(self, key: str, content: dict, relevance: float):
        """Stores or updates a memory item in the database."""
        try:
            content_json = json.dumps(content) # Serialize content to JSON string
            timestamp = time.time()
            # Use INSERT OR REPLACE (or INSERT ON CONFLICT DO UPDATE) to handle existing keys
            query = f'''INSERT INTO {self.table_name}
                        (memory_key, memory_content, relevance, last_accessed, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(memory_key) DO UPDATE SET
                            memory_content=excluded.memory_content,
                            relevance=excluded.relevance,
                            last_accessed=excluded.last_accessed'''
            self._execute_query(query, (key, content_json, float(relevance), timestamp, timestamp), commit=True)
        except (TypeError, sqlite3.Error, json.JSONDecodeError) as e:
            print(f"ERROR storing memory ('{key}'): {e}")
        finally:
            self._close_connection() # Ensure connection is closed after operation

    def retrieve_memory(self, key: str) -> Optional[dict]:
        """Retrieves a specific memory item by its key and updates its last accessed time."""
        memory_content = None
        try:
            query = f'SELECT memory_content FROM {self.table_name} WHERE memory_key = ?'
            result = self._execute_query(query, (key,), fetch_one=True)
            if result and result[0]:
                try:
                    memory_content = json.loads(result[0]) # Deserialize JSON string
                    # Update last accessed time upon successful retrieval
                    update_query = f'UPDATE {self.table_name} SET last_accessed = ? WHERE memory_key = ?'
                    self._execute_query(update_query, (time.time(), key), commit=True)
                except json.JSONDecodeError:
                    print(f"ERROR decoding JSON for memory key '{key}'")
                    memory_content = None # Return None if JSON is corrupted
        except sqlite3.Error as e:
            print(f"ERROR retrieving memory ('{key}'): {e}")
        finally:
            self._close_connection()
        return memory_content

    def retrieve_relevant_memories(self, min_relevance: float = MEMORY_RELEVANCE_THRESHOLD, limit: int = 5) -> List[Dict]:
        """Retrieves the most relevant memories based on relevance and recency."""
        memories = []
        try:
            query = f'''SELECT memory_key, memory_content FROM {self.table_name}
                        WHERE relevance >= ?
                        ORDER BY relevance DESC, last_accessed DESC
                        LIMIT ?'''
            results = self._execute_query(query, (float(min_relevance), limit), fetch_all=True)
            if results:
                 for row in results:
                     try:
                         memories.append({"key": row[0], "content": json.loads(row[1])})
                     except json.JSONDecodeError:
                         print(f"Skipping corrupted JSON for memory key '{row[0]}' during relevant retrieval.")
        except sqlite3.Error as e:
            print(f"ERROR retrieving relevant memories: {e}")
        finally:
            self._close_connection()
        return memories

    def store_reflection(self, entry: Dict):
        """Stores a reflection log entry into the database."""
        try:
            log_data_json = json.dumps(entry.get('data', {})) # Serialize optional data field
            query = f'''INSERT INTO {self.reflection_table_name}
                        (epoch, timestamp, message, log_data) VALUES (?, ?, ?, ?)'''
            self._execute_query(query, (
                entry.get('epoch'),
                entry.get('timestamp', time.time()),
                entry.get('message'),
                log_data_json
            ), commit=True)
        except (TypeError, sqlite3.Error, json.JSONDecodeError) as e:
            print(f"ERROR storing reflection log: {e}")
        finally:
            self._close_connection()

    def retrieve_reflections(self, limit: int = 20) -> List[Dict]:
        """Retrieves the most recent reflection log entries."""
        logs = []
        try:
            query = f'''SELECT epoch, timestamp, message, log_data FROM {self.reflection_table_name}
                        ORDER BY timestamp DESC LIMIT ?'''
            results = self._execute_query(query, (limit,), fetch_all=True)
            if results:
                for row in results:
                    try:
                        # Handle potentially null or invalid JSON in log_data
                        data = json.loads(row[3] or '{}')
                    except json.JSONDecodeError:
                        data = {"error": "Invalid JSON in log data"}
                    logs.append({
                        "epoch": row[0],
                        "timestamp": row[1],
                        "message": row[2],
                        "data": data
                    })
        except sqlite3.Error as e:
            print(f"ERROR retrieving reflection logs: {e}")
        finally:
            self._close_connection()
        return logs

    def close(self):
        """Closes the database connection if it's open."""
        print("Closing persistent memory database connection...")
        self._close_connection()

# --- Specialized Cognitive Module Classes ---
# These modules inherit the multi-qubit Node structure but primarily use
# classical logic for their specialized functions, operating on the quantum activations.

class LimbusAffektus(Node):
    """Models the emotional state (PAD) based on network activity."""
    def __init__(self, label: str = "Limbus Affektus", neuron_type: str = "interneuron"):
        # Uses quantum system, potentially with more parameters for complex behavior
        super().__init__(label, neuron_type=neuron_type, num_qubits=NUM_QUBITS_PER_NODE) # Uses multi-qubit system
        self.emotion_state = INITIAL_EMOTION_STATE.copy()

    def update_emotion_state(self, all_nodes: List[Node], module_outputs: Dict[str, deque]) -> Dict[str, float]:
        """Updates the PAD emotional state based on network signals."""
        global CURRENT_EMOTION_STATE, activation_history # Access global state and history

        pleasure_signal = 0.0
        arousal_signal = 0.0
        dominance_signal = 0.0

        # Gather current activations (floats from quantum measurements) from relevant nodes
        relevant_nodes = [
            n for n in all_nodes
            if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number))
            and n.activation > 0.1 # Consider nodes with non-negligible activation
            and not np.isnan(n.activation) and hasattr(n, 'label')
        ]
        activations = [float(n.activation) for n in relevant_nodes]

        # Calculate overall network activity metrics
        avg_act = np.mean(activations) if activations else 0.0
        std_act = np.std(activations) if len(activations) > 1 else 0.0

        # --- Pleasure Calculation ---
        # Influenced by semantic content of active nodes and critic module output
        pos_triggers = ["chance", "wachstum", "positiv", "erfolg", "gut", "hoch", "ja", "innovation", "lösung", "empfehlung"]
        neg_triggers = ["risiko", "problem", "negativ", "fehler", "schlecht", "gering", "nein", "kritik", "sicherheit", "bedrohung", "abraten"]
        for node in relevant_nodes:
            if not isinstance(node.label, str): continue
            label_lower = node.label.lower()
            is_pos = any(trigger in label_lower for trigger in pos_triggers)
            is_neg = any(trigger in label_lower for trigger in neg_triggers)
            # Adjust pleasure signal based on positive/negative triggers and activation
            if is_pos and not is_neg: pleasure_signal += node.activation * 0.7
            elif is_neg and not is_pos: pleasure_signal -= node.activation * 0.9

        # Incorporate critic evaluation (more negative if critic scores are low)
        critic_evals_deque = module_outputs.get("Cortex Criticus")
        # Check if the deque exists and the last element is a list of evaluations
        if critic_evals_deque and isinstance(critic_evals_deque[-1], list):
            # Extract valid scores from the last evaluation batch
            scores = [e.get('score', 0.5) for e in critic_evals_deque[-1]
                      if isinstance(e, dict) and isinstance(e.get('score'), (float, np.number))]
            if scores:
                avg_critic_score = np.mean(scores)
                # Low scores decrease pleasure, high scores slightly increase it
                pleasure_signal += (avg_critic_score - 0.5) * 1.5 # Scaled influence from critic

        # --- Arousal Calculation ---
        # Influenced by average activation, standard deviation, and change in activation
        last_avg_activation = 0.0
        # Estimate previous average activation from global history
        if activation_history:
            all_last_acts = [
                float(h[-1]) for h in activation_history.values()
                if h and isinstance(h[-1], (float, np.number)) and not np.isnan(h[-1])
            ]
            if all_last_acts: last_avg_activation = np.mean(all_last_acts)
        # Change in average activation from the previous step
        activation_change = abs(avg_act - last_avg_activation)
        # Combine factors: base activation, variability, and rate of change
        arousal_signal = float(np.clip(avg_act * 0.4 + std_act * 0.3 + activation_change * 6.0, 0, 1))

        # --- Dominance Calculation ---
        # Influenced by MetaCognitio activity and network coherence (inverse of std dev)
        meta_cog_node = next((n for n in all_nodes if isinstance(n, MetaCognitio)), None)
        # MetaCognitio node also has quantum activation
        meta_cog_act = float(meta_cog_node.activation) if meta_cog_node and hasattr(meta_cog_node, 'activation') else 0.0
        # Coherence proxy: lower standard deviation implies higher perceived control/coherence
        control_proxy = 1.0 - std_act # Higher value for lower std dev
        # Combine meta-cognitive activity and coherence
        dominance_signal = float(np.clip(meta_cog_act * 0.5 + control_proxy * 0.5, 0, 1))

        # --- Update PAD State ---
        # Apply changes with decay, update rate, and noise
        for dim, signal in [("pleasure", pleasure_signal), ("arousal", arousal_signal), ("dominance", dominance_signal)]:
            current_val = self.emotion_state.get(dim, 0.0)
            # Map arousal/dominance signals (0-1) to PAD range (-1 to 1), pleasure already directional
            target_val = np.clip(float(signal), -1.0, 1.0) if dim == "pleasure" else np.clip(float(signal) * 2.0 - 1.0, -1.0, 1.0)

            # Decay towards neutral (0.0)
            decayed_val = current_val * (1.0 - EMOTION_DECAY_TO_NEUTRAL)
            # Calculate change towards target value, moderated by update rate
            change_emo = (target_val - decayed_val) * EMOTION_UPDATE_RATE
            # Add random volatility (noise)
            noise = np.random.normal(0, EMOTION_VOLATILITY)
            # Update state, ensuring it stays within [-1, 1]
            self.emotion_state[dim] = float(np.clip(decayed_val + change_emo + noise, -1.0, 1.0))

        # Update the global emotion state
        CURRENT_EMOTION_STATE = self.emotion_state.copy()

        # Set the Limbus node's *own* activation based on the overall emotional intensity.
        # This overrides the activation calculated by its q_system.activate() for this specific module,
        # making its activation reflect the current emotional state directly.
        self.activation = float(np.mean(np.abs(list(self.emotion_state.values()))))

        # Store this calculated activation in its history
        self.activation_history.append(self.activation)
        return self.emotion_state

    def get_emotion_influence_factors(self) -> Dict[str, float]:
        """Provides factors based on the current emotion state to modulate other processes."""
        p = float(self.emotion_state.get("pleasure", 0.0))
        a = float(self.emotion_state.get("arousal", 0.0))
        d = float(self.emotion_state.get("dominance", 0.0))
        return {
            # Modulates overall signal strength in the network
            "signal_modulation": 1.0 + p * 0.15, # Positive mood slightly boosts signals
            # Modulates classical learning rate
            "learning_rate_factor": float(np.clip(1.0 + a * 0.25 + p * 0.10, 0.6, 1.8)),
            # Could influence exploration (e.g., quantum shots or sprouting rate)
            "exploration_factor": float(np.clip(1.0 + a * 0.35 - d * 0.20, 0.5, 1.7)),
            # Modulates the influence/activity of the critic module
            "criticism_weight_factor": float(np.clip(1.0 - p * 0.25 + d * 0.10, 0.6, 1.4)),
            # Modulates the influence/activity of the creative module
            "creativity_weight_factor": float(np.clip(1.0 + p * 0.20 + a * 0.10, 0.6, 1.7)),
        }

class MetaCognitio(Node):
    """Monitors network state, logs reflections, and adapts learning strategies."""
    def __init__(self, label: str = "Meta Cognitio", neuron_type: str = "interneuron"):
        # Meta-cognition uses its own quantum system
        super().__init__(label, neuron_type=neuron_type, num_qubits=NUM_QUBITS_PER_NODE)
        self.reflection_log: deque = deque(maxlen=REFLECTION_LOG_MAX_LEN)
        # Internal state for strategy adaptation
        self.strategy_state: Dict[str, Any] = {
            "lr_boost": 1.0,           # Current learning rate multiplier
            "last_avg_activation": 0.5, # Average activation from previous epoch
            "stagnation_counter": 0,   # Counter for consecutive low-change epochs
            "oscillation_detected": False # Flag if oscillation was recently detected
        }

    def log_reflection(self, message: str, epoch: int, data: Optional[Dict] = None):
        """Logs a meta-cognitive event or observation."""
        log_entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            "message": message,
            "data": data or {} # Include optional structured data
        }
        self.reflection_log.append(log_entry)
        # Also store important reflections persistently
        if persistent_memory_manager:
            persistent_memory_manager.store_reflection(log_entry)

    def adapt_strategy(self, condition: str):
        """Adjusts the learning rate boost based on detected network conditions."""
        lr_boost_before = float(self.strategy_state.get("lr_boost", 1.0))
        new_lr_boost = lr_boost_before # Default to no change

        if condition == "stagnation":
            # Increase learning rate boost significantly if stagnating
            new_lr_boost = min(lr_boost_before * 1.25, 2.5) # Increase by 25%, max boost 2.5x
        elif condition == "oscillation":
            # Decrease learning rate boost if oscillating
            new_lr_boost = max(lr_boost_before * 0.75, 0.5) # Decrease by 25%, min boost 0.5x
        elif condition in ["stagnation_resolved", "oscillation_resolved"]:
            # Gradually return to normal LR after resolving issue
            new_lr_boost = lr_boost_before * 0.9 + 1.0 * 0.1 # Move 10% towards 1.0
        else: # Default case or unknown condition
             # Slowly decay boost back towards 1.0 if no issue detected
            new_lr_boost = lr_boost_before * 0.98 + 1.0 * 0.02

        self.strategy_state["lr_boost"] = float(np.clip(new_lr_boost, 0.5, 2.5)) # Keep boost within bounds

        if abs(self.strategy_state["lr_boost"] - lr_boost_before) > 0.01:
            print(f"[Meta] Strategy Update: Condition '{condition}' -> LR Boost: {self.strategy_state['lr_boost']:.2f}")

    def get_meta_cognitive_state(self) -> Dict[str, Any]:
        """Returns the current state of the meta-cognitive strategies."""
        return self.strategy_state.copy()

    def analyze_network_state(self, all_nodes: List[Node], activation_history_local: Dict[str, deque],
                              weights_history_local: Dict[str, deque], epoch: int):
        """Analyzes the network's activation dynamics to detect stagnation or oscillation."""
        # Filter nodes with valid activation history
        nodes_with_history = [
            n for n in all_nodes
            if hasattr(n, 'activation') and hasattr(n, 'label')
            and isinstance(n.activation, (float, np.number)) and not np.isnan(n.activation)
            and n.label in activation_history_local and activation_history_local[n.label]
        ]
        if not nodes_with_history: return # Cannot analyze without history

        # Calculate current average and standard deviation of activations
        activations = [float(n.activation) for n in nodes_with_history]
        avg_activation = np.mean(activations) if activations else 0.0
        std_activation = np.std(activations) if len(activations) > 1 else 0.0

        # --- Stagnation Detection ---
        last_avg_activation_float = float(self.strategy_state.get("last_avg_activation", 0.5))
        activation_change = abs(avg_activation - last_avg_activation_float)

        # Check if change is below threshold and activation is not already high
        if activation_change < STAGNATION_THRESHOLD and avg_activation < 0.7: # Avoid triggering on stable high activation
            self.strategy_state["stagnation_counter"] += 1
        else:
            # If stagnation was previously detected and now resolved
            if self.strategy_state["stagnation_counter"] >= STAGNATION_DETECTION_WINDOW:
                self.log_reflection(f"Stagnation condition resolved (AvgAct {avg_activation:.3f}, Change {activation_change:.4f})", epoch)
                self.adapt_strategy("stagnation_resolved")
            # Reset stagnation counter if change is sufficient
            self.strategy_state["stagnation_counter"] = 0

        # If stagnation counter reaches window size, log and adapt
        if self.strategy_state["stagnation_counter"] >= STAGNATION_DETECTION_WINDOW:
            if self.strategy_state["stagnation_counter"] == STAGNATION_DETECTION_WINDOW: # Log only once per stagnation period
                 self.log_reflection(f"Stagnation suspected (AvgAct {avg_activation:.3f}, Change {activation_change:.4f} < {STAGNATION_THRESHOLD})", epoch,
                                     data={"avg_act": avg_activation, "change": activation_change})
                 self.adapt_strategy("stagnation")

        # Store current average activation for next epoch's comparison
        self.strategy_state["last_avg_activation"] = avg_activation

        # --- Oscillation Detection ---
        oscillating_nodes = []
        for label, history in activation_history_local.items():
            # Check if enough history exists for the window
            if len(history) >= OSCILLATION_DETECTION_WINDOW:
                # Get the last N activations, ensuring they are numeric
                window_history_raw = list(history)[-OSCILLATION_DETECTION_WINDOW:]
                numeric_history = [h for h in window_history_raw if isinstance(h, (float, np.number)) and not np.isnan(h)]
                # Calculate standard deviation if enough numeric points exist
                if len(numeric_history) >= OSCILLATION_DETECTION_WINDOW // 2: # Require at least half the window size
                     std_dev_window = np.std(numeric_history)
                     if std_dev_window > OSCILLATION_THRESHOLD_STD:
                         oscillating_nodes.append(label)

        # Handle oscillation state changes
        currently_oscillating = len(oscillating_nodes) > 0
        was_oscillating = self.strategy_state.get("oscillation_detected", False)

        if currently_oscillating and not was_oscillating:
            self.log_reflection(f"Oscillations detected in nodes: {oscillating_nodes[:5]}...", epoch, data={"nodes": oscillating_nodes})
            self.adapt_strategy("oscillation")
            self.strategy_state["oscillation_detected"] = True
        elif not currently_oscillating and was_oscillating:
            self.log_reflection("Oscillation condition seems resolved.", epoch)
            self.adapt_strategy("oscillation_resolved")
            self.strategy_state["oscillation_detected"] = False

        # --- Other Meta-Cognitive Observations ---
        # Log memory promotions
        promoted_nodes_this_epoch = [
            n.label for n in all_nodes
            if isinstance(n, MemoryNode) and n.time_in_memory == 0 # time_in_memory resets on promotion
            and n.memory_type != "short_term" # Was promoted *to* mid or long
        ]
        if promoted_nodes_this_epoch:
            self.log_reflection(f"Memory Promotion: Nodes promoted: {promoted_nodes_this_epoch}", epoch)

        # The MetaCognitio node's own activation is determined by its quantum system
        # in the main simulation loop. We just need to record it here.
        if hasattr(self, 'activation') and hasattr(self, 'activation_history'):
             self.activation_history.append(float(self.activation))

class CortexCreativus(Node):
    """Generates new ideas by combining or focusing on active concepts."""
    def __init__(self, label: str = "Cortex Creativus", neuron_type: Optional[str] = None):
        # Uses quantum system, potentially more parameters for richer behavior
        super().__init__(label, neuron_type or random_neuron_type(), num_qubits=NUM_QUBITS_PER_NODE)

    def generate_new_ideas(self, active_nodes: List[Node], creativity_factor: float = 1.0) -> List[str]:
        """Generates potential new ideas based on currently active nodes."""
        ideas = []
        # Adjust activation threshold for considering nodes based on creativity factor
        # Lower threshold means more nodes considered, potentially leading to more diverse ideas
        threshold = max(0.1, 0.5 / max(float(creativity_factor), 0.1)) # Dynamic threshold

        # Filter and sort active nodes based on their quantum activation
        relevant_nodes = [
            n for n in active_nodes
            if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number))
            and not np.isnan(n.activation) and n.activation > threshold and hasattr(n, 'label')
        ]
        relevant_nodes.sort(key=lambda n: float(n.activation), reverse=True)

        # Determine number of ideas to generate based on creativity factor and own activation
        # Own activation influences the 'drive' to generate ideas
        num_ideas_to_generate = int(1 + float(creativity_factor) * 1.2 + self.activation * 2.0)

        # Combine highly active nodes
        if len(relevant_nodes) >= 2:
            for i in range(min(num_ideas_to_generate // 2, len(relevant_nodes) - 1)):
                # Create combined idea labels (truncated for brevity)
                idea_label = f"Idea_comb_{relevant_nodes[i].label[:8]}_and_{relevant_nodes[i+1].label[:8]}"
                ideas.append(idea_label)

        # Focus on single highly active nodes
        if len(relevant_nodes) >= 1:
            ideas.append(f"Idea_focus_on_{relevant_nodes[0].label}")

        # Generate 'wild' ideas by combining less related or random nodes
        if float(creativity_factor) > 1.1 or (len(ideas) < num_ideas_to_generate and active_nodes):
             try:
                 # Pick a random active node
                 random_node1 = random.choice(active_nodes)
                 # Find potential partners (different from random_node1)
                 potential_partners = [n for n in active_nodes if n != random_node1]
                 wild_idea = f"Wild_focus_{getattr(random_node1, 'label', 'Node?')}" # Fallback if no partners
                 if potential_partners:
                     # Combine with another random active node
                     wild_idea = f"Wild_link_{getattr(random_node1, 'label', '?')[:8]}_{getattr(random.choice(potential_partners), 'label', '?')[:8]}"

                 # Add the wild idea if it's novel
                 if wild_idea not in ideas: ideas.append(wild_idea)
             except IndexError: pass # Handle empty active_nodes list

        # Return the generated ideas, capped by the calculated number
        return ideas[:num_ideas_to_generate]

class SimulatrixNeuralis(Node):
    """Simulates potential future scenarios based on active nodes and emotional state."""
    def __init__(self, label: str = "Simulatrix Neuralis", neuron_type: Optional[str] = None):
        # Uses quantum system
        super().__init__(label, neuron_type or random_neuron_type(), num_qubits=NUM_QUBITS_PER_NODE)

    def simulate_scenarios(self, active_nodes: List[Node]) -> List[str]:
        """Generates hypothetical scenarios based on dominant active nodes."""
        scenarios = []
        # Get current emotional state to bias scenario generation (optimistic/pessimistic)
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)
        mood_modifier = "Optimistic" if pleasure > 0.25 else ("Pessimistic" if pleasure < -0.25 else "Neutral")

        # Select highly active nodes to base scenarios on (use higher threshold?)
        scenario_nodes = [
            n for n in active_nodes
            if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number))
            and not np.isnan(n.activation) and n.activation > 0.65 and hasattr(n, 'label') # Higher threshold
        ]
        scenario_nodes.sort(key=lambda n: float(n.activation), reverse=True)

        # Generate scenarios for the top N most active nodes
        for node in scenario_nodes[:3]: # Base scenarios on top 3 active nodes
            base_scenario = f"{mood_modifier}Scenario_if_{node.label}_dominates(Act:{node.activation:.2f})"
            scenarios.append(base_scenario)

            # Generate variations based on active ValueNodes (which have classical activation)
            value_nodes_dict = {v.label: float(v.activation)
                                for v in active_nodes if isinstance(v, ValueNode)} # Get current value strengths
            node_label_lower = node.label.lower() if isinstance(node.label, str) else ""

            # Add cautious variant if 'Sicherheit' value is high and node isn't inherently risky
            if value_nodes_dict.get("Sicherheit", 0.0) > 0.6 and "risiko" not in node_label_lower:
                scenarios.append(f"CautiousVar_of_{node.label}")
            # Add innovative variant if 'Innovation' value is high and node isn't inherently a chance
            if value_nodes_dict.get("Innovation", 0.0) > 0.6 and "chance" not in node_label_lower:
                scenarios.append(f"InnovativeVar_of_{node.label}")

        return scenarios

class CortexCriticus(Node):
    """Evaluates ideas and scenarios based on values, emotions, and internal critique level."""
    def __init__(self, label: str = "Cortex Criticus", neuron_type: Optional[str] = None):
        # Uses quantum system, often inhibitory
        super().__init__(label, neuron_type or "inhibitory", num_qubits=NUM_QUBITS_PER_NODE)

    def evaluate_ideas(self, items_to_evaluate: List[str], current_network_state_nodes: List[Node],
                         criticism_factor: float = 1.0) -> List[Dict]:
        """Assigns a critique score (0.0 to 1.0) to each provided item."""
        evaluated = []
        if not items_to_evaluate: return evaluated

        # Get current value strengths (classical activation)
        value_nodes = {n.label: float(n.activation)
                       for n in current_network_state_nodes if isinstance(n, ValueNode)}
        sicherheit_val = value_nodes.get("Sicherheit", 0.5)
        ethik_val = value_nodes.get("Ethik", 0.5)
        # Get current emotional state
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)

        # Determine base criticism level: influenced by own quantum activation,
        # the external criticism factor, and current pleasure (lower pleasure -> more critical)
        # Own activation reflects internal 'critical drive'
        base_criticism = 0.4 + (self.activation * 0.3) + (float(criticism_factor) - 1.0) * 0.15 - pleasure * 0.2

        for item in items_to_evaluate:
            score_adjustment = 0.0
            item_lower = item.lower() if isinstance(item, str) else ""

            # Adjust score based on keywords and values
            # Semantic analysis of the item description
            if "risiko" in item_lower or "problem" in item_lower or "pessimistic" in item_lower:
                score_adjustment -= 0.25 * sicherheit_val # Higher safety value increases penalty for risk
            if "chance" in item_lower or "potential" in item_lower or "optimistic" in item_lower:
                score_adjustment += 0.15 * (1.0 - sicherheit_val) # Lower safety value increases reward for chance
            if "ethik" in item_lower or "moral" in item_lower:
                score_adjustment += 0.2 * ethik_val # Higher ethics value increases reward
            if "wild" in item_lower: # Wild ideas get a slight penalty based on criticism factor
                score_adjustment -= 0.1 * float(criticism_factor)
            if "cautious" in item_lower: # Cautious variants slightly boosted by safety value
                score_adjustment += 0.05 * sicherheit_val
            if "innovative" in item_lower: # Innovative variants slightly boosted if safety is not dominant
                score_adjustment += 0.08 * (1.0 - sicherheit_val)

            # Calculate raw score and add some minor random fluctuation
            raw_score = base_criticism + score_adjustment + random.uniform(-0.04, 0.04)
            # Clip score to valid range [0, 1]
            final_score = float(np.clip(raw_score, 0.0, 1.0))
            evaluated.append({"item": item, "score": round(final_score, 3)})

        return evaluated

class CortexSocialis(Node):
    """Models social context awareness and applies social influence."""
    def __init__(self, label: str = "Cortex Socialis", neuron_type: Optional[str] = None):
        # Uses quantum system
        super().__init__(label, neuron_type or random_neuron_type(), num_qubits=NUM_QUBITS_PER_NODE)

    def update_social_factors(self, social_network: Dict[str, float], active_nodes: List[Node]) -> Dict[str, float]:
        """Updates the perceived social relevance/influence of different concepts (nodes)."""
        # Get current dominance emotion (influences perception of social dynamics)
        dominance = CURRENT_EMOTION_STATE.get("dominance", 0.0)
        # Global influence modifier: high dominance slightly amplifies perceived changes
        global_influence_mod = 1.0 + dominance * 0.05
        updated_social_network = social_network.copy()

        # Consider MemoryNodes (categories) as the entities within the social context
        target_nodes = [n for n in active_nodes if isinstance(n, MemoryNode) and hasattr(n, 'activation')]

        for node in target_nodes:
            label = getattr(node, 'label', None)
            # Skip if node has no label or is not part of the current social context map
            if label is None or label not in updated_social_network: continue

            activation = float(node.activation) # Quantum activation (normalized Hamming weight)
            current_factor = updated_social_network[label]
            change_factor = 0.0

            # Adjust social factor based on node activation (high activation increases perceived relevance)
            # Thresholds may need tuning based on typical activation range
            if activation > 0.75: change_factor = 0.04 # Strong activation increases factor
            elif activation < 0.35: change_factor = -0.02 # Low activation decreases factor

            # Apply change, modulated by global dominance factor
            new_factor = current_factor + change_factor * global_influence_mod
            # Keep factor within reasonable bounds [0.05, 0.95]
            updated_social_network[label] = float(np.clip(new_factor, 0.05, 0.95))

        return updated_social_network


# --- Network Initialization and Connection ---
def initialize_network_nodes(
    categories: List[str] | np.ndarray | pd.Series,
    initial_values: Dict[str, float] = DEFAULT_VALUES
) -> Tuple[List[MemoryNode], List[Node], List[ValueNode]]:
    """Initializes all nodes (Category, Module, Value) for the network."""
    global CURRENT_EMOTION_STATE
    CURRENT_EMOTION_STATE = INITIAL_EMOTION_STATE.copy() # Reset emotion state

    # Input Validation
    if not isinstance(categories, (list, np.ndarray, pd.Series)) or len(categories) == 0:
        print("FEHLER: Ungültige oder leere Kategorienliste für Initialisierung.")
        return [], [], [] # Return empty lists

    # Convert input categories to a Pandas Series for easier handling
    categories_series: pd.Series
    if isinstance(categories, (np.ndarray, list)):
        categories_series = pd.Series(categories)
    elif isinstance(categories, pd.Series):
        categories_series = categories.copy()
    else:
        # Attempt conversion if unexpected type, issue warning
        try:
            categories_series = pd.Series(list(categories))
            print(f"WARNUNG: Unerwarteter Typ für Kategorien: {type(categories)}. Konvertierung zu Series versucht.")
        except Exception as e:
            print(f"FEHLER: Konnte Kategorien nicht in Series umwandeln: {e}")
            return [], [], []

    # Clean category names and get unique, valid categories
    cleaned_categories = categories_series.astype(str).str.strip().replace('', 'Unkategorisiert').fillna('Unkategorisiert')
    # Filter out 'Unkategorisiert' and potential empty strings after cleaning
    unique_categories = sorted([
        c for c in cleaned_categories.unique()
        if c and c != 'Unkategorisiert'
    ])

    # Check if any valid categories remain after filtering
    if not unique_categories:
        print("WARNUNG: Keine gültigen, nicht-'Unkategorisiert' Kategorien gefunden.")
        category_nodes = [] # Initialize empty list if no valid categories
        # Proceed with module and value nodes anyway
    else:
        print(f"Initialisiere Netzwerk ({NUM_QUBITS_PER_NODE}-Qubit Knoten): {len(unique_categories)} Kategorien, {len(initial_values)} Werte...")
        # Create MemoryNode instances for each unique category (these will have quantum systems)
        category_nodes = [MemoryNode(label=cat) for cat in unique_categories]

    # Initialize cognitive module nodes (these also have quantum systems)
    module_nodes = [
        CortexCreativus(), SimulatrixNeuralis(), CortexCriticus(),
        LimbusAffektus(), MetaCognitio(), CortexSocialis()
    ]
    # Initialize value nodes (these use classical activation, no quantum system)
    value_nodes = [ValueNode(label=key, initial_value=float(value)) for key, value in initial_values.items()]

    print(f"Netzwerk Initialisierung abgeschlossen: {len(category_nodes)} Kategorie-Knoten, "
          f"{len(module_nodes)} Modul-Knoten, {len(value_nodes)} Wert-Knoten.")
    return category_nodes, module_nodes, value_nodes

def connect_network_components(
    category_nodes: List[MemoryNode],
    module_nodes: List[Node],
    question_nodes: List[Node], # Input nodes (typically classical)
    value_nodes: List[ValueNode]
) -> List[Node]:
    """Creates connections between different types of nodes based on predefined rules."""
    print("Verbinde Netzwerkkomponenten...")
    # Combine all nodes into a single list and create a map for easy lookup
    all_nodes_unfiltered = category_nodes + module_nodes + question_nodes + value_nodes
    # Filter out any potential non-Node objects and create map
    all_nodes = [node for node in all_nodes_unfiltered if isinstance(node, Node) and hasattr(node, 'label')]
    node_map = {node.label: node for node in all_nodes}

    # Define connection specifications: source, target, probability, weight range, directionality, special logic
    connection_specs = [
        # Connections between modules (moderate probability, bidirectional)
        {"source_list": module_nodes, "target_list": module_nodes, "prob": 0.4, "weight_range": (0.05, 0.15), "bidirectional": True},
        # Modules influencing categories (higher probability, unidirectional)
        {"source_list": module_nodes, "target_list": category_nodes, "prob": 0.6, "weight_range": (0.1, 0.25), "bidirectional": False},
        # Categories influencing modules (moderate probability, unidirectional)
        {"source_list": category_nodes, "target_list": module_nodes, "prob": 0.5, "weight_range": (0.05, 0.15), "bidirectional": False},
        # Connections between categories (lower probability, bidirectional, represents associations)
        {"source_list": category_nodes, "target_list": category_nodes, "prob": 0.10, "weight_range": (0.02, 0.1), "bidirectional": True}, # Increased intra-category connection prob
        # Input questions connecting strongly to their respective category (unidirectional)
        {"source_list": question_nodes, "target_list": category_nodes, "prob": 1.0, "weight_range": (0.8, 1.0), "bidirectional": False, "special": "question_to_category"},
        # Value nodes influencing specific modules (unidirectional, specific mapping)
        {"source_list": value_nodes, "target_list": module_nodes, "prob": 1.0, "weight_range": (0.1, 0.4), "bidirectional": False, "special": "value_to_module"},
        # Modules influencing specific value nodes (unidirectional, specific mapping)
        {"source_list": module_nodes, "target_list": value_nodes, "prob": 1.0, "weight_range": (0.05, 0.2), "bidirectional": False, "special": "module_to_value"},
        # Value nodes influencing related categories (lower probability, bidirectional, thematic links)
        {"source_list": value_nodes, "target_list": category_nodes, "prob": 0.25, "weight_range": (0.05, 0.15), "bidirectional": True, "special": "value_category_thematic"}, # Increased value-category prob
    ]

    connections_created = 0
    # Iterate through each connection specification
    for spec in connection_specs:
        # Get source and target nodes valid within the current network map
        source_list = [node for node in spec["source_list"] if hasattr(node, 'label') and node.label in node_map]
        target_list = [node for node in spec["target_list"] if hasattr(node, 'label') and node.label in node_map]
        prob, (w_min, w_max), bidi = spec["prob"], spec["weight_range"], spec["bidirectional"]
        special_logic = spec.get("special")

        # Iterate through potential source nodes
        for i, src_node in enumerate(source_list):
            if not hasattr(src_node, 'connections'): continue # Skip if source cannot have connections

            # Adjust starting index for target loop to avoid duplicate checks in bidirectional cases
            start_j = i + 1 if (source_list is target_list and bidi) else 0

            # Iterate through potential target nodes
            for j in range(start_j, len(target_list)):
                tgt_node = target_list[j]
                # Skip if target is invalid or same as source
                if not isinstance(tgt_node, Node) or src_node == tgt_node: continue
                # Skip if target cannot have connections (needed for bidirectional check)
                if bidi and not hasattr(tgt_node, 'connections'): continue

                # Probabilistic check for standard connections
                if random.random() >= prob and special_logic is None: continue

                # Determine connection weight
                weight = random.uniform(w_min, w_max)
                connected = False # Flag to track if a connection was made

                # --- Apply Special Connection Logic ---
                if special_logic == "question_to_category":
                    # Connect question node Q_idx_Category_Text to category node Category
                    if not hasattr(src_node, 'label') or not hasattr(tgt_node, 'label'): continue
                    try:
                        # Extract category name from question label (assuming format Q_idx_Category_...)
                        q_label_parts = src_node.label.split('_', 2) # Split max 2 times
                        cat_label_from_q = q_label_parts[2] if len(q_label_parts) > 2 else q_label_parts[-1] # Take last part
                    except (IndexError, AttributeError):
                        continue # Skip if label format is unexpected
                    # Compare extracted category with target node label (case-insensitive)
                    if tgt_node.label.lower() == cat_label_from_q.lower():
                        src_node.add_connection(tgt_node, weight=weight)
                        connections_created += 1
                        connected = True

                elif special_logic == "value_to_module":
                    # Connect specific values to specific modules
                    if not hasattr(src_node, 'label') or not hasattr(tgt_node, 'label'): continue
                    # Define the value-to-module mapping {ValueLabel: (TargetModuleLabel, WeightFactor)}
                    v_m_map = {
                        "Innovation": ("Cortex Creativus", 1.0), "Sicherheit": ("Cortex Criticus", 1.0),
                        "Effizienz": ("Meta Cognitio", 0.8), "Neugier": ("Cortex Creativus", 0.5),
                        "Ethik": ("Cortex Criticus", 0.9) # Ethik primarily influences Criticus
                    }
                    if src_node.label in v_m_map:
                        target_module_label, factor = v_m_map[src_node.label]
                        # Check if the target node is the specified module
                        if tgt_node.label == target_module_label:
                            src_node.add_connection(tgt_node, weight=(weight * factor))
                            connections_created += 1
                            connected = True

                elif special_logic == "module_to_value":
                     # Connect specific modules back to specific values
                     if not hasattr(src_node, 'label') or not hasattr(tgt_node, 'label'): continue
                     # Define the module-to-value mapping {ModuleLabel: (TargetValueLabel, WeightFactor)}
                     m_v_map={
                         "Cortex Creativus":("Innovation", 0.7), # Creativity boosts Innovation value
                         "Cortex Criticus":("Sicherheit", 0.7), # Criticism boosts Sicherheit value
                         "Meta Cognitio":("Effizienz", 0.6),    # Meta-cognition boosts Effizienz value
                         "CortexCriticus_Ethik":("Ethik", 0.6)  # Specific case: Critic also boosts Ethik
                         # Limbus could potentially influence Neugier?
                     }
                     # Handle specific Critic->Ethik case first
                     if src_node.label=="Cortex Criticus" and tgt_node.label=="Ethik":
                         factor=m_v_map["CortexCriticus_Ethik"][1]
                         src_node.add_connection(tgt_node, weight=(weight * factor))
                         connections_created += 1
                         connected = True
                     # Handle general cases
                     elif src_node.label in m_v_map and m_v_map[src_node.label][0] == tgt_node.label:
                         target_value_label, factor = m_v_map[src_node.label]
                         src_node.add_connection(tgt_node, weight=(weight * factor))
                         connections_created += 1
                         connected = True

                elif special_logic == "value_category_thematic":
                    # Connect values to categories with related themes
                    if not hasattr(src_node, 'label') or not hasattr(tgt_node, 'label'): continue
                    cat_lower = tgt_node.label.lower() # Target is category node
                    value_label = src_node.label # Source is value node
                    connect = False
                    # Define thematic links (e.g., Sicherheit -> 'risiko', Innovation -> 'chance')
                    if value_label == "Sicherheit" and ("risiko" in cat_lower or "problem" in cat_lower or "bedrohung" in cat_lower): connect = True
                    if value_label == "Innovation" and ("chance" in cat_lower or "neu" in cat_lower or "potential" in cat_lower): connect = True
                    if value_label == "Ethik" and ("ethik" in cat_lower or "moral" in cat_lower): connect = True
                    if value_label == "Effizienz" and ("effizienz" in cat_lower or "optimierung" in cat_lower): connect = True
                    if value_label == "Neugier" and ("forschung" in cat_lower or "entdeckung" in cat_lower or "unbekannt" in cat_lower): connect = True

                    if connect:
                        # Create connection (potentially stronger weight for thematic link)
                        src_node.add_connection(tgt_node, weight=(weight * 1.3))
                        connections_created += 1
                        # If bidirectional, add return connection (usually weaker)
                        if bidi and hasattr(tgt_node, 'connections'):
                            tgt_node.add_connection(src_node, weight=(weight * 0.6))
                            connections_created += 1
                        connected = True

                elif special_logic is None: # Standard connection based on probability
                    src_node.add_connection(tgt_node, weight=weight)
                    connections_created += 1
                    # Add return connection if bidirectional
                    if bidi and hasattr(tgt_node, 'connections'):
                        # Return connection might have slightly different weight characteristics
                        tgt_node.add_connection(src_node, weight=random.uniform(w_min * 0.8, w_max * 0.8))
                        connections_created += 1
                    connected = True

    print(f"Netzwerkkomponenten verbunden. Insgesamt {connections_created} Verbindungen erstellt.")
    # Return the original list, potentially containing disconnected nodes if filtering occurred
    return all_nodes_unfiltered


# --- Signal Propagation (Calculates Classical Input Sum) ---
def calculate_classic_input_sum(nodes_list: List[Node], emotion_factors: Dict[str, float], value_state: Dict[str, float]):
    """
    Calculates the weighted sum of incoming signals for each node based on the
    activations from the *previous* time step. This sum serves as the classical
    input for the node's quantum activation in the *current* time step.
    """
    # Reset the activation sum for all nodes before calculation
    for node in nodes_list:
        if hasattr(node, 'activation_sum'):
             node.activation_sum = 0.0 # Initialize sum to zero

    # Get global signal modulation factor from emotions
    signal_modulation = float(emotion_factors.get("signal_modulation", 1.0))

    # Iterate through each node as a potential signal source
    for source_node in nodes_list:
        # Get the activation from the *previous* step (stored in node.activation)
        # Ensure activation is a valid float
        current_activation_raw = getattr(source_node, 'activation', 0.0)
        current_activation = float(current_activation_raw) if isinstance(current_activation_raw, (float, np.number)) and not np.isnan(current_activation_raw) else 0.0

        # Skip if node has no connections or its activation is negligible
        if not hasattr(source_node, 'connections') or not source_node.connections or current_activation < 0.01:
            continue

        # Determine base signal strength, modulated by emotion
        base_signal = current_activation * signal_modulation
        # Check if the source node is inhibitory
        is_inhibitory = hasattr(source_node, 'neuron_type') and source_node.neuron_type == "inhibitory"

        # Propagate the signal through outgoing connections
        for connection in source_node.connections:
             target_node = connection.target_node
             # Ensure target exists and can receive input sum
             if not target_node or not hasattr(target_node, 'activation_sum'): continue

             # Get connection weight, ensure it's float
             conn_weight = float(connection.weight)
             # Calculate signal strength for this specific connection
             signal_strength = base_signal * conn_weight

             # Apply inhibitory effect if source node is inhibitory
             if is_inhibitory:
                 signal_strength *= -1.5 # Inhibitory signals have stronger negative impact

             # Apply classical value influence (modulates signal based on target node type and related values)
             target_value_mod = 1.0 # Default multiplier
             # Example: Increase input to Criticus if 'Sicherheit' value is high
             if isinstance(target_node, CortexCriticus):
                 target_value_mod *= 1.0 + float(value_state.get("Sicherheit", 0.5)) * VALUE_INFLUENCE_FACTOR * 1.5
             # Example: Increase input to Creativus if 'Innovation' value is high
             elif isinstance(target_node, CortexCreativus):
                 target_value_mod *= 1.0 + float(value_state.get("Innovation", 0.5)) * VALUE_INFLUENCE_FACTOR * 1.5
             # Example: Increase input to MetaCognitio if 'Effizienz' value is high
             elif isinstance(target_node, MetaCognitio):
                 target_value_mod *= 1.0 + float(value_state.get("Effizienz", 0.5)) * VALUE_INFLUENCE_FACTOR
             # Add more rules for other target types or values as needed

             final_signal = signal_strength * target_value_mod

             # Add the calculated signal to the target node's activation sum
             # Clip the sum to prevent extreme values which might destabilize quantum activation
             target_node.activation_sum = float(np.clip(target_node.activation_sum + final_signal, -75.0, 75.0)) # Wider clip range?

# --- Core Simulation Cycle (Multi-Qubit Version) ---
def simulate_learning_cycle(
    data: pd.DataFrame,
    category_nodes: List[MemoryNode],
    module_nodes: List[Node],
    value_nodes: List[ValueNode],
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE, # Classical LR
    reward_interval: int = DEFAULT_REWARD_INTERVAL,
    decay_rate: float = DEFAULT_DECAY_RATE,      # Classical decay
    initial_emotion_state: Optional[Dict[str, float]] = None,
    context_factors: Optional[Dict[str, Any]] = None, # External context (optional)
    persistent_memory: Optional[PersistentMemoryManager] = None,
    load_state_from: Optional[str] = None,       # File to load state from
    status_callback: Callable[[str], None] = _default_status_callback,
    quantum_shots_per_node: int = QUANTUM_ACTIVATION_SHOTS, # Shots for Q-activation
    quantum_param_lr: float = QUANTUM_PARAM_LEARNING_RATE # LR for Q-parameters
) -> Tuple[Dict, Dict, List, List[MemoryNode], List[Node], List[ValueNode], List[Node], Dict, Dict]:
    """
    Runs the main simulation loop for the NeuroPersona network with multi-qubit nodes.

    Returns:
        Tuple containing final histories (activation, weights, q_params, values),
        interpretation log, final node lists (categories, modules, values, all),
    """
    global CURRENT_EMOTION_STATE, activation_history # Use global variables

    status_callback(f"Beginne Simulationszyklus ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2 - EXPERIMENTELL)...")
    start_sim_time = time.time()

    # --- Initialization ---
    if initial_emotion_state is None: initial_emotion_state = INITIAL_EMOTION_STATE.copy()
    CURRENT_EMOTION_STATE = initial_emotion_state.copy()
    if context_factors is None: context_factors = {}

    # Create dictionaries for quick access to modules and values
    module_dict = {m.label: m for m in module_nodes if hasattr(m, 'label')}
    value_dict  = {v.label: v for v in value_nodes if hasattr(v, 'label')}

    # Initialize history storage
    weights_history: Dict[str, deque] = {} # Stores classical connection weights over time
    activation_history.clear() # Clear global activation history at start
    q_param_history: Dict[str, deque] = {} # Stores history of the *first* quantum parameter for nodes
    module_outputs_log = {label: deque(maxlen=HISTORY_MAXLEN // 2) for label in module_dict.keys()} # Log module outputs
    interpretation_log: List[Dict] = [] # Stores interpretation of each epoch
    value_history: Dict[str, deque] = {v.label: deque(maxlen=HISTORY_MAXLEN) for v in value_nodes} # History of classical value activations

    # --- Create Input Nodes (Classical Nodes) ---
    question_nodes: List[Node] = []
    cat_node_map = {node.label: node for node in category_nodes if hasattr(node, 'label')} # Map category labels to nodes

    for idx, row in data.iterrows():
        # Ensure category exists, default to 'Unkategorisiert' if missing/invalid
        cat_label = str(row.get('Kategorie', 'Unkategorisiert')).strip()
        if not cat_label: cat_label = 'Unkategorisiert'

        # Create a unique label for the question node
        q_label = f"Q_{idx}_{cat_label}_{str(row.get('Frage', 'Frage?'))[:25]}"
        # Create as a standard Node, but explicitly disable its quantum system (num_qubits=0)
        q_node = Node(q_label, neuron_type="excitatory", num_qubits=0) # No quantum system for inputs
        # q_node.q_system = None # Explicitly set to None (already done by num_qubits=0)
        question_nodes.append(q_node)

    # --- Connect Network Components ---
    all_nodes_sim_unfiltered = connect_network_components(category_nodes, module_nodes, question_nodes, value_nodes)
    # Filter again to ensure all elements are valid Nodes
    all_nodes_sim = [node for node in all_nodes_sim_unfiltered if isinstance(node, Node)]
    if not all_nodes_sim:
        raise RuntimeError("Network initialization failed: No valid nodes after connection.")

    # Initialize history deques for all valid nodes
    activation_history = {node.label: deque(maxlen=HISTORY_MAXLEN) for node in all_nodes_sim if hasattr(node, 'label')}
    # Initialize QParam history only for nodes that *have* a quantum system
    q_param_history = {
        node.label: deque(maxlen=HISTORY_MAXLEN)
        for node in all_nodes_sim
        if hasattr(node, 'label') and hasattr(node, 'q_system') and node.q_system is not None and node.q_system.num_params > 0
    }
    # Initialize weight history keys based on existing connections
    for node in all_nodes_sim:
        if hasattr(node, 'connections') and hasattr(node, 'label'):
            for conn in node.connections:
                 if conn.target_node and hasattr(conn.target_node, 'label'):
                      history_key = f"{node.label} → {conn.target_node.label}"
                      weights_history.setdefault(history_key, deque(maxlen=HISTORY_MAXLEN))

    # --- Load State (Optional) ---
    # This section needs to correctly handle the multi-qubit parameter arrays
    if load_state_from and os.path.exists(load_state_from):
        status_callback(f"Versuche, Zustand aus {load_state_from} zu laden...")
        try:
            with open(load_state_from, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # Load classical components
            CURRENT_EMOTION_STATE = loaded_data.get('emotion_state', INITIAL_EMOTION_STATE.copy())
            loaded_connections = {(c['source'], c['target']): float(c['weight'])
                                  for c in loaded_data.get('connections', [])}
            loaded_values = loaded_data.get('value_node_states', {})
            # Load quantum parameters and activations
            loaded_nodes_data = {n['label']: n for n in loaded_data.get('nodes', []) if 'label' in n}

            node_map_load = {node.label: node for node in all_nodes_sim if hasattr(node, 'label')}

            loaded_q_param_count = 0
            # Apply loaded states to the current network nodes
            for node_label, node in node_map_load.items():
                if node_label in loaded_nodes_data:
                    node_data = loaded_nodes_data[node_label]
                    # Set last activation (important for first step calculation)
                    node.activation = float(node_data.get('activation', 0.0))

                    # Load ValueNode state (classical)
                    if isinstance(node, ValueNode) and node_label in loaded_values:
                         node.activation = loaded_values[node_label] # Overwrite activation with saved value

                    # Load Quantum Parameters if node has a q_system
                    elif hasattr(node, 'q_system') and node.q_system is not None:
                        if 'q_params' in node_data:
                            try:
                                params_to_load = np.array(node_data['q_params'], dtype=float)
                                # CRITICAL: Check if the loaded shape matches the expected shape
                                if params_to_load.shape == node.q_system.params.shape:
                                    node.q_system.set_params(params_to_load)
                                    loaded_q_param_count += 1
                                else:
                                    print(f"WARNUNG: Geladene QParams für {node_label} haben falsches Shape "
                                          f"({params_to_load.shape} vs {node.q_system.params.shape}). Ignoriere Parameter.")
                            except Exception as q_param_err:
                                print(f"FEHLER beim Laden/Setzen der QParams für {node_label}: {q_param_err}")
                        else:
                             print(f"INFO: Keine 'q_params' im gespeicherten Zustand für Quantenknoten {node_label} gefunden.")

                # Apply loaded connection weights
                if hasattr(node, 'connections'):
                    for conn in node.connections:
                        if conn.target_node and hasattr(conn.target_node, 'label'):
                            conn_key = (node.label, conn.target_node.label)
                            if conn_key in loaded_connections:
                                conn.weight = loaded_connections[conn_key]

            status_callback(f"Netzwerkzustand geladen ({loaded_q_param_count} Knoten mit QParams).")
            # TODO: Optionally load history deques if saved?

        except FileNotFoundError:
             status_callback(f"WARNUNG: Speicherdatei {load_state_from} nicht gefunden. Starte mit Initialzustand.")
        except Exception as e:
            status_callback(f"FEHLER beim Laden des Zustands aus {load_state_from}: {e}. Starte mit Initialzustand.")
            traceback.print_exc()
            # Reset state if loading failed to ensure clean start
            CURRENT_EMOTION_STATE = INITIAL_EMOTION_STATE.copy()
            # Consider re-initializing nodes/weights here if load fails critically


    # Initialize social network factors (if not loaded)
    # Check if 'social_network' was loaded, otherwise initialize
    # Assuming it's not saved/loaded for now, initialize randomly
    social_network = {node.label: random.uniform(0.2, 0.6)
                     for node in category_nodes if hasattr(node, 'label')}

    base_lr = learning_rate # Base classical learning rate
    current_dr = decay_rate # Base classical decay rate

    status_callback("Starte Epochen-Simulation...")
    limbus_module = module_dict.get("Limbus Affektus")
    meta_cog_module = module_dict.get("Meta Cognitio")

    # --- Main Simulation Loop ---
    iterator = range(epochs)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Sim Cognitive Cycle (MQ)", unit="epoch", dynamic_ncols=True)

    for epoch in iterator:
        epoch_start_time = time.time()

        # --- A. Get Current Context ---
        # Get emotional modulation factors (classical factors based on PAD state)
        emotion_factors = limbus_module.get_emotion_influence_factors() if isinstance(limbus_module, LimbusAffektus) else {}
        # Get current value node activations (classical state)
        current_value_state = {v.label: float(v.activation) for v in value_nodes if hasattr(v,'label')}

        # --- B. Calculate Classical Input Sum ---
        # This uses the activations from the *end* of the previous epoch (node.activation)
        # to calculate the input sum (node.activation_sum) for the *current* epoch's quantum activation.
        calculate_classic_input_sum(all_nodes_sim, emotion_factors, current_value_state)

        # --- C. Set Input Node Activations ---
        # Activate question nodes based on the input data (classical activation)
        for idx, q_node in enumerate(question_nodes):
             if idx < len(data):
                 # Get normalized answer from preprocessed data
                 norm_answer = data['normalized_answer'].iloc[idx]
                 q_node.activation = float(norm_answer) # Direct classical activation
             else:
                 q_node.activation = 0.0 # Default to zero if no corresponding data row
             # Record activation in history
             if hasattr(q_node, 'label') and q_node.label in activation_history:
                 activation_history[q_node.label].append(q_node.activation)

        # --- D. Calculate Quantum Activations ---
        # Calculate activation for all nodes possessing a quantum system
        all_active_nodes_this_epoch = [] # List to store nodes active in this epoch
        # Nodes requiring quantum activation (categories and modules)
        nodes_to_activate_quanten = [node for node in all_nodes_sim if hasattr(node, 'q_system') and node.q_system is not None]
        # Shuffle order of activation calculation slightly? (Optional)
        # random.shuffle(nodes_to_activate_quanten)

        nodes_with_q_activation_count = 0
        for node in nodes_to_activate_quanten:
            # Get the classical input sum calculated in step B
            classic_input = node.activation_sum
            node_label_for_log = getattr(node, 'label', None) # Get label for logger
            # Run the quantum circuit simulation to get the new activation (normalized Hamming weight)
            try:
                # Pass node label to activate for better log filenames
                new_activation = node.q_system.activate(classic_input,
                                                        n_shots=quantum_shots_per_node,
                                                        node_label=node_label_for_log)
                node.activation = new_activation # Update the node's activation state
                nodes_with_q_activation_count += 1
            except Exception as q_act_err:
                 print(f"ERROR during quantum activation for node {node.label}: {q_act_err}")
                 node.activation = 0.0 # Set to inactive on error

            # Store activation in history (for all nodes with labels, quantum or classical)
            if hasattr(node, 'label') and node.label in activation_history:
                # Ensure activation is float before appending
                activation_to_store = float(node.activation) if isinstance(node.activation, (float, np.number)) else 0.0
                activation_history[node.label].append(activation_to_store)

            # Store the *first* quantum parameter in its history deque for analysis
            if hasattr(node, 'q_system') and node.q_system and node.q_system.num_params > 0 and node.label in q_param_history:
                try:
                    first_param = float(node.q_system.get_params()[0])
                    q_param_history[node.label].append(first_param)
                except (IndexError, TypeError):
                    pass # Ignore if params are somehow unavailable

            # Collect nodes considered 'active' based on activation threshold
            # This threshold might need tuning for normalized Hamming weight
            if hasattr(node, 'activation') and node.activation > 0.25: # Lower threshold?
                all_active_nodes_this_epoch.append(node)

        # --- E. Update Classical Nodes & States ---
        # Add ValueNode activations to their history (they don't run q_system.activate)
        for v_node in value_nodes:
             if hasattr(v_node, 'label') and v_node.label in value_history:
                 value_history[v_node.label].append(float(v_node.activation))
             # Add classical value nodes to active list if they meet threshold
             if hasattr(v_node, 'activation') and v_node.activation > 0.25 and v_node not in all_active_nodes_this_epoch:
                  all_active_nodes_this_epoch.append(v_node)

        # Update emotional state (classical calculation using quantum activations as input)
        if isinstance(limbus_module, LimbusAffektus):
            new_emotion_state = limbus_module.update_emotion_state(all_nodes_sim, module_outputs_log)
            # Log the new emotion state if available
            if isinstance(new_emotion_state, dict):
                 module_outputs_log["Limbus Affektus"].append(new_emotion_state.copy())
            # Get updated emotion factors for the next steps
            emotion_factors = limbus_module.get_emotion_influence_factors()

        # --- F. Run Cognitive Module Logic ---
        # Modules perform their classical functions based on the network's current quantum activation state
        ideas, evaluated, scenarios = [], [], []
        # Get modulation factors from emotion state
        creativity_factor = emotion_factors.get('creativity_weight_factor', 1.0)
        criticism_factor = emotion_factors.get('criticism_weight_factor', 1.0)

        # Run Creativity Module
        if "Cortex Creativus" in module_dict and isinstance(module_dict["Cortex Creativus"], CortexCreativus):
            ideas = module_dict["Cortex Creativus"].generate_new_ideas(all_active_nodes_this_epoch, creativity_factor)
            if ideas: module_outputs_log["Cortex Creativus"].append(ideas)

        # Run Simulation Module
        if "Simulatrix Neuralis" in module_dict and isinstance(module_dict["Simulatrix Neuralis"], SimulatrixNeuralis):
            scenarios = module_dict["Simulatrix Neuralis"].simulate_scenarios(all_active_nodes_this_epoch)
            if scenarios: module_outputs_log["Simulatrix Neuralis"].append(scenarios)

        # Run Critic Module
        if "Cortex Criticus" in module_dict and isinstance(module_dict["Cortex Criticus"], CortexCriticus):
            items_to_evaluate = ideas + scenarios # Evaluate both new ideas and scenarios
            if items_to_evaluate:
                 evaluated = module_dict["Cortex Criticus"].evaluate_ideas(items_to_evaluate, all_nodes_sim, criticism_factor)
                 if evaluated: module_outputs_log["Cortex Criticus"].append(evaluated)

        # Run Social Module & Apply Influence
        if "Cortex Socialis" in module_dict and isinstance(module_dict["Cortex Socialis"], CortexSocialis):
            current_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
            social_network = module_dict["Cortex Socialis"].update_social_factors(social_network, current_category_nodes)
            if isinstance(social_network, dict): module_outputs_log["Cortex Socialis"].append({"factors_updated": True})
            # Apply the updated social factors to network connections
            social_influence(all_nodes_sim, social_network)

        # --- G. Update Value Nodes (Classical) ---
        for v_node in value_nodes:
            if isinstance(v_node, ValueNode) and hasattr(v_node, 'label'):
                # Calculate adjustment based on network state and module outputs
                adjustment = calculate_value_adjustment(
                    v_node.label, all_nodes_sim, module_outputs_log, value_dict, module_dict
                )
                v_node.update_value(adjustment)

        # --- H. Meta-Cognition and Adaptive Learning Rate ---
        dynamic_lr = base_lr # Start with base classical LR
        if isinstance(meta_cog_module, MetaCognitio):
            # Analyze network state using current quantum activations and history
            meta_cog_module.analyze_network_state(all_nodes_sim, activation_history, weights_history, epoch + 1)
            meta_cognitive_state = meta_cog_module.get_meta_cognitive_state()
            # Calculate dynamically adjusted classical learning rate
            dynamic_lr = calculate_dynamic_learning_rate(base_lr, CURRENT_EMOTION_STATE, meta_cognitive_state)
            # Log meta-cognitive state
            if isinstance(meta_cognitive_state, dict):
                 module_outputs_log["Meta Cognitio"].append({"lr_classical": dynamic_lr, "state": meta_cognitive_state.copy()})

        # --- I. Learning Step (Hebbian with Quantum Feedback) ---
        # Apply learning rule to connections based on co-activation
        learning_applied_count = 0
        for node in all_nodes_sim:
            if hasattr(node, 'connections'):
                 for conn in node.connections:
                     # Apply the modified Hebbian rule which updates classical weight
                     # and provides feedback to presynaptic quantum parameters
                     hebbian_learning_quantum_node(
                         node_a=node, connection=conn,
                         learning_rate_classical=dynamic_lr, # Use dynamic classical LR
                         learning_rate_quantum=quantum_param_lr, # Use separate QParam LR
                         weight_limit=1.0,
                         reg_factor=0.001 # Regularization factor
                     )
                     learning_applied_count +=1

        # --- J. Reinforcement and Decay ---
        # Apply reinforcement periodically based on pleasure and critic score
        if (epoch + 1) % reward_interval == 0:
            apply_reinforcement(all_nodes_sim, module_outputs_log)

        # Apply decay to classical connection weights
        decay_weights(all_nodes_sim, current_dr)

        # --- K. Structural Plasticity ---
        # Prune weak connections and sprout new ones periodically
        if (epoch + 1) % STRUCTURAL_PLASTICITY_INTERVAL == 0:
            pruned_conn = prune_connections(all_nodes_sim, threshold=PRUNING_THRESHOLD)
            sprouted_conn = sprout_connections(all_nodes_sim, activation_history, threshold=SPROUTING_THRESHOLD, max_conns=MAX_CONNECTIONS_PER_NODE, new_weight_mean=SPROUTING_NEW_WEIGHT_MEAN)
            # Prune inactive nodes (if enabled)
            pruned_nodes_list, pruned_nodes_count = prune_inactive_nodes(all_nodes_sim, activation_history, epoch + 1, enabled=NODE_PRUNING_ENABLED)

            # If nodes were pruned, update the main list and derived structures
            if pruned_nodes_count > 0:
                status_callback(f"E{epoch+1}: Plastizität (-{pruned_conn} conn, +{sprouted_conn} conn, -{pruned_nodes_count} nodes)")
                all_nodes_sim = pruned_nodes_list # Update the active node list
                # Rebuild node map and filter histories
                node_map_updated = {node.label: node for node in all_nodes_sim if hasattr(node, 'label')}
                activation_history = {k: v for k, v in activation_history.items() if k in node_map_updated}
                q_param_history = {k: v for k, v in q_param_history.items() if k in node_map_updated}
                # Remove entries from weight history if either source or target was pruned
                keys_to_remove = {key for key in weights_history
                                  if key.split(" → ")[0] not in node_map_updated or key.split(" → ")[1] not in node_map_updated}
                for key in keys_to_remove: del weights_history[key]
            elif pruned_conn > 0 or sprouted_conn > 0:
                 status_callback(f"E{epoch+1}: Plastizität (-{pruned_conn} conn, +{sprouted_conn} conn, -0 nodes)")


        # --- L. Log Connection Weights ---
        # Record the state of classical weights at the end of the epoch
        for node in all_nodes_sim:
            if hasattr(node, 'connections') and hasattr(node, 'label'):
                for conn in node.connections:
                     target_node = conn.target_node
                     if target_node and hasattr(target_node, 'label'):
                          history_key = f"{node.label} → {target_node.label}"
                          current_weight = float(conn.weight) # Ensure float
                          weights_history.setdefault(history_key, deque(maxlen=HISTORY_MAXLEN)).append(current_weight)

        # --- M. Memory Consolidation ---
        # Periodically store highly relevant long-term memories persistently
        if persistent_memory and (epoch + 1) % MEMORY_CONSOLIDATION_INTERVAL == 0:
            consolidate_memories(all_nodes_sim, persistent_memory, epoch, status_callback)

        # --- N. Memory Promotion ---
        # Check if any MemoryNodes qualify for promotion based on sustained activation
        for node in all_nodes_sim:
            if isinstance(node, MemoryNode):
                node.promote(activation_threshold=DEFAULT_ACTIVATION_THRESHOLD_PROMOTION)

        # --- O. Interpret Epoch State ---
        # Analyze the state at the end of the epoch for logging and reporting
        # Get current lists of nodes (might have changed due to pruning)
        current_final_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
        current_final_module_nodes = [n for n in all_nodes_sim if isinstance(n, Node) and not isinstance(n, MemoryNode) and not isinstance(n, ValueNode) and not n.label.startswith("Q_")]
        current_final_value_nodes = [n for n in all_nodes_sim if isinstance(n, ValueNode)]
        epoch_interpretation = interpret_epoch_state(
            epoch, current_final_category_nodes, current_final_module_nodes,
            module_outputs_log, activation_history, current_final_value_nodes
        )
        interpretation_log.append(epoch_interpretation)

        # --- P. Update Progress Bar (Optional) ---
        if TQDM_AVAILABLE and isinstance(iterator, tqdm):
             dom_cat = str(epoch_interpretation.get('dominant_category','?'))[:12] # Truncate label
             dom_act = epoch_interpretation.get('dominant_activation',0)
             pleasure = CURRENT_EMOTION_STATE.get('pleasure', 0)
             # Optionally display average QParam change or other metrics
             iterator.set_description(f"Sim E{epoch+1}/{epochs} | Dom:{dom_cat}({dom_act:.2f}) P:{pleasure:.2f} LRc:{dynamic_lr:.4f}")

        # --- Q. End of Epoch Housekeeping ---
        epoch_duration = time.time() - epoch_start_time
        # Optional: print(f"Epoch {epoch+1} duration: {epoch_duration:.3f}s")


    # --- End of Simulation Loop ---
    status_callback("Simulationszyklus abgeschlossen.")
    end_sim_time = time.time()
    print(f"Gesamte Simulationsdauer: {end_sim_time - start_sim_time:.2f} Sekunden")

    # --- Collect Final Results ---
    status_callback("Sammle finale Netzwerkzustände und Historien...")
    # Convert deques to lists for easier handling/serialization
    final_activation_history = {k: list(v) for k, v in activation_history.items() if v}
    final_weights_history = {k: list(v) for k, v in weights_history.items() if v}
    final_value_history = {k: list(v) for k, v in value_history.items() if v}
    final_q_param_history = {k: list(v) for k, v in q_param_history.items() if v} # Include QParam history

    # Get final lists of nodes after potential pruning
    final_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
    # Ensure final_module_nodes only includes actual module instances
    module_labels_set = { "Cortex Creativus", "Simulatrix Neuralis", "Cortex Criticus",
                          "Limbus Affektus", "Meta Cognitio", "Cortex Socialis"}
    final_module_nodes = [n for n in all_nodes_sim if hasattr(n, 'label') and n.label in module_labels_set]
    final_value_nodes = [n for n in all_nodes_sim if isinstance(n, ValueNode)]

    # Return all collected data
    return (final_activation_history, final_weights_history, interpretation_log,
            final_category_nodes, final_module_nodes, final_value_nodes, all_nodes_sim,
            final_value_history, final_q_param_history) # Added q_param_history


# --- Simulation Cycle Helper Functions ---
def calculate_value_adjustment(value_label: str, all_nodes: List[Node], module_outputs_log: Dict[str, deque],
                               value_dict: Dict[str, ValueNode], module_dict: Dict[str, Node]) -> float:
    """
    Calculates the adjustment for a specific ValueNode based on network state.
    Operates primarily on classical logic, using quantum activations as inputs.
    """
    adjustment = 0.0

    # Extract current activations (quantum for modules/categories, classical for values)
    module_activations = {
        m.label: float(m.activation)
        for m in all_nodes
        if hasattr(m, 'label') and m.label in module_dict # Check if it's a known module
        and hasattr(m, 'activation') and isinstance(m.activation, (float, np.number))
        and not np.isnan(m.activation)
    }
    category_activations = {
        c.label: float(c.activation)
        for c in all_nodes
        if isinstance(c, MemoryNode) and hasattr(c, 'label') # Check if it's a MemoryNode
        and hasattr(c, 'activation') and isinstance(c.activation, (float, np.number))
        and not np.isnan(c.activation)
    }
    # Value activations are needed for some cross-value influence
    # value_activations = {label: node.activation for label, node in value_dict.items()} # Already available via value_dict

    # --- Value-Specific Adjustment Logic ---
    if value_label == "Innovation":
        # Boosted by Cortex Creativus activation
        creativus_act = module_activations.get("Cortex Creativus", 0.5) # Default 0.5 if module inactive
        adjustment += (creativus_act - 0.5) * 0.5 # Innovation increases if Creativus > 0.5
        # Boosted by 'chance' or 'potential' related categories
        chance_acts = [act for label, act in category_activations.items()
                       if "chance" in label.lower() or "potential" in label.lower()]
        avg_chance_act = np.mean(chance_acts) if chance_acts else 0.0
        adjustment += (avg_chance_act - 0.4) * 0.3 # Increase if average chance activation > 0.4

    elif value_label == "Sicherheit":
        # Boosted by Cortex Criticus activation
        criticus_act = module_activations.get("Cortex Criticus", 0.5)
        adjustment += (criticus_act - 0.5) * 0.6 # Sicherheit increases if Criticus > 0.5
        # Boosted strongly by 'risiko' or 'problem' related categories
        risiko_acts = [act for label, act in category_activations.items()
                       if "risiko" in label.lower() or "problem" in label.lower() or "bedrohung" in label.lower()]
        avg_risiko_act = np.mean(risiko_acts) if risiko_acts else 0.0
        adjustment += (avg_risiko_act - 0.4) * 0.6 # Stronger increase if average risk activation > 0.4

    elif value_label == "Effizienz":
        # Boosted by Meta Cognitio activation (reflects focus on optimization/control)
        meta_act = module_activations.get("Meta Cognitio", 0.5)
        adjustment += (meta_act - 0.5) * 0.4
        # Could also be negatively impacted by high Creativus activation? (Exploration vs Exploitation)
        creativus_act = module_activations.get("Cortex Creativus", 0.5)
        adjustment -= (creativus_act - 0.6) * 0.1 # Slight decrease if creativity is very high

    elif value_label == "Ethik":
        # Boosted by categories explicitly mentioning 'ethik' or 'moral'
        ethik_acts = [act for label, act in category_activations.items() if "ethik" in label.lower() or "moral" in label.lower()]
        avg_ethik_act = np.mean(ethik_acts) if ethik_acts else 0.0
        adjustment += (avg_ethik_act - 0.3) * 0.5 # Increase if ethics category activation > 0.3
        # Influenced by Cortex Criticus evaluations (using the log)
        critic_deque = module_outputs_log.get("Cortex Criticus")
        if critic_deque and isinstance(critic_deque[-1], list) and critic_deque[-1]:
             # Extract valid scores from the last batch of evaluations
             scores = [e.get('score') for e in critic_deque[-1]
                       if isinstance(e, dict) and 'score' in e and isinstance(e.get('score'), (int, float, np.number)) and not np.isnan(e.get('score'))]
             if scores:
                 avg_score = np.mean(scores)
                 # Low average critic score might slightly decrease Ethik if Sicherheit is also low
                 sicherheit_node = value_dict.get("Sicherheit")
                 sicherheit_activation = float(sicherheit_node.activation) if isinstance(sicherheit_node, ValueNode) else 0.5
                 if avg_score < 0.35 and sicherheit_activation < 0.45:
                     adjustment -= 0.03 # Small penalty for low scores when safety is not prioritized

    elif value_label == "Neugier":
        # Boosted by emotional state (moderate arousal, non-negative pleasure)
        pleasure = CURRENT_EMOTION_STATE.get('pleasure', 0.0)
        arousal = CURRENT_EMOTION_STATE.get('arousal', 0.0)
        if arousal > 0.25 and pleasure > -0.15:
            adjustment += 0.05 * arousal # More curious when aroused and not unhappy
        # Boosted by high Innovation value (classical activation)
        innov_node = value_dict.get("Innovation")
        innov_activation = float(innov_node.activation) if isinstance(innov_node, ValueNode) else 0.5
        if innov_activation > 0.7:
             adjustment += (innov_activation - 0.7) * 0.06 # Increase if Innovation value is high

    # Apply global update rate and clip the adjustment to prevent drastic changes
    max_adjustment_step = 0.04 # Limit change per epoch
    final_adjustment = float(np.clip(float(adjustment) * VALUE_UPDATE_RATE, -max_adjustment_step, max_adjustment_step))

    return final_adjustment

def apply_reinforcement(all_nodes: List[Node], module_outputs: Dict[str, deque]):
    """Applies reinforcement to connections based on positive emotion and critic evaluation."""
    reward_signal = 0.0
    reinforced_connections = 0

    # Calculate reward based on pleasure level
    pleasure = CURRENT_EMOTION_STATE.get('pleasure', 0.0)
    if pleasure > REINFORCEMENT_PLEASURE_THRESHOLD:
        # Scale reward based on how much pleasure exceeds the threshold
        reward_signal += (pleasure - REINFORCEMENT_PLEASURE_THRESHOLD) / (1.0 - REINFORCEMENT_PLEASURE_THRESHOLD) * 0.5 # Pleasure contribution

    # Calculate reward based on critic evaluations
    critic_evals_deque = module_outputs.get("Cortex Criticus")
    if critic_evals_deque and isinstance(critic_evals_deque[-1], list) and critic_evals_deque[-1]:
        # Get valid scores from the last batch
        scores = [e.get('score', 0.0) for e in critic_evals_deque[-1]
                  if isinstance(e, dict) and isinstance(e.get('score'), (float, np.number))]
        if scores:
            avg_score = np.mean(scores)
            # Add reward if average score exceeds threshold
            if avg_score > REINFORCEMENT_CRITIC_THRESHOLD:
                reward_signal += (avg_score - REINFORCEMENT_CRITIC_THRESHOLD) / (1.0 - REINFORCEMENT_CRITIC_THRESHOLD) * 0.5 # Critic contribution

    # Apply reinforcement if reward signal is significant
    if reward_signal > 0.05: # Only apply if there's a noticeable positive signal
        effective_reinforcement = float(REINFORCEMENT_FACTOR) * reward_signal
        # Strengthen connections between co-active nodes
        for node in all_nodes:
            # Check source node activation (quantum or classical)
            node_act_raw = getattr(node, 'activation', 0.0)
            node_act = float(node_act_raw) if isinstance(node_act_raw, (float, np.number)) and not np.isnan(node_act_raw) else 0.0

            if hasattr(node, 'connections') and node_act > 0.4: # Source must be reasonably active
                for conn in node.connections:
                    target_node = conn.target_node
                    if target_node:
                        # Check target node activation
                        target_act_raw = getattr(target_node, 'activation', 0.0)
                        target_act = float(target_act_raw) if isinstance(target_act_raw, (float, np.number)) and not np.isnan(target_act_raw) else 0.0

                        if target_act > 0.4: # Target must also be reasonably active
                            # Calculate weight increase based on joint activation and reward
                            delta_weight = effective_reinforcement * node_act * target_act
                            conn.weight = float(np.clip(float(conn.weight) + delta_weight, 0.0, 1.0)) # Apply and clip
                            reinforced_connections += 1

    # Optional: Log reinforcement application
    # if reinforced_connections > 0:
    #    print(f"[Reinforcement] Reward Signal: {reward_signal:.3f} -> {reinforced_connections} connections reinforced.")

def consolidate_memories(all_nodes: List[Node], pm_manager: PersistentMemoryManager, epoch: int, status_callback: Callable):
    """Stores highly relevant long-term memories (nodes) in the persistent database."""
    consolidated_count = 0
    if pm_manager is None:
        # status_callback("WARNUNG: Persistent Memory Manager nicht verfügbar für Konsolidierung.") # Maybe too verbose
        return

    # Iterate through all nodes to find candidates for consolidation
    for node in all_nodes:
        # Check if it's a MemoryNode in the long-term state and sufficiently active
        if isinstance(node, MemoryNode) and node.memory_type == "long_term" and hasattr(node, 'activation'):
             activation_raw = node.activation
             # Ensure activation is valid float
             activation_val = float(activation_raw) if isinstance(activation_raw, (float, np.number)) and not np.isnan(activation_raw) else 0.0

             if activation_val > MEMORY_RELEVANCE_THRESHOLD:
                 # Create a unique key for the memory entry
                 memory_key = f"category_{node.label}"
                 # Prepare the content to be stored (as a dictionary)
                 content = {
                     "label": node.label,
                     "activation_at_consolidation": round(activation_val, 4),
                     "type": "long_term_category",
                     "consolidation_epoch": epoch + 1, # Record epoch of consolidation
                     "neuron_type": getattr(node, 'neuron_type', 'unknown')
                 }
                 # Store quantum parameters if the node has them
                 if hasattr(node, 'q_system') and node.q_system:
                      try:
                          content["q_params"] = node.q_system.get_params().tolist() # Store as list
                      except AttributeError:
                          content["q_params"] = None # Indicate missing params

                 # Use current activation as relevance score for storage
                 relevance = activation_val

                 # Store the memory using the persistent memory manager
                 try:
                     pm_manager.store_memory(memory_key, content, relevance)
                     consolidated_count += 1
                 except Exception as store_err:
                     status_callback(f"FEHLER beim Speichern von MemKey '{memory_key}': {store_err}")

    # Log consolidation activity if any nodes were stored
    if consolidated_count > 0:
         status_callback(f"E{epoch+1}: {consolidated_count} Langzeit-Knoten in DB konsolidiert.")


# --- Interpretation & Report Generation ---
def interpret_epoch_state(epoch: int, category_nodes: List[MemoryNode], module_nodes: List[Node],
                          module_outputs: Dict[str, deque], activation_history_local: Dict[str, deque],
                          value_nodes: List[ValueNode]) -> Dict[str, Any]:
    """Analyzes the network state at the end of an epoch and returns a summary dictionary."""
    interpretation = {'epoch': epoch + 1} # Start epoch count from 1 for reporting

    # --- Category Analysis ---
    # Filter for valid category nodes with numeric activations
    valid_category_nodes = [
        n for n in category_nodes
        if hasattr(n, 'activation') and isinstance(getattr(n, 'activation', None), (float, np.number))
        and not np.isnan(n.activation) and hasattr(n, 'label')
    ]
    if valid_category_nodes:
        # Sort categories by activation (descending)
        sorted_cats = sorted(valid_category_nodes, key=lambda n: float(n.activation), reverse=True)
        # Identify dominant category and its activation
        interpretation['dominant_category'] = getattr(sorted_cats[0], 'label', 'Unbekannt')
        interpretation['dominant_activation'] = round(float(sorted_cats[0].activation), 4)
        # Provide ranking of top categories
        interpretation['category_ranking'] = [
            (getattr(n, 'label', 'Unbekannt'), round(float(n.activation), 4))
            for n in sorted_cats[:5] # Top 5
        ]
        # Calculate average and standard deviation of category activations
        float_activations = [float(n.activation) for n in valid_category_nodes]
        interpretation['avg_category_activation'] = round(np.mean(float_activations), 4) if float_activations else 0.0
        if len(valid_category_nodes) > 1:
            interpretation['std_category_activation'] = round(np.std(float_activations), 4)
        else:
            interpretation['std_category_activation'] = 0.0
    else:
        # Default values if no valid categories found
        interpretation['dominant_category'] = 'N/A'
        interpretation['dominant_activation'] = 0.0
        interpretation['category_ranking'] = []
        interpretation['avg_category_activation'] = 0.0
        interpretation['std_category_activation'] = 0.0

    # --- Module Activation ---
    module_acts = {}
    for i, m_node in enumerate(module_nodes):
        if hasattr(m_node, 'activation') and hasattr(m_node, 'label'):
             act_raw = getattr(m_node, 'activation', 0.0)
             act_val = float(act_raw) if isinstance(act_raw, (float, np.number)) and not np.isnan(act_raw) else 0.0
             module_acts[m_node.label] = round(act_val, 4)
    interpretation['module_activations'] = module_acts

    # --- Value Node Activation ---
    interpretation['value_node_activations'] = {
        v.label: round(float(v.activation), 4) for v in value_nodes if hasattr(v, 'label')
    } # Values use classical activation

    # --- Emotion State ---
    interpretation['emotion_state'] = CURRENT_EMOTION_STATE.copy()

    # --- Meta-Cognition ---
    # Find the MetaCognitio module instance
    meta_cog_module = next((m for m in module_nodes if isinstance(m, MetaCognitio)), None)
    # Get the latest reflection log entry if available
    interpretation['last_reflection'] = None
    if meta_cog_module and hasattr(meta_cog_module, 'reflection_log') and meta_cog_module.reflection_log:
        interpretation['last_reflection'] = meta_cog_module.reflection_log[-1]

    return interpretation

def generate_final_report(
    category_nodes: List[MemoryNode], module_nodes: List[Node], value_nodes: List[ValueNode],
    original_data: pd.DataFrame, interpretation_log: List[Dict]
) -> Tuple[str, Dict[str, Any]]:
    """Generates a final textual report and structured results dictionary based on the simulation outcome."""
    print(f"\n--- Generiere finalen Bericht ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2) ---")
    report_lines = [f"**NeuroPersona Analysebericht ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2 - EXPERIMENTELL)**\n"]
    # Initialize structured results dictionary
    structured_results = {
        "dominant_category": "N/A", "dominant_activation": 0.0, "category_ranking": [],
        "module_activations": {}, "value_node_activations": {}, "emotion_state": {},
        "final_recommendation": "Keine klare Tendenz.", # Default recommendation
        "frequent_dominant_category": None, "stability_assessment": "Unbekannt",
        "reflection_summary": []
    }
    # Activation thresholds for interpreting significance (might need tuning)
    threshold_high = 0.65
    threshold_low = 0.35

    # --- Final Category Analysis ---
    report_lines.append("**Finale Netzwerk-Tendenzen (Kategorien - Multi-Qubit Aktivierung):**")
    valid_category_nodes = [
        n for n in category_nodes if isinstance(n, MemoryNode)
        and hasattr(n, 'activation') and isinstance(n.activation, (float, np.number))
        # Hinzugefügt: Filtere NaN-Werte explizit heraus
        and not np.isnan(n.activation) and hasattr(n, 'label')
    ]
    if not valid_category_nodes:
        report_lines.append("- Keine aktiven Kategorieknoten am Ende der Simulation.")
    else:
        # Sort categories by final activation
        sorted_categories = sorted(valid_category_nodes, key=lambda n: float(n.activation), reverse=True)
        report_lines.append("  Aktivste Kategorien (Top 5):")
        category_ranking_data = []
        for i, node in enumerate(sorted_categories[:5]):
            label = getattr(node, 'label', 'Unbekannt')
            # Stelle sicher, dass Aktivierung ein Float ist vor dem Runden
            act = round(float(node.activation), 3)
            report_lines.append(f"  {i+1}. {label}: {act}")
            category_ranking_data.append((label, act))
        structured_results["category_ranking"] = category_ranking_data
        # Store dominant category info
        # Stelle sicher, dass Aktivierung ein Float ist vor dem Runden
        structured_results["dominant_category"] = sorted_categories[0].label
        structured_results["dominant_activation"] = round(float(sorted_categories[0].activation), 4)

    # --- Analysis over Time (from interpretation log) ---
    if interpretation_log:
        # Find most frequent dominant category over the simulation
        dom_cats_time = [e.get('dominant_category') for e in interpretation_log if e.get('dominant_category') != 'N/A']
        if dom_cats_time:
            try:
                most_freq, freq_count = Counter(dom_cats_time).most_common(1)[0]
                report_lines.append(f"- Verlauf: '{most_freq}' war am häufigsten dominant ({freq_count}/{len(interpretation_log)} Epochen).")
                structured_results["frequent_dominant_category"] = most_freq
            except IndexError: pass # Handle case where counter is empty

            # Assess stability based on recent dominant categories
            last_n = min(len(dom_cats_time), max(5, len(interpretation_log) // 4)) # Look at last 5 or quarter of epochs
            recent_doms = dom_cats_time[-last_n:]
            unique_recent = len(set(recent_doms))
            if unique_recent == 1: stability = f"Stabil ('{recent_doms[0]}', letzte {last_n} Ep.)"
            elif unique_recent <= 2: stability = f"Leicht wechselnd ({unique_recent} Kategorien, letzte {last_n} Ep.)"
            else: stability = f"Instabil ({unique_recent} Kategorien, letzte {last_n} Ep.)"
            structured_results["stability_assessment"] = stability
            report_lines.append(f"- Stabilität (Letzte {last_n} Ep.): {stability}")
        else:
            report_lines.append("- Verlauf: Keine dominante Kategorie während der Simulation aufgezeichnet.")
            structured_results["stability_assessment"] = "Keine Daten für Stabilität"
    else:
        report_lines.append("- Verlauf: Kein Interpretationsprotokoll verfügbar.")

    # --- Final Module State ---
    report_lines.append("\n**Finaler Zustand kognitiver Module (Multi-Qubit Aktivierung):**")
    module_activation_data = {}
    for i, m_node in enumerate(module_nodes):
         if hasattr(m_node, 'activation') and hasattr(m_node, 'label'):
             act_raw = getattr(m_node, 'activation', 0.0)
             # Stelle sicher, dass Aktivierung ein Float ist und nicht NaN
             act_val = float(act_raw) if isinstance(act_raw, (float, np.number)) and not np.isnan(act_raw) else 0.0
             module_activation_data[m_node.label] = round(act_val, 3)
    # Sort modules by activation for reporting
    sorted_modules = sorted(module_activation_data.items(), key=lambda item: item[1], reverse=True)
    for label, activation in sorted_modules:
        report_lines.append(f"- {label}: {activation}")
    structured_results["module_activations"] = module_activation_data

    # --- Final Value State ---
    report_lines.append("\n**Aktive Wertvorstellungen (Klassisch):**")
    value_activation_data = {
        v.label: round(float(v.activation), 3)
        for v in value_nodes if hasattr(v, 'label') and hasattr(v, 'activation')
        # Stelle sicher, dass Aktivierung ein Float ist und nicht NaN
        and isinstance(v.activation, (float, np.number)) and not np.isnan(v.activation)
    }
    sorted_values = sorted(value_activation_data.items(), key=lambda item: item[1], reverse=True)
    for label, activation in sorted_values:
        report_lines.append(f"- {label}: {activation}")
    structured_results["value_node_activations"] = value_activation_data

    # --- Final Emotional State ---
    report_lines.append("\n**Finale Emotionale Grundstimmung (PAD - Klassisch):**")
    limbus_module = next((m for m in module_nodes if isinstance(m, LimbusAffektus)), None)
    final_emotion_state = limbus_module.emotion_state if limbus_module else CURRENT_EMOTION_STATE
    for dim, value in final_emotion_state.items():
        report_lines.append(f"- {dim.capitalize()}: {value:.3f}")
    structured_results["emotion_state"] = final_emotion_state

    # --- Meta-Cognitive Reflections ---
    report_lines.append("\n**Meta-Kognitive Reflexion (Letzte Einträge):**")
    meta_cog_module = next((m for m in module_nodes if isinstance(m, MetaCognitio)), None)
    reflection_summary = []
    if meta_cog_module and hasattr(meta_cog_module, 'reflection_log') and meta_cog_module.reflection_log:
        logged_reflections = list(meta_cog_module.reflection_log)
        # Report the last few reflections
        for i, entry in enumerate(reversed(logged_reflections)):
            if i >= 5: break # Limit to last 5
            msg = entry.get('message', '(Keine Nachricht)')
            epoch_num = entry.get('epoch', '?')
            report_lines.append(f"- E{epoch_num}: {msg}")
            reflection_summary.append(entry) # Add full entry to structured results
    if not reflection_summary:
        report_lines.append("- Keine besonderen Vorkommnisse oder Reflexionen protokolliert.")
    structured_results["reflection_summary"] = reflection_summary

    # --- Final Overall Assessment (Set previously, just report it) ---
    final_assessment_text = structured_results.get("final_recommendation", "Keine klare Tendenz.")
    report_lines.append(f"\n**Gesamteinschätzung:** {final_assessment_text}")

    # Combine report lines into a single string
    final_report_text = "\n".join(report_lines)
    print(final_report_text) # Print to console as well

    return final_report_text, structured_results

# --- Save Network State ---
def save_final_network_state(nodes_list: List[Node], emotion_state: Dict[str, float],
                               value_nodes: List[ValueNode], meta_cog_module: Optional[MetaCognitio],
                               filename: str = MODEL_FILENAME):
    """Saves the final state of the network (nodes, connections, q_params, etc.) to a JSON file."""
    # Prepare data structure for saving
    model_data = {
        "version": f"quantum_node_mq{NUM_QUBITS_PER_NODE}_v2_save_1",
        "nodes": [],
        "connections": [],
        "emotion_state": {},
        "value_node_states": {},
        "reflection_log": []
    }

    # Filter for valid nodes with labels
    valid_nodes = [node for node in nodes_list if isinstance(node, Node) and hasattr(node, 'label')]
    node_labels_set = {node.label for node in valid_nodes} # Set of labels for quick checking
    print(f"Speichere Zustand von {len(valid_nodes)} Knoten in {filename}...")

    saved_node_count = 0
    saved_conn_count = 0
    # --- Save Node Information ---
    for node in valid_nodes:
         # Get activation, ensure it's float and not NaN
         activation_save_raw = getattr(node, 'activation', 0.0)
         activation_save = float(activation_save_raw) if isinstance(activation_save_raw, (float, np.number)) and not np.isnan(activation_save_raw) else 0.0

         node_info = {
             "label": node.label,
             "class": type(node).__name__,
             "activation": round(activation_save, 5),
             "neuron_type": getattr(node, 'neuron_type', "excitatory"),
         }
         # Add specific info for MemoryNodes
         if isinstance(node, MemoryNode):
             node_info.update({
                 "memory_type": getattr(node, 'memory_type', 'short_term'),
                 "time_in_memory": getattr(node, 'time_in_memory', 0)
             })
         # --- Save Quantum Parameters ---
         if hasattr(node, 'q_system') and node.q_system is not None:
              try:
                  # Convert numpy array to list for JSON serialization
                  # Filter out potential NaN/Inf values before converting to list
                  q_params_raw = node.q_system.get_params()
                  q_params_safe = [float(p) if np.isfinite(p) else 0.0 for p in q_params_raw] # Replace non-finite with 0.0
                  node_info["q_params"] = q_params_safe
              except Exception as e:
                  print(f"Warnung: Konnte QParams für Knoten {node.label} nicht speichern: {e}")
                  node_info["q_params"] = None # Indicate params couldn't be saved
         # --- End Save Quantum Parameters ---

         model_data["nodes"].append(node_info)
         saved_node_count += 1

         # --- Save Connection Information ---
         if hasattr(node, 'connections'):
             for conn in node.connections:
                 target_node = conn.target_node
                 # Ensure target exists, has a label, and is part of the saved network
                 if target_node and hasattr(target_node, 'label') and target_node.label in node_labels_set:
                     # Ensure weight is float and not NaN
                     weight_save_raw = getattr(conn, 'weight', 0.0)
                     weight_save = float(weight_save_raw) if isinstance(weight_save_raw, (float, np.number)) and not np.isnan(weight_save_raw) else 0.0

                     # Optionally, only save connections above a minimal threshold to reduce file size
                     if weight_save > PRUNING_THRESHOLD * 0.5:
                         model_data["connections"].append({
                             "source": node.label,
                             "target": target_node.label,
                             "weight": round(weight_save, 5)
                         })
                         saved_conn_count += 1

    # --- Save Other Global States ---
    model_data["emotion_state"] = emotion_state
    # Save classical value node activations separately for clarity, ensure finite
    value_states_safe = {}
    for v in value_nodes:
        if hasattr(v, 'label') and hasattr(v, 'activation'):
             act_raw = v.activation
             act_safe = float(act_raw) if isinstance(act_raw, (float, np.number)) and np.isfinite(act_raw) else 0.0
             value_states_safe[v.label] = round(act_safe, 5)
    model_data["value_node_states"] = value_states_safe

    # Save reflection log if MetaCognitio module exists
    if meta_cog_module and hasattr(meta_cog_module, 'reflection_log'):
        # Convert deque to list for saving
        try:
            model_data["reflection_log"] = list(meta_cog_module.reflection_log)
        except Exception as log_err:
            print(f"Warnung: Konnte Reflection Log nicht speichern: {log_err}")
            model_data["reflection_log"] = []


    # --- Write to File ---
    try:
        with open(filename, "w", encoding='utf-8') as file:
            # Use indent for readability, ensure_ascii=False for potential special characters in labels
            # Add option to handle non-finite floats in json.dump if necessary, though pre-filtering is better
            json.dump(model_data, file, indent=2, ensure_ascii=False) # allow_nan=False is default
        print(f"Netzwerkzustand erfolgreich gespeichert: {filename}")
        print(f"  ({saved_node_count} Knoten, {saved_conn_count} Verbindungen)")
    except (IOError, TypeError) as e:
        print(f"FEHLER beim Speichern des Netzwerkzustands in '{filename}': {e}")
    except Exception as e:
        print(f"Unbekannter FEHLER beim Speichern des Zustands: {e}")
        traceback.print_exc()


# --- Plotting Functions ---
def filter_module_history(activation_history: Dict[str, deque], module_labels: List[str]) -> Dict[str, deque]:
    """Filters the activation history to include only specified module labels."""
    return {label: history for label, history in activation_history.items()
            if label in module_labels and history}

def plot_activation_and_weights(activation_history: Dict[str, List[float]],
                                weights_history: Dict[str, List[float]],
                                q_param_history: Optional[Dict[str, List[float]]] = None,
                                filename: str = "plot_act_weights.png") -> Optional[plt.Figure]:
    """Plots the evolution of node activations, classical weights, and optionally the first quantum parameter."""
    print("Erstelle Plot: Aktivierungs-, Gewichts- & QParam-Entwicklung...")
    if not activation_history and not weights_history:
        print("Übersprungen: Keine Aktivierungs- oder Gewichtsdaten zum Plotten.")
        return None

    # Determine number of subplots needed (2 or 3)
    # Korrigiert: Nur 3 Plots, wenn q_param_history tatsächlich Daten enthält
    has_q_param_data = q_param_history and any(v for v in q_param_history.values())
    num_plots = 3 if has_q_param_data else 2

    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 7), sharey=False)
    axes = np.atleast_1d(axes) # Ensure axes is always iterable

    max_lines_to_plot = 20 # Limit lines per plot for readability

    # --- Plot 1: Node Activations (Multi-Qubit Normalized Hamming Weight) ---
    ax1 = axes[0]
    # Validate and filter activation history
    valid_act_hist = {k: [float(a) for a in v if isinstance(a, (float, np.number)) and not np.isnan(a)]
                      for k, v in activation_history.items() if isinstance(v, (list, deque)) and v}
    valid_act_hist = {k: v for k, v in valid_act_hist.items() if v} # Remove empty lists
    plot_count_act = 0
    if valid_act_hist:
        # Sort keys by standard deviation (most dynamic first)
        sorted_act_keys = sorted(valid_act_hist.keys(),
                                 key=lambda k: np.std(valid_act_hist[k]) if len(valid_act_hist[k]) > 1 else 0,
                                 reverse=True)
        for label in sorted_act_keys:
            if plot_count_act >= max_lines_to_plot: break
            activations = valid_act_hist[label]
            if len(activations) > 1: # Need at least 2 points to plot a line
                ax1.plot(range(1, len(activations) + 1), activations, label=label, alpha=0.75, linewidth=1.5)
                plot_count_act += 1
    ax1.set_title(f"Aktivierung (MQ, Top {plot_count_act} dyn.)")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Aktivierung (Norm. Hamming W.)")
    ax1.set_ylim(0, 1.05) # Activation is normalized between 0 and 1
    ax1.grid(True, alpha=0.5)
    if plot_count_act > 0:
        ax1.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1.03, 0.5))

    # --- Plot 2: Classical Connection Weights ---
    ax2 = axes[1]
    # Validate and filter weight history
    valid_weight_hist = {k: [float(w) for w in v if isinstance(w, (float, np.number)) and not np.isnan(w)]
                         for k, v in weights_history.items() if isinstance(v, (list, deque)) and v}
    valid_weight_hist = {k: v for k, v in valid_weight_hist.items() if v} # Remove empty lists
    plot_count_weights = 0
    if valid_weight_hist:
        # Sort keys by standard deviation
        sorted_weights_keys = sorted(valid_weight_hist.keys(),
                                     key=lambda k: np.std(valid_weight_hist[k]) if len(valid_weight_hist[k]) > 1 else 0,
                                     reverse=True)
        for label in sorted_weights_keys:
            if plot_count_weights >= max_lines_to_plot: break
            weights = valid_weight_hist[label]
            if len(weights) > 1:
                ax2.plot(range(1, len(weights) + 1), weights, label=label, alpha=0.65, linewidth=1.0)
                plot_count_weights += 1
    ax2.set_title(f"Klass. Gewichte (Top {plot_count_weights} dyn.)")
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Gewicht")
    ax2.set_ylim(0, 1.05) # Weights are typically clipped between 0 and 1
    ax2.grid(True, alpha=0.5)
    if plot_count_weights > 0:
        ax2.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1.03, 0.5))

    # --- Plot 3: First Quantum Parameter (Optional) ---
    if has_q_param_data and num_plots == 3: # Prüfe explizit ob Daten vorhanden und 3 Plots erwartet werden
        ax3 = axes[2]
        # Validate and filter q_param history
        valid_qparam_hist = {k: [float(p) for p in v if isinstance(p, (float, np.number)) and not np.isnan(p)]
                             for k, v in q_param_history.items() if isinstance(v, (list, deque)) and v}
        valid_qparam_hist = {k: v for k, v in valid_qparam_hist.items() if v} # Remove empty lists
        plot_count_qparam = 0
        if valid_qparam_hist:
            # Sort keys by standard deviation
            sorted_qparam_keys = sorted(valid_qparam_hist.keys(),
                                       key=lambda k: np.std(valid_qparam_hist[k]) if len(valid_qparam_hist[k]) > 1 else 0,
                                       reverse=True)
            for label in sorted_qparam_keys:
                 if plot_count_qparam >= max_lines_to_plot: break
                 qparams = valid_qparam_hist[label]
                 if len(qparams) > 1:
                     ax3.plot(range(1, len(qparams) + 1), qparams, label=label, alpha=0.7, linewidth=1.2)
                     plot_count_qparam += 1
        ax3.set_title(f"Quanten-Parameter (P0, Top {plot_count_qparam} dyn.)")
        ax3.set_xlabel("Epoche")
        ax3.set_ylabel("Parameterwert (z.B. Winkel Rad)")
        # Set y-limits based on typical parameter range (e.g., 0 to 2*pi)
        ax3.set_ylim(bottom=0, top=2*np.pi + 0.1)
        ax3.grid(True, alpha=0.5)
        if plot_count_qparam > 0:
            ax3.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1.03, 0.5))
    elif num_plots == 3: # Case where subplot exists but no data
        axes[2].set_title("Quanten-Parameter (Keine Daten)")
        axes[2].text(0.5, 0.5, "Keine QParam-Historie verfügbar", horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)


    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legends
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        print(f"Plot gespeichert: {filepath}")
        return fig
    except Exception as e:
        print(f"FEHLER beim Speichern des Plots '{filepath}': {e}")
        return None
    finally:
        plt.close(fig) # Close the figure to free memory

def plot_dynamics(activation_history: Dict[str, List[float]],
                    weights_history: Dict[str, List[float]],
                    filename: str = "plot_dynamics.png") -> Optional[plt.Figure]:
    """Plots overall network dynamics: average activation, weights, activity count, dominance."""
    print("Erstelle Plot: Netzwerk-Dynamiken...")
    if not activation_history and not weights_history:
        print("Übersprungen: Keine Aktivierungs- oder Gewichtsdaten zum Plotten.")
        return None

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axs = axs.flatten() # Flatten axes array for easy indexing

    # --- Data Validation and Preparation ---
    # Filter and convert histories to lists of floats
    valid_act_hist = {k: [float(a) for a in v if isinstance(a, (float, np.number)) and not np.isnan(a)]
                      for k, v in activation_history.items() if isinstance(v, (list, deque)) and v}
    valid_weight_hist = {k: [float(w) for w in v if isinstance(w, (float, np.number)) and not np.isnan(w)]
                         for k, v in weights_history.items() if isinstance(v, (list, deque)) and v}
    # Remove keys with empty lists after filtering
    valid_act_hist = {k: v for k, v in valid_act_hist.items() if v}
    valid_weight_hist = {k: v for k, v in valid_weight_hist.items() if v}

    # Determine the number of epochs based on the longest history
    num_epochs = 0
    if valid_act_hist: num_epochs = max(num_epochs, max(len(h) for h in valid_act_hist.values()))
    if valid_weight_hist: num_epochs = max(num_epochs, max(len(h) for h in valid_weight_hist.values()))

    if num_epochs == 0:
        print("Übersprungen: Keine gültigen Datenpunkte in Historien gefunden.")
        plt.close(fig)
        return None
    epochs_range = np.arange(1, num_epochs + 1)

    # Helper function to calculate mean and std dev over epochs
    def get_stats(hist_dict, ep_count):
        means, stds = [], []
        for i in range(ep_count):
            # Get values for epoch i from all histories that have data for this epoch
            vals_at_i = [h[i] for h in hist_dict.values() if len(h) > i]
            means.append(np.mean(vals_at_i) if vals_at_i else np.nan)
            stds.append(np.std(vals_at_i) if len(vals_at_i) > 1 else 0.0) # Std dev is 0 for single point
        return np.array(means), np.array(stds)

    # --- Plot 1: Average Network Activation (Multi-Qubit) ---
    avg_act, std_act = get_stats(valid_act_hist, num_epochs)
    valid_idx_act = ~np.isnan(avg_act) # Indices where average activation is not NaN
    if np.any(valid_idx_act):
        axs[0].plot(epochs_range[valid_idx_act], avg_act[valid_idx_act], label="Avg. Aktivierung (MQ)", color='dodgerblue')
        # Add shaded area for standard deviation
        axs[0].fill_between(epochs_range[valid_idx_act],
                            np.maximum(0, avg_act[valid_idx_act] - std_act[valid_idx_act]), # Lower bound, clipped at 0
                            np.minimum(1, avg_act[valid_idx_act] + std_act[valid_idx_act]), # Upper bound, clipped at 1
                            alpha=0.25, color='dodgerblue', label="StdAbw")
        axs[0].legend(fontsize='small')
    axs[0].set_title("Netzwerkaktivierung (Durchschnitt MQ)")
    axs[0].set_ylabel("Aktivierung (Norm. Hamming W.)")
    axs[0].set_ylim(0, 1.05)
    axs[0].grid(True, alpha=0.5)

    # --- Plot 2: Average Weight Development (Classical) ---
    avg_w, std_w = get_stats(valid_weight_hist, num_epochs)
    valid_idx_w = ~np.isnan(avg_w) # Indices where average weight is not NaN
    if np.any(valid_idx_w):
        # Determine max weight for y-axis limit dynamically
        # Hinzugefügt: Behandlung von leerem Array nach NaN-Filterung
        if np.any(valid_idx_w):
            max_w_limit = np.nanmax(avg_w[valid_idx_w] + std_w[valid_idx_w])
            max_w_limit = min(max_w_limit, 1.5) # Cap limit
        else:
            max_w_limit = 1.0 # Default if no valid weights

        axs[1].plot(epochs_range[valid_idx_w], avg_w[valid_idx_w], label="Avg. Gewicht (Klass.)", color='forestgreen')
        axs[1].fill_between(epochs_range[valid_idx_w],
                            np.maximum(0, avg_w[valid_idx_w] - std_w[valid_idx_w]), # Lower bound, clipped at 0
                            np.minimum(max_w_limit, avg_w[valid_idx_w] + std_w[valid_idx_w]), # Upper bound, clipped
                            alpha=0.25, color='forestgreen', label="StdAbw")
        axs[1].legend(fontsize='small')
        axs[1].set_ylim(0, max(1.05, max_w_limit)) # Ensure ylim includes at least 0-1
    else:
        axs[1].set_ylim(0, 1.05)
    axs[1].set_title("Gewichtsentwicklung (Durchschnitt Klass.)")
    axs[1].set_ylabel("Gewicht")
    axs[1].grid(True, alpha=0.5)

    # --- Plot 3: Network Activity (Number of Active Nodes) ---
    active_nodes_count = []
    total_nodes_with_history = len(valid_act_hist)
    activation_threshold_for_plot = 0.5 # Threshold to consider a node "active" for this plot
    if total_nodes_with_history > 0:
        for i in range(num_epochs):
            # Count nodes with activation above threshold at epoch i
            count_at_i = sum(1 for h in valid_act_hist.values() if len(h) > i and h[i] > activation_threshold_for_plot)
            active_nodes_count.append(count_at_i)
        if active_nodes_count:
            axs[2].plot(epochs_range, active_nodes_count, label=f"Aktive (> {activation_threshold_for_plot:.2f})", color='crimson')
            axs[2].set_ylim(0, total_nodes_with_history * 1.05) # Y-limit based on total nodes
            axs[2].legend(fontsize='small')
    axs[2].set_title("Netzwerk-Aktivität")
    axs[2].set_ylabel("Anzahl Knoten")
    axs[2].grid(True, alpha=0.5)

    # --- Plot 4: Dominance (Activation of the Strongest Category Node) ---
    dominant_activation_over_time = []
    # Define labels of nodes that are *not* categories
    module_labels = {m.label for m in [CortexCreativus(), SimulatrixNeuralis(), CortexCriticus(), LimbusAffektus(), MetaCognitio(), CortexSocialis()] if hasattr(m, 'label')}
    value_labels = set(DEFAULT_VALUES.keys())
    input_labels_prefix = "Q_"

    # Filter activation history to include only potential category nodes
    category_history = {
        k: h for k, h in valid_act_hist.items()
        if not k.startswith(input_labels_prefix) and k not in module_labels and k not in value_labels
    }

    if category_history:
        for i in range(num_epochs):
            # Get activations of category nodes at epoch i
            acts_at_i = [h[i] for h in category_history.values() if len(h) > i]
            # Find the maximum activation among categories
            dominant_activation_over_time.append(max(acts_at_i) if acts_at_i else np.nan)

        dom_act_np = np.array(dominant_activation_over_time)
        valid_idx_dom = ~np.isnan(dom_act_np) # Indices where max activation is not NaN
        if np.any(valid_idx_dom):
            axs[3].plot(epochs_range[valid_idx_dom], dom_act_np[valid_idx_dom], label="Max. Kat.-Aktivierung", color='darkviolet')
            axs[3].legend(fontsize='small')

    axs[3].set_title("Dominanz Stärkste Kategorie")
    axs[3].set_ylabel("Max. Aktivierung (MQ)")
    axs[3].set_ylim(0, 1.05)
    axs[3].grid(True, alpha=0.5)

    # Set common X labels
    axs[2].set_xlabel("Epoche")
    axs[3].set_xlabel("Epoche")
    plt.tight_layout()

    # Save the plot
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        print(f"Plot gespeichert: {filepath}")
        return fig
    except Exception as e:
        print(f"FEHLER beim Speichern des Plots '{filepath}': {e}")
        return None
    finally:
        plt.close(fig) # Close the figure to free memory


def plot_module_activation_comparison(module_activation_history: Dict[str, deque],
                                       filename: str = "plot_modules.png") -> Optional[plt.Figure]:
    """Plots the activation trends of different cognitive modules over time."""
    print("Erstelle Plot: Modul-Aktivierungsvergleich...")
    if not module_activation_history:
        print("Übersprungen: Keine Modul-Aktivierungsdaten zum Plotten.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_something = False
    # Plot activation for each module
    for label, activations_deque in module_activation_history.items():
        # Convert deque to list of valid floats
        activations = [float(a) for a in list(activations_deque) if isinstance(a, (float, np.number)) and not np.isnan(a)]
        if len(activations) > 1: # Need at least 2 points
            ax.plot(range(1, len(activations) + 1), activations, label=label, linewidth=2, alpha=0.8)
            plotted_something = True

    if not plotted_something:
        print("Übersprungen: Keine ausreichenden Modul-Aktivierungsdaten zum Plotten.")
        plt.close(fig)
        return None

    # Configure plot appearance
    ax.set_title(f"Vergleich Aktivierungen Kognitiver Module ({NUM_QUBITS_PER_NODE}-Qubit Knoten)")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Aktivierung (Norm. Hamming W.)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.5)
    # Place legend outside the plot area
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to accommodate legend

    # Save the plot
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        print(f"Plot gespeichert: {filepath}")
        return fig
    except Exception as e:
        print(f"FEHLER beim Speichern des Plots '{filepath}': {e}")
        return None
    finally:
        plt.close(fig) # Close the figure


def plot_network_structure(nodes_list: List[Node], filename: str = "plot_structure_stats.png") -> Optional[plt.Figure]:
    """Plots statistics about the network structure: node types, neuron types, weight distribution."""
    print("Erstelle Plot: Netzwerk-Struktur (Statistiken)...")
    if not nodes_list:
        print("Übersprungen: Keine Knotendaten für Strukturplot.")
        return None

    # Counters and lists for statistics
    node_counts_by_type = Counter()
    neuron_type_counts = Counter()
    connection_weights = []
    total_connections = 0
    # Define module classes for categorization
    module_classes = (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis)
    valid_nodes = [node for node in nodes_list if isinstance(node, Node)] # Ensure we only process Node objects

    # --- Collect Statistics ---
    for node in valid_nodes:
        node_type_label = "Unbekannt" # Default label
        label = getattr(node, 'label', '')

        # Categorize node based on its class and label
        if isinstance(node, module_classes): node_type_label = f"Modul ({label})" if label else "Modul"
        elif isinstance(node, ValueNode): node_type_label = 'Wert (Klassisch)'
        elif isinstance(node, MemoryNode): node_type_label = 'Kategorie (MQ)' # MQ for Multi-Qubit
        elif isinstance(label, str) and label.startswith("Q_"): node_type_label = 'Frage (Input)'
        # Check if it's a generic node with a quantum system
        elif isinstance(node, Node) and hasattr(node, 'q_system') and node.q_system: node_type_label = 'Basis-Knoten (MQ)'
        elif isinstance(node, Node): node_type_label = 'Basis-Knoten (Unbekannt)' # Fallback

        node_counts_by_type[node_type_label] += 1
        neuron_type_counts[getattr(node, 'neuron_type', 'unbekannt')] += 1

        # Collect connection weights (classical)
        if hasattr(node, 'connections'):
            valid_conns = [conn for conn in node.connections
                           if hasattr(conn, 'weight') and isinstance(conn.weight, (float, np.number))
                           # Prüfe explizit auf NaN
                           and not np.isnan(conn.weight)]
            total_connections += len(valid_conns)
            connection_weights.extend([float(conn.weight) for conn in valid_conns])

    # --- Create Plot ---
    num_node_types = len(node_counts_by_type)
    # Adjust figure width based on the number of node types for better label visibility
    fig_width = max(16, 8 + num_node_types * 0.6)
    # Define subplot layout with adjusted width ratios
    fig, axs = plt.subplots(1, 3, figsize=(fig_width, 6), gridspec_kw={'width_ratios': [max(3, num_node_types*0.4), 1.5, 2]})

    # Plot 1: Node Type Distribution
    if node_counts_by_type:
        types = sorted(node_counts_by_type.keys())
        counts = [node_counts_by_type[t] for t in types]
        axs[0].bar(types, counts, color='skyblue', edgecolor='black', linewidth=0.5)
        axs[0].set_title(f'Knotentypen (Gesamt: {len(valid_nodes)})')
        axs[0].set_ylabel('Anzahl')
        # Apply rotation and size using tick_params
        axs[0].tick_params(axis='x', rotation=45, labelsize='small')
        # Set horizontal alignment separately for the x-tick labels
        plt.setp(axs[0].get_xticklabels(), ha="right", rotation_mode="anchor")
        axs[0].grid(True, axis='y', alpha=0.6, linestyle=':')
    else:
        # Display message if no node data
        axs[0].text(0.5, 0.5, 'Keine Knotendaten', ha='center', va='center', transform=axs[0].transAxes)
        axs[0].set_title('Knotentypen (Gesamt: 0)')


    # Plot 2: Neuron Type Distribution
    if neuron_type_counts:
        n_types = sorted(neuron_type_counts.keys())
        n_counts = [neuron_type_counts[nt] for nt in n_types]
        axs[1].bar(n_types, n_counts, color='lightcoral', edgecolor='black', linewidth=0.5)
        axs[1].set_title('Neuronentypen')
        axs[1].set_ylabel('Anzahl')
        axs[1].grid(True, axis='y', alpha=0.6, linestyle=':')
    else:
         # Display message if no neuron data
         axs[1].text(0.5, 0.5, 'Keine Neuronendaten', ha='center', va='center', transform=axs[1].transAxes)
         axs[1].set_title('Neuronentypen')

    # Plot 3: Classical Connection Weight Distribution
    if connection_weights:
        # Filter out potential NaNs just in case (redundant if checked during collection, but safe)
        connection_weights_float = [w for w in connection_weights if isinstance(w, (float, np.number)) and not np.isnan(w)]
        if connection_weights_float:
             axs[2].hist(connection_weights_float, bins=30, color='lightgreen', edgecolor='black', alpha=0.8, linewidth=0.5)
             # Add statistics (mean, std dev)
             avg_w = np.mean(connection_weights_float)
             std_w = np.std(connection_weights_float)
             axs[2].axvline(avg_w, color='red', linestyle='--', linewidth=1.5, label=f'Avg: {avg_w:.3f}')
             axs[2].set_title(f'Klassische Gewichte (N={total_connections})')
             axs[2].set_xlabel('Gewicht')
             axs[2].set_ylabel('Häufigkeit')
             axs[2].legend(fontsize='small')
             axs[2].grid(True, axis='y', alpha=0.6, linestyle=':')
             # Add text box with more stats
             stats_text=f'StdAbw: {std_w:.3f}\nMin: {min(connection_weights_float):.3f}\nMax: {max(connection_weights_float):.3f}'
             axs[2].text(0.97, 0.97, stats_text, transform=axs[2].transAxes, fontsize='small',
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        else:
             # Display message if no valid connections
             axs[2].text(0.5, 0.5, 'Keine gültigen Verbindungen', ha='center', va='center', transform=axs[2].transAxes)
             axs[2].set_title(f'Klassische Gewichte (N={total_connections})')
    else:
        # Display message if no connections at all
        axs[2].text(0.5, 0.5, 'Keine Verbindungen', ha='center', va='center', transform=axs[2].transAxes)
        axs[2].set_title('Klassische Gewichte (N=0)')

    # Adjust overall layout
    plt.tight_layout(pad=2.0)
    # Save the plot
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True) # Ensure plot directory exists
        filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100) # Save with tight bounding box
        print(f"Plot gespeichert: {filepath}")
        return fig # Return the figure object
    except Exception as e:
        # Print error if saving fails
        print(f"FEHLER beim Speichern des Plots '{filepath}': {e}")
        return None # Return None on error
    finally:
        plt.close(fig) # Close the figure to free memory

def plot_network_graph(nodes_list: List[Node], filename: str = "plot_network_graph.png",
                       min_activation_for_plot: float = 1e-5) -> Optional[plt.Figure]:
    """
    Creates and saves a visualization of the network graph using NetworkX.
    Filters nodes based on type (module, value) or activation level.
    Labels all nodes shown in the graph.
    """
    print(f"Erstelle Plot: Netzwerk-Graph (Zeige Module, Werte, und aktive Knoten >= {min_activation_for_plot})...")
    if not NETWORKX_AVAILABLE:
        print("Übersprungen: networkx Bibliothek fehlt.")
        return None
    if not nodes_list or not isinstance(nodes_list, list):
        print(f"Übersprungen: Keine oder ungültige Knotendaten für Graph übergeben (Typ: {type(nodes_list)}).")
        return None

    G = nx.DiGraph()
    node_info: Dict[str, Dict[str, Any]] = {} # Stores info about nodes added to the graph
    nodes_to_include_in_graph: List[Node] = [] # Nodes that meet criteria for plotting
    module_classes = (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis)

    # --- Step 1: Filter nodes to include in the graph ---
    # Include all Modules, all ValueNodes, and any other node type if its activation is above threshold
    for i, node in enumerate(nodes_list):
        # Basic check: must be a Node instance with a valid label
        if not isinstance(node, Node): continue
        label = getattr(node, 'label', None)
        if label is None or not isinstance(label, str) or not label.strip(): continue

        # Get activation safely, ensure it's finite
        try:
            act_val_raw = getattr(node, 'activation', 0.0)
            activation = float(act_val_raw) if isinstance(act_val_raw, (float, np.number)) and np.isfinite(act_val_raw) else 0.0
        except: activation = 0.0 # Default to 0 if conversion fails or is non-finite

        # Determine if node should be included
        is_module = isinstance(node, module_classes)
        is_value = isinstance(node, ValueNode)
        is_active_enough = activation > min_activation_for_plot

        if is_module or is_value or is_active_enough:
            nodes_to_include_in_graph.append(node)
            # Store info for styling later
            node_type = "base_mq" # Default for quantum nodes
            if is_module: node_type = "module"
            elif is_value: node_type = "value" # Classical value node
            elif isinstance(node, MemoryNode): node_type = "memory_mq"
            elif label.startswith("Q_"): node_type = "question_input" # Input node
            # Check if it's a non-module/value/memory/input node without a Q system
            elif not hasattr(node, 'q_system') or node.q_system is None: node_type = "base_classical"

            node_info[label] = {"type": node_type, "activation": activation}

    print(f"Netzwerk-Graph: {len(nodes_to_include_in_graph)} Knoten werden berücksichtigt.")

    # --- Step 2: Build the graph with included nodes ---
    labels_in_graph = set()
    for node in nodes_to_include_in_graph:
        label = node.label
        if label not in G:
            G.add_node(label)
            labels_in_graph.add(label) # Keep track of nodes actually added

    # --- Step 3: Add edges between included nodes ---
    edges_added = 0
    weight_threshold_for_plot = 0.02 # Don't draw very weak connections
    for node in nodes_to_include_in_graph:
        source_label = node.label
        if hasattr(node, 'connections'):
            for conn in node.connections:
                target_node = conn.target_node
                target_label = getattr(target_node, 'label', None)

                # Add edge only if BOTH source and target are in the graph set
                if target_label and source_label in labels_in_graph and target_label in labels_in_graph:
                    # Get weight safely, ensure finite
                    try:
                        weight_raw = getattr(conn, 'weight', 0.0)
                        weight = float(weight_raw) if isinstance(weight_raw, (float, np.number)) and np.isfinite(weight_raw) else 0.0
                    except: weight = 0.0

                    # Add edge if weight is above threshold
                    if weight > weight_threshold_for_plot:
                         G.add_edge(source_label, target_label, weight=weight)
                         edges_added += 1

    # --- Step 4: Draw the graph (only if nodes exist) ---
    if not G.nodes:
        print("Keine Knoten zum Zeichnen nach Filterung. Graph wird nicht erstellt.")
        return None

    print(f"Netzwerk-Graph: Zeichne {len(G.nodes())} Knoten und {edges_added} Kanten.")

    # Determine node colors and sizes based on type and activation
    node_colors, node_sizes = [], []
    actual_nodes_drawn = list(G.nodes()) # Get the final list of nodes in the graph

    for node_label in actual_nodes_drawn:
        info = node_info.get(node_label, {"type": "unknown", "activation": 0.0})
        node_type, act = info["type"], info["activation"]

        # Assign colors and sizes based on type
        color = 'grey'; size = 250 + act * 1200 # Base size + activation bonus
        if node_type == "module": color = 'orangered'; size *= 1.6
        elif node_type == "value": color = 'gold'; size *= 1.3
        elif node_type == "memory_mq": color = 'skyblue'; size *= 1.1
        elif node_type == "question_input": color = 'lightgreen'; size *= 0.7
        elif node_type == "base_mq": color = 'slateblue'
        elif node_type == "base_classical": color = 'lightgrey'; size *= 0.9

        node_colors.append(color)
        node_sizes.append(max(40, size)) # Ensure a minimum node size

    # Create the plot figure
    fig, ax = plt.subplots(figsize=(26, 22)) # Increase figure size for potentially large graphs
    try:
        # Calculate layout (spring layout often works well for separation)
        # Adjust k based on graph size; smaller k for denser graphs, larger for sparser
        k_val = 0.8 / np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 0.1
        pos = nx.spring_layout(G, k=k_val*1.8, iterations=80, seed=42, scale=2.0) # Increase k and iterations

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=actual_nodes_drawn, node_size=node_sizes,
                               node_color=node_colors, alpha=0.8, ax=ax)

        # Draw edges
        # Use edge weights for transparency?
        # edge_alphas = [G[u][v]['weight'] * 0.4 + 0.1 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.2, edge_color='darkgrey', style='solid',
                               arrows=True, arrowstyle='-|>', arrowsize=9, node_size=node_sizes, ax=ax)

        # Draw labels for ALL nodes in the graph - use small font size
        labels_to_draw = {n: n for n in actual_nodes_drawn}
        nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=5, ax=ax, font_color='black',
                                bbox=dict(facecolor='white', alpha=0.4, boxstyle='round,pad=0.1')) # Add subtle background


        ax.set_title(f"Netzwerk Graph ({NUM_QUBITS_PER_NODE}-Qubit) (N={len(G.nodes())}, E={len(G.edges())})", fontsize=16)
        plt.axis('off') # Hide axes
        plt.tight_layout()

        # Save the graph
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        filepath = os.path.join(PLOTS_FOLDER, filename)
        # Use higher DPI for better resolution of small labels
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        print(f"Plot gespeichert: {filepath}")
        return fig

    except Exception as e:
        print(f"FEHLER beim Zeichnen des Netzwerk-Graphen '{filename}': {e}")
        traceback.print_exc()
        return None
    finally:
        plt.close(fig) # Ensure figure is closed


def plot_emotion_value_trends(interpretation_log: List[Dict], value_history: Dict[str, List[float]],
                                filename: str = "plot_emo_values.png") -> Optional[plt.Figure]:
    """Plots the trends of PAD emotions and classical value node activations over epochs."""
    print("Erstelle Plot: Emotions- & Werte-Trends...")
    if not interpretation_log:
        print("Übersprungen: Keine Interpretationsdaten für Emotions/Werte-Plot.")
        return None

    # Extract data from interpretation log
    epochs = [log.get('epoch', i + 1) for i, log in enumerate(interpretation_log)]
    if not epochs: return None # Need epochs to plot against

    # Extract PAD emotion values, handling potential missing data with NaN
    pleasure = [float(log.get('emotion_state', {}).get('pleasure', np.nan)) for log in interpretation_log]
    arousal = [float(log.get('emotion_state', {}).get('arousal', np.nan)) for log in interpretation_log]
    dominance = [float(log.get('emotion_state', {}).get('dominance', np.nan)) for log in interpretation_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True) # Shared X-axis

    # --- Plot 1: Emotion Trends ---
    ax1.plot(epochs, pleasure, label='Pleasure', color='limegreen', alpha=0.85, marker='.', markersize=4, linestyle='-')
    ax1.plot(epochs, arousal, label='Arousal', color='tomato', alpha=0.85, marker='.', markersize=4, linestyle='-')
    ax1.plot(epochs, dominance, label='Dominance', color='cornflowerblue', alpha=0.85, marker='.', markersize=4, linestyle='-')
    ax1.set_title('Emotionsverlauf (PAD - Klassisch)')
    ax1.set_ylabel('Level [-1, 1]')
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.5, linestyle=':')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.7) # Zero line for reference

    # --- Plot 2: Value Node Trends (Classical) ---
    plotted_values = False
    if value_history:
        # Ensure all value histories are lists of valid floats
        valid_value_histories = {
            k: [float(v_val) for v_val in v if isinstance(v_val, (float, np.number)) and not np.isnan(v_val)]
            for k, v in value_history.items() if v
        }
        # Find the minimum length among valid histories to align plots
        min_len = min((len(h) for h in valid_value_histories.values()), default=0)
        effective_epochs = epochs[:min_len] if min_len > 0 else []

        if effective_epochs: # Only plot if we have aligned epochs and data
            for v_label, history in valid_value_histories.items():
                 if history and len(history) >= min_len: # Ensure history is long enough
                     ax2.plot(effective_epochs, history[:min_len], label=v_label, alpha=0.75, marker='.', markersize=4, linestyle='-')
                     plotted_values = True

    if plotted_values:
        ax2.set_title('Werteverlauf (Klassisch)')
        ax2.set_ylabel('Aktivierung [0, 1]')
        ax2.set_ylim(-0.05, 1.05)
        # Place legend outside plot
        ax2.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax2.grid(True, alpha=0.5, linestyle=':')
    else:
        ax2.set_title('Werteverlauf (Keine Daten)')
        ax2.text(0.5, 0.5, "Keine gültigen Werte-Historien verfügbar", ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xlabel('Epoche')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend

    # Save plot
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        print(f"Plot gespeichert: {filepath}")
        return fig
    except Exception as e:
        print(f"FEHLER beim Speichern des Plots '{filepath}': {e}")
        return None
    finally:
        plt.close(fig) # Close figure

# --- Get Important Categories ---
def get_important_categories(category_nodes: List[MemoryNode], top_n: int = 5) -> List[Tuple[str, str]]:
    """Identifies the top N most active category nodes and assigns an importance level."""
    # Filter for valid MemoryNodes with activation
    valid_nodes = [
        n for n in category_nodes if isinstance(n, MemoryNode)
        and hasattr(n, 'label') and hasattr(n, 'activation')
        # Prüfe ob Aktivierung numerisch und endlich ist
        and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)
    ]
    # Sort by activation descending
    valid_nodes.sort(key=lambda n: float(n.activation), reverse=True)

    important_categories = []
    # Determine importance level based on activation ranges
    for node in valid_nodes[:top_n]:
        act = float(node.activation)
        if act >= 0.8: importance = "sehr hoch"
        elif act >= 0.65: importance = "hoch"
        elif act >= 0.5: importance = "mittel" # Adjusted threshold for MQ
        elif act >= 0.3: importance = "gering" # Adjusted threshold for MQ
        else: importance = "sehr gering"
        important_categories.append((node.label, importance))

    return important_categories


# --- Main Simulation Runner ---
def run_neuropersona_simulation(
    input_df: pd.DataFrame,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,     # Classical LR
    decay_rate: float = DEFAULT_DECAY_RATE,         # Classical Decay
    reward_interval: int = DEFAULT_REWARD_INTERVAL,
    generate_plots: bool = True,
    save_state: bool = False,
    load_state: bool = False,
    status_callback: Callable[[str], None] = _default_status_callback,
    # Multi-Qubit specific parameters
    quantum_shots: int = QUANTUM_ACTIVATION_SHOTS,     # Shots per node activation
    quantum_lr: float = QUANTUM_PARAM_LEARNING_RATE   # LR for quantum parameters
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Executes the full NeuroPersona simulation workflow with multi-qubit nodes.

    Args:
        input_df: DataFrame with columns 'Frage', 'Antwort', 'Kategorie'.
        epochs: Number of simulation epochs.
        learning_rate: Base learning rate for classical weights.
        decay_rate: Decay rate for classical weights.
        reward_interval: Frequency of reinforcement application.
        generate_plots: Whether to generate plots.
        save_state: Whether to save the final network state.
        load_state: Whether to load a previously saved state.
        status_callback: Function to report status updates.
        quantum_shots: Number of measurement shots for quantum activation.
        quantum_lr: Learning rate for internal quantum parameters.

    Returns:
        A tuple containing:
        - final_report_text (str | None): The generated textual report, or None on failure.
        - structured_results (dict | None): Dictionary containing structured results, or None on failure.
    """
    sim_start_time = time.time()
    status_callback(f"\n--- Starte NeuroPersona Simulation ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2 - EXPERIMENTELL) ---")
    status_callback(f"Parameter: E={epochs}, LRc={learning_rate:.4f}, LRq={quantum_lr:.4f}, DRc={decay_rate:.4f}, "
                    f"RI={reward_interval}, QShots={quantum_shots}, Load={load_state}, Save={save_state}, Plots={generate_plots}")

    # --- 1. Input Validation & Preprocessing ---
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        status_callback("FEHLER: Eingabe-DataFrame ist leer oder ungültig.")
        return None, {"error": "Leeres oder ungültiges Input-DataFrame."}
    required_cols = ['Frage', 'Antwort', 'Kategorie']
    if not all(col in input_df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in input_df.columns]
        status_callback(f"FEHLER: Erforderliche Spalten fehlen: {missing}")
        return None, {"error": f"Fehlende Spalten: {missing}"}

    processed_data = preprocess_data(input_df)
    if processed_data.empty or 'Kategorie' not in processed_data.columns or processed_data['Kategorie'].nunique() == 0:
        status_callback("FEHLER: Datenvorverarbeitung fehlgeschlagen oder keine gültigen Kategorien gefunden.")
        return None, {"error": "Datenvorverarbeitung fehlgeschlagen oder keine gültigen Kategorien."}

    # --- 2. Persistence Setup ---
    global persistent_memory_manager
    if persistent_memory_manager is None: # Initialize if not already done globally
        try:
            persistent_memory_manager = PersistentMemoryManager(db_path=PERSISTENT_MEMORY_DB)
            status_callback("Persistent Memory Manager initialisiert.")
        except Exception as e:
            status_callback(f"WARNUNG: Initialisierung des Persistent Memory fehlgeschlagen: {e}")
            persistent_memory_manager = None # Ensure it's None if init fails

    # --- 3. Network Initialization ---
    categories = processed_data['Kategorie'].unique()
    # Ensure initial values are floats
    initial_values_float = {k: float(v) for k, v in DEFAULT_VALUES.items()}
    try:
        category_nodes, module_nodes, value_nodes = initialize_network_nodes(
            categories, initial_values=initial_values_float
        )
        # Check if initialization yielded any nodes
        if not category_nodes and not module_nodes and not value_nodes:
             raise ValueError("Netzwerkinitialisierung ergab keine Knoten.")
        if not category_nodes:
             status_callback("WARNUNG: Keine Kategorie-Knoten initialisiert. Simulation könnte eingeschränkt sein.")
        # Add more checks if needed (e.g., ensure specific modules exist)

    except Exception as init_err:
        status_callback(f"FEHLER bei der Netzwerkinitialisierung: {init_err}")
        return None, {"error": f"Netzwerkinitialisierung fehlgeschlagen: {init_err}"}

    # --- 4. Run Simulation Cycle ---
    activation_history_final, weights_history_final, interpretation_log = {}, {}, []
    final_category_nodes, final_module_nodes, final_value_nodes = [], [], []
    all_final_nodes, final_value_history, final_q_param_history = [], {}, {}
    # Determine if loading should be attempted
    load_filename = MODEL_FILENAME if load_state else None
    simulation_successful = False
    sim_results = None

    try:
        sim_results = simulate_learning_cycle(
            data=processed_data,
            category_nodes=category_nodes,
            module_nodes=module_nodes,
            value_nodes=value_nodes,
            epochs=epochs,
            learning_rate=learning_rate,     # Pass classical LR
            reward_interval=reward_interval,
            decay_rate=decay_rate,         # Pass classical decay
            initial_emotion_state=INITIAL_EMOTION_STATE.copy(), # Pass initial state
            persistent_memory=persistent_memory_manager,
            load_state_from=load_filename, # Pass filename for loading
            status_callback=status_callback,
            # Pass Multi-Qubit specific parameters
            quantum_shots_per_node=quantum_shots,
            quantum_param_lr=quantum_lr
        )
        # Unpack results
        if sim_results:
             activation_history_final, weights_history_final, interpretation_log, \
             final_category_nodes, final_module_nodes, final_value_nodes, \
             all_final_nodes, final_value_history, final_q_param_history = sim_results
             simulation_successful = True # Mark as successful if it returned results
        else:
             status_callback("FEHLER: Simulationszyklus gab keine Ergebnisse zurück.")

    except Exception as e:
        status_callback(f"FATALER FEHLER im Simulationszyklus: {e}")
        print(f"FATALER FEHLER im Simulationszyklus: {e}")
        traceback.print_exc()
        # Return partial results if available, otherwise error indication
        error_payload = {"error": f"Simulationsfehler: {e}",
                         "partial_log": interpretation_log[-5:] if interpretation_log else []}
        # Versuche trotzdem, finale Listen zu bekommen, falls sie existieren
        if 'all_nodes_sim' in locals(): error_payload["final_nodes_count"] = len(all_nodes_sim)
        return "Simulation fehlgeschlagen.", error_payload

    # --- Post-Simulation Checks & Debugging ---
    if not simulation_successful or not all_final_nodes:
        status_callback("WARNUNG: Simulation unvollständig oder keine finalen Knoten vorhanden.")
        # Versuch, einen Bericht aus den bereits vorhandenen Logs zu generieren
        if interpretation_log:
            try:
                status_callback("Versuche, Bericht aus unvollständigen Daten zu generieren...")
                # Generiere Bericht; nutze falls nötig leere Listen
                final_report_text_err, structured_results_err = generate_final_report(
                    final_category_nodes or [],
                    final_module_nodes   or [],
                    final_value_nodes    or [],
                    processed_data,
                    interpretation_log
                )
                # Füge Warnung zum Ergebnis hinzu
                structured_results_err["warning"] = "Simulation unvollständig oder keine finalen Knoten."
                # Standard-Empfehlung setzen, falls sie fehlt
                if "final_recommendation" not in structured_results_err:
                    structured_results_err["final_recommendation"] = "Abwarten (Unvollständig)"
                return final_report_text_err, structured_results_err

            except Exception as report_err:
                # Fange alle Fehler beim Berichtsgenerieren ab
                status_callback(f"FEHLER bei der Berichtgenerierung nach unvollständiger Simulation: {report_err}")
                return (
                    "Simulation fehlgeschlagen, Bericht konnte nicht erstellt werden.",
                    {"error": "Keine finalen Knoten und Logs, Berichtsfehler."}
                )
        else:
            # Weder Logs noch Knoten vorhanden -> deutliche Fehlermeldung
            return "Simulation fehlgeschlagen.", {"error": "Keine finalen Knoten und keine Logs."}


    # --- 5. Generate Final Report ---
    status_callback("Generiere finalen Bericht...")
    try:
        final_report_text, structured_results = generate_final_report(
            final_category_nodes, final_module_nodes, final_value_nodes,
            processed_data, interpretation_log
        )
    except Exception as report_gen_err:
         status_callback(f"FEHLER bei der finalen Berichtsgenerierung: {report_gen_err}")
         print(f"FEHLER bei Berichtsgenerierung: {report_gen_err}")
         traceback.print_exc()
         # Use placeholder report if generation fails
         final_report_text = "Berichtgenerierung fehlgeschlagen."
         # Populate structured results with error and available data
         structured_results = {
             "error": f"Berichtsgenerierung fehlgeschlagen: {report_gen_err}",
             "dominant_category": "N/A", "dominant_activation": 0.0,
             "emotion_state": CURRENT_EMOTION_STATE, # Include final emotion state if available
             "value_node_activations": {v.label: float(v.activation) for v in final_value_nodes if hasattr(v, 'label')},
             "final_recommendation": "Abwarten (Fehler)"
         }

    # --- 6. Generate Plots (Optional) ---
    if generate_plots:
        status_callback(f"\n--- Generiere Plots ({NUM_QUBITS_PER_NODE}-Qubit Knoten) ---")
        os.makedirs(PLOTS_FOLDER, exist_ok=True) # Ensure plot folder exists
        plot_errors = []
        try:
            # Pass q_param_history to the relevant plot function
            plot_activation_and_weights(activation_history_final, weights_history_final, final_q_param_history) # Includes QParam plot
            plot_dynamics(activation_history_final, weights_history_final)
            # Filter history for module nodes before plotting
            module_labels = [m.label for m in final_module_nodes if hasattr(m, 'label')]
            module_hist = filter_module_history(activation_history_final, module_labels)
            plot_module_activation_comparison(module_hist)
            plot_network_structure(all_final_nodes)
            if NETWORKX_AVAILABLE:
                 plot_network_graph(all_final_nodes) # Plot network graph if library available
            else:
                 status_callback("Info: Netzwerk-Graph Plot übersprungen (networkx fehlt).")
            # Emotion/Value plot uses interpretation log and value history
            plot_emotion_value_trends(interpretation_log, final_value_history)
            plt.close('all') # Close all plot figures to free memory
            status_callback("Plots generiert.")
        except Exception as plot_error:
            status_callback(f"FEHLER bei der Plot-Generierung: {plot_error}")
            print(f"FEHLER bei Plot-Generierung: {plot_error}")
            traceback.print_exc()
            plot_errors.append(str(plot_error))
            plt.close('all') # Attempt to close figures even on error
        # Add plot errors to results if any occurred
        if plot_errors and structured_results: structured_results["plot_errors"] = plot_errors

    # --- 7. Derive Final Recommendation ---
    if structured_results: # Only derive if we have structured results
        dom_cat = structured_results.get("dominant_category", "N/A")
        dom_act = structured_results.get("dominant_activation", 0.0)
        pleasure = structured_results.get("emotion_state", {}).get('pleasure', 0.0)
        stability = structured_results.get("stability_assessment", "Unbekannt")
        final_recommendation = "Abwarten" # Default

        # Keywords for semantic assessment of dominant category
        pos_kws = ["chance", "wachstum", "positiv", "potential", "innovation", "lösung", "möglichkeit", "vorteil"]
        neg_kws = ["risiko", "problem", "negativ", "bedrohung", "schwierigkeit", "nachteil", "gefahr"]
        dom_cat_lower = str(dom_cat).lower() if dom_cat != "N/A" else ""

        # Rule-based recommendation based on dominant category, activation, emotion, stability
        if dom_cat != "N/A" and dom_act > 0.55: # Use threshold for normalized Hamming weight
            is_pos = any(kw in dom_cat_lower for kw in pos_kws)
            is_neg = any(kw in dom_cat_lower for kw in neg_kws)

            if is_pos and not is_neg: # Positive dominant category
                # Strong recommendation if stable, high activation, positive mood
                if dom_act > 0.7 and pleasure > 0.1 and "Stabil" in stability: final_recommendation = "Empfehlung"
                else: final_recommendation = "Empfehlung (moderat)"
            elif is_neg and not is_pos: # Negative dominant category
                # Strong warning if stable, high activation, negative mood
                 if dom_act > 0.65 and pleasure < -0.1 and "Stabil" in stability: final_recommendation = "Abraten"
                 else: final_recommendation = "Abraten (moderat)"
            # Neutral/ambiguous dominant category or conflicting signals
            elif dom_act < 0.45 or "Instabil" in stability:
                 final_recommendation = "Abwarten (Instabil/Schwach)"
            else: # Default to Abwarten if not clearly positive/negative
                 final_recommendation = "Abwarten"
        elif "Instabil" in stability: # Instability overrides other factors if activation is low/category unclear
            final_recommendation = "Abwarten (Instabil/Schwach)"

        structured_results["final_recommendation"] = final_recommendation

    # --- 8. Generate HTML Report ---
    if structured_results: # Only generate HTML if we have results
        important_categories = get_important_categories(final_category_nodes, top_n=5)
        # Generate unique filename for the report
        html_filename = f"neuropersona_report_MQ{NUM_QUBITS_PER_NODE}_{time.strftime('%Y%m%d_%H%M%S')}.html"
        try:
            create_html_report(
                final_report_text if final_report_text else "Bericht nicht verfügbar.",
                structured_results.get("final_recommendation", "N/A"),
                interpretation_log, important_categories, structured_results,
                PLOTS_FOLDER, html_filename
            )
        except Exception as html_err:
            status_callback(f"FEHLER bei der HTML-Report-Erstellung: {html_err}")

    # --- 9. Save Final State (Optional) ---
    if save_state:
        if all_final_nodes:
            meta_cog = next((m for m in final_module_nodes if isinstance(m, MetaCognitio)), None)
            save_final_network_state(all_final_nodes, CURRENT_EMOTION_STATE, final_value_nodes, meta_cog, MODEL_FILENAME)
        else:
            status_callback("Info: Speichern übersprungen, keine finalen Knoten zum Speichern.")

    # --- 10. Finalize and Return ---
    sim_end_time = time.time()
    exec_time = round(sim_end_time - sim_start_time, 2)
    status_callback(f"--- NeuroPersona Simulation ({NUM_QUBITS_PER_NODE}-Qubit) abgeschlossen ({exec_time:.2f}s) ---")
    if structured_results: structured_results["execution_time_seconds"] = exec_time

    # Close database connection if it was opened
    if persistent_memory_manager:
        persistent_memory_manager.close()

    return final_report_text, structured_results


# --- GUI Code (Adapted for Multi-Qubit Parameters) ---
# Global variable to store the simulation thread instance
simulation_thread: Optional[threading.Thread] = None

def start_gui():
    """Starts the Tkinter GUI for the NeuroPersona Workflow."""
    root = tk.Tk()
    root.title(f"NeuroPersona Workflow ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2)")
    root.geometry("550x480") # Slightly wider for new parameter

    style = ttk.Style()
    try:
        # Try using a modern theme if available
        style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'
    except tk.TclError:
        print("Hinweis: 'clam' Theme nicht verfügbar, verwende Standard-Theme.")

    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # --- Parameter Input Section ---
    param_container = ttk.LabelFrame(main_frame, text="Simulationsparameter", padding="10")
    param_container.pack(fill=tk.X, padx=5, pady=(5, 10))
    # Configure columns for two-column layout
    param_container.columnconfigure(1, weight=1)
    param_container.columnconfigure(3, weight=1)

    global entry_widgets # Use global dict to store entry widgets
    entry_widgets = {}
    # Define parameters and their defaults
    params_classic = [
        ("Klass. Lernrate:", 'learning_rate', DEFAULT_LEARNING_RATE),
        ("Klass. Decay Rate:", 'decay_rate', DEFAULT_DECAY_RATE),
        ("Reward Int.:", 'reward_interval', DEFAULT_REWARD_INTERVAL), # Abbreviated
        ("Epochen:", 'epochs', DEFAULT_EPOCHS),
    ]
    params_quantum = [
        ("Q Shots/Node:", 'quantum_shots', QUANTUM_ACTIVATION_SHOTS),
        ("Q Param Lernrate:", 'quantum_lr', QUANTUM_PARAM_LEARNING_RATE), # Multi-Qubit parameter
    ]

    # Create labels and entry fields in two columns
    row_idx_left = 0
    for label_text, key, default_value in params_classic:
        ttk.Label(param_container, text=label_text).grid(row=row_idx_left, column=0, sticky=tk.W, pady=3, padx=5)
        entry = ttk.Entry(param_container, width=10)
        entry.insert(0, str(default_value))
        entry.grid(row=row_idx_left, column=1, sticky=tk.EW, pady=3, padx=5)
        entry_widgets[key] = entry
        row_idx_left += 1

    row_idx_right = 0
    for label_text, key, default_value in params_quantum:
        ttk.Label(param_container, text=label_text).grid(row=row_idx_right, column=2, sticky=tk.W, pady=3, padx=15) # Added padx for spacing
        entry = ttk.Entry(param_container, width=10)
        entry.insert(0, str(default_value))
        entry.grid(row=row_idx_right, column=3, sticky=tk.EW, pady=3, padx=5)
        entry_widgets[key] = entry
        row_idx_right += 1

    # --- Options Checkboxes ---
    options_container = ttk.Frame(param_container)
    # Place options below the parameter entries
    options_row = max(row_idx_left, row_idx_right)
    options_container.grid(row=options_row, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))

    generate_plots_var = tk.BooleanVar(value=True)
    save_state_var = tk.BooleanVar(value=False)
    load_state_var = tk.BooleanVar(value=False)

    ttk.Checkbutton(options_container, text="Plots", variable=generate_plots_var).pack(side=tk.LEFT, padx=(0,10))
    ttk.Checkbutton(options_container, text="Speichern", variable=save_state_var).pack(side=tk.LEFT, padx=10)
    ttk.Checkbutton(options_container, text="Laden", variable=load_state_var).pack(side=tk.LEFT, padx=10)

    # --- User Prompt Input ---
    prompt_container = ttk.LabelFrame(main_frame, text="Analyse-Anfrage / Thema", padding="10")
    prompt_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    user_prompt_text = scrolledtext.ScrolledText(prompt_container, height=6, width=50, wrap=tk.WORD,
                                                 relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 9))
    user_prompt_text.pack(fill=tk.BOTH, expand=True, pady=5)
    # Default prompt text
    user_prompt_text.insert("1.0", "Geben Sie hier Ihre Frage oder das zu analysierende Thema ein...\n"
                                   "z.B. Analyse der Chancen und Risiken von Quanten-inspirierten neuronalen Netzen für Finanzprognosen.")

    # --- Status Label ---
    status_label = ttk.Label(main_frame, text=f"Status: Bereit ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2)",
                             anchor=tk.W, relief=tk.GROOVE, padding=(5, 2))
    status_label.pack(fill=tk.X, padx=5, pady=5)

    # --- Button Bar ---
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=(5, 10))
    # Configure columns to push Start button to the right
    button_frame.columnconfigure(0, weight=1) # Empty space expander
    button_frame.columnconfigure(1, weight=0)
    button_frame.columnconfigure(2, weight=0)
    button_frame.columnconfigure(3, weight=0)

    # --- GUI Action Functions ---
    def save_gui_settings():
        """Saves current GUI parameter values to a JSON file."""
        settings_data = {
            "basic_params": {name: widget.get() for name, widget in entry_widgets.items()},
            "options": {
                "generate_plots": generate_plots_var.get(),
                "save_state": save_state_var.get(),
                "load_state": load_state_var.get()
            }
        }
        # Ask user for save location
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=SETTINGS_FILENAME, # Suggest default filename
            title="Simulationsparameter speichern"
        )
        if not filepath: # User cancelled
            status_label.config(text="Status: Speichern abgebrochen.")
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, indent=2)
            status_label.config(text=f"Status: Parameter gespeichert: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Fehler beim Speichern", f"Speichern der Parameter fehlgeschlagen:\n{e}")
            status_label.config(text="Status: Fehler beim Speichern.")

    def load_gui_settings():
        """Loads GUI parameter values from a JSON file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=SETTINGS_FILENAME,
            title="Simulationsparameter laden"
        )
        if not filepath or not os.path.exists(filepath):
            status_label.config(text="Status: Laden abgebrochen oder Datei nicht gefunden.")
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)

            # Load basic parameters, checking if they exist in the file and the GUI
            loaded_params = settings_data.get("basic_params", {})
            for name, widget in entry_widgets.items():
                if name in loaded_params:
                    widget.delete(0, tk.END)
                    widget.insert(0, str(loaded_params[name]))

            # Load options checkboxes
            options = settings_data.get("options", {})
            generate_plots_var.set(options.get("generate_plots", True))
            save_state_var.set(options.get("save_state", False))
            load_state_var.set(options.get("load_state", False))

            status_label.config(text=f"Status: Parameter geladen: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Fehler beim Laden", f"Laden der Parameter fehlgeschlagen:\n{e}")
            status_label.config(text="Status: Fehler beim Laden.")

    def display_final_result(result_text: str, parent_root):
        """Displays the final simulation report in a new window."""
        if not parent_root.winfo_exists(): return # Don't create window if main window closed

        result_window = tk.Toplevel(parent_root)
        result_window.title(f"NeuroPersona Ergebnis ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2)")
        result_window.geometry("750x550")
        result_window.transient(parent_root) # Stay on top of parent
        result_window.grab_set() # Modal behavior

        # Scrolled text widget for the report
        st_widget = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, padx=10, pady=10,
                                             relief=tk.FLAT, font=("Consolas", 9)) # Monospaced font?
        st_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        st_widget.insert(tk.END, result_text if result_text else "Kein Ergebnis Text verfügbar.")
        st_widget.configure(state='disabled') # Make read-only

        # Close button
        button_bar = ttk.Frame(result_window)
        button_bar.pack(pady=(0, 10))
        ttk.Button(button_bar, text="Schließen", command=result_window.destroy).pack()

        result_window.wait_window() # Wait for the result window to close

    def run_workflow_in_thread(user_prompt, params_dict, gen_plots, save_st, load_st,
                               status_cb_ref, start_btn_ref, save_btn_ref, load_btn_ref, root_ref):
        """Executes the simulation workflow in a separate thread to keep the GUI responsive."""
        global persistent_memory_manager # Ensure access to the global instance
        final_result_text = "Workflow gestartet..."
        try:
            status_cb_ref(f"Führe Simulation aus ({NUM_QUBITS_PER_NODE}-Qubit)...")

            # Generate dummy data based on user prompt (replace with actual data loading if needed)
            try:
                # Simple way to vary dummy data size based on prompt length
                num_dummy_rows = max(5, min(50, len(user_prompt) // 15))
                # More diverse categories
                categories = ["Technologie", "Markt", "Ethik", "Risiko", "Chance", "Implementierung", "Skalierung", "Regulierung"]
                dummy_data = {
                    'Frage': [f"F{i+1}_{user_prompt[:15]}?" for i in range(num_dummy_rows)],
                    'Antwort': [random.choice(["hoch", "mittel", "niedrig", "ja", "nein", "positiv", "negativ", "unsicher"])
                                for _ in range(num_dummy_rows)],
                    'Kategorie': [random.choice(categories) for _ in range(num_dummy_rows)]
                }
                input_dataframe = pd.DataFrame(dummy_data)
                status_cb_ref(f"{num_dummy_rows} Dummy-Datenpunkte für Simulation erstellt.")
            except Exception as data_err:
                status_cb_ref(f"Fehler bei Dummy-Datenerstellung: {data_err}")
                raise # Stop if data generation fails

            # Call the main simulation function (multi-qubit version)
            report_text, structured_res = run_neuropersona_simulation(
                input_df=input_dataframe,
                epochs=params_dict['epochs'],
                learning_rate=params_dict['learning_rate'], # Classical LR
                decay_rate=params_dict['decay_rate'],       # Classical Decay
                reward_interval=params_dict['reward_interval'],
                generate_plots=gen_plots,
                save_state=save_st,
                load_state=load_st,
                status_callback=status_cb_ref,
                quantum_shots=params_dict['quantum_shots'], # Q Shots
                quantum_lr=params_dict['quantum_lr']        # Q Param LR
            )

            # Process results
            final_result_text = report_text if report_text else "Simulation beendet, kein Bericht generiert."
            if structured_res and "error" in structured_res:
                status_cb_ref(f"Workflow mit Fehlern beendet: {structured_res['error']}")
                final_result_text += f"\n\nFEHLER: {structured_res['error']}" # Append error to text
            elif final_result_text:
                status_cb_ref("Workflow erfolgreich abgeschlossen.")
            else:
                status_cb_ref("Workflow beendet, aber kein Ergebnis erhalten.")

            # Display results in the GUI thread
            if final_result_text and root_ref.winfo_exists():
                 # Use root.after to schedule GUI update from the main thread
                 root_ref.after(0, lambda: display_final_result(final_result_text, root_ref))

        except Exception as e:
            # Catch any unexpected errors during the simulation run
            error_traceback = traceback.format_exc()
            print(f"FATALER FEHLER im Workflow-Thread: {e}\n{error_traceback}")
            status_cb_ref(f"Schwerwiegender Fehler: {e}")
            # Show error message box in the GUI thread
            if root_ref.winfo_exists():
                root_ref.after(0, lambda: messagebox.showerror("Workflow Fehler",
                               f"Ein unerwarteter Fehler ist im Hintergrund aufgetreten:\n{e}\n\nDetails siehe Konsole."))
        finally:
            # Re-enable buttons in the GUI thread, regardless of success or failure
            if root_ref.winfo_exists():
                root_ref.after(0, lambda: start_btn_ref.config(state=tk.NORMAL))
                root_ref.after(0, lambda: save_btn_ref.config(state=tk.NORMAL))
                root_ref.after(0, lambda: load_btn_ref.config(state=tk.NORMAL))
            # Ensure database connection is closed IF IT WAS OPENED BY THIS THREAD
            # The global manager might be closed later by the main thread closing handler
            # It's generally safer to let the main thread handle the final close.
            # However, if the workflow creates its own instance, it should close it.
            # Assuming the global instance is used:
            # if persistent_memory_manager:
            #     persistent_memory_manager.close() # Might cause issues if main thread also closes it.

    def start_full_workflow_action():
        """Called when the 'Start Workflow' button is pressed."""
        global simulation_thread # Access the global thread variable
        user_prompt = user_prompt_text.get("1.0", tk.END).strip()
        # Basic check if prompt is empty or default
        if not user_prompt or user_prompt.startswith("Geben Sie hier"):
            messagebox.showwarning("Eingabe fehlt", "Bitte geben Sie eine Analyse-Anfrage oder ein Thema ein.")
            return

        # Check if another simulation is already running
        if simulation_thread and simulation_thread.is_alive():
             messagebox.showwarning("Läuft bereits", "Eine Simulation wird bereits ausgeführt. Bitte warten.")
             return

        params_values = {}
        try:
            # --- Retrieve and validate parameters from GUI entries ---
            # Classical Parameters
            params_values['learning_rate'] = float(entry_widgets['learning_rate'].get().replace(',', '.'))
            params_values['decay_rate'] = float(entry_widgets['decay_rate'].get().replace(',', '.'))
            params_values['reward_interval'] = int(entry_widgets['reward_interval'].get())
            params_values['epochs'] = int(entry_widgets['epochs'].get())
            # Quantum Parameters
            params_values['quantum_shots'] = int(entry_widgets['quantum_shots'].get())
            params_values['quantum_lr'] = float(entry_widgets['quantum_lr'].get().replace(',', '.'))

            # --- Basic Parameter Validation ---
            if not (0 < params_values['learning_rate'] <= 0.5): raise ValueError("Klass. Lernrate sollte > 0 und <= 0.5 sein.")
            if not (0 <= params_values['decay_rate'] < 1.0): raise ValueError("Klass. Decay Rate sollte >= 0 und < 1 sein.")
            if not (params_values['reward_interval'] >= 1): raise ValueError("Reward Interval muss >= 1 sein.")
            if not (1 <= params_values['epochs'] <= 1000): raise ValueError("Epochen müssen zwischen 1 und 1000 liegen.") # Added upper bound
            if not (1 <= params_values['quantum_shots'] <= 100): raise ValueError("Quantum Shots müssen zwischen 1 und 100 liegen.") # Added upper bound
            if not (0 < params_values['quantum_lr'] <= 0.5): raise ValueError("Q Param Lernrate sollte > 0 und <= 0.5 sein.")

        except (ValueError, KeyError, TypeError) as ve:
            messagebox.showerror("Eingabefehler", f"Ungültiger oder fehlender Parameterwert.\nStellen Sie sicher, dass alle Werte korrekt sind.\n({ve})")
            return

        # Update status and disable buttons
        status_label.config(text="Status: Starte Workflow...")
        start_button.config(state=tk.DISABLED)
        save_button.config(state=tk.DISABLED)
        load_button.config(state=tk.DISABLED)

        # Define the status update function to be passed to the thread
        def gui_status_update(message: str):
            # Ensure GUI update happens in the main thread
            if root.winfo_exists():
                # Truncate long messages for the status bar
                truncated_msg = message[:120] + ('...' if len(message) > 120 else '')
                # Use after to schedule the update
                root.after(0, lambda: status_label.config(text=f"Status: {truncated_msg}"))
                # Optional: Print full message to console
                # print(f"[Sim Status] {message}")

        # Start the workflow in a separate thread
        # Store the thread instance in the global variable
        simulation_thread = threading.Thread(
            target=run_workflow_in_thread,
            args=(user_prompt, params_values, generate_plots_var.get(), save_state_var.get(), load_state_var.get(),
                  gui_status_update, start_button, save_button, load_button, root),
            daemon=True # Allows the main program to exit even if thread is running
        )
        simulation_thread.start()

    # --- Create and Place Buttons ---
    save_button = ttk.Button(button_frame, text="Params Speichern", command=save_gui_settings)
    save_button.grid(row=0, column=1, padx=5, sticky=tk.E)

    load_button = ttk.Button(button_frame, text="Params Laden", command=load_gui_settings)
    load_button.grid(row=0, column=2, padx=5, sticky=tk.E)

    # Style the Start button
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="white", background="#007bff")
    style.map("Accent.TButton", background=[('active', '#0056b3'), ('disabled', '#cccccc')]) # Add disabled style

    start_button = ttk.Button(button_frame, text="Workflow starten", style="Accent.TButton", command=start_full_workflow_action)
    start_button.grid(row=0, column=3, padx=(15, 0), sticky=tk.E)


    # --- Load previous GUI settings if available ---
    if os.path.exists(SETTINGS_FILENAME):
        try:
            load_gui_settings() # Attempt to load settings on startup
            status_label.config(text=f"Status: Letzte Parameter '{os.path.basename(SETTINGS_FILENAME)}' geladen.")
        except Exception as e:
            status_label.config(text=f"Status: Fehler beim Laden von '{os.path.basename(SETTINGS_FILENAME)}'.")

    # --- Handle Window Closing ---
    def on_closing():
        """Handles the event when the user tries to close the window."""
        global simulation_thread
        is_running = simulation_thread and simulation_thread.is_alive()
        confirm_close = True # Assume we can close unless a simulation is running

        if is_running:
             # Ask user if they really want to close while simulation runs
             confirm_close = messagebox.askyesno("Simulation läuft", "Eine Simulation wird gerade ausgeführt.\nWirklich beenden? Der aktuelle Lauf geht verloren.")

        if confirm_close:
            print("GUI wird geschlossen.")
            # Optional: Wait a short time for the thread to potentially finish IO if closing was confirmed
            # This is a pragmatic compromise, not a perfect guarantee.
            # if is_running:
            #     print("Warte kurz auf Thread (max 1 Sekunde)...")
            #     simulation_thread.join(timeout=1.0) # Wait max 1 second
            # Ensure database connection is closed on exit
            global persistent_memory_manager
            if persistent_memory_manager:
                print("Schließe DB Verbindung von on_closing...")
                persistent_memory_manager.close()
                # Verhindere doppeltes Schließen, indem die Instanz auf None gesetzt wird
                persistent_memory_manager = None # Prevent closing again in thread's finally block
            root.destroy() # Close the Tkinter window

    root.protocol("WM_DELETE_WINDOW", on_closing) # Register closing handler
    root.mainloop() # Start the Tkinter event loop


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Starte NeuroPersona Core ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2)")

    # Attempt to initialize Persistent Memory Manager globally ONCE at startup
    # This instance can then be passed to the simulation function.
    try:
        if persistent_memory_manager is None: # Initialize only if not already done
            persistent_memory_manager = PersistentMemoryManager(db_path=PERSISTENT_MEMORY_DB)
    except Exception as e:
        print(f"FEHLER bei globaler Initialisierung von PersistentMemoryManager: {e}")
        persistent_memory_manager = None # Ensure it's None if init fails

    # Start the GUI
    start_gui()

    # Final message after GUI closes
    print(f"NeuroPersona Core ({NUM_QUBITS_PER_NODE}-Qubit Knoten v2) beendet.")