#!/usr/bin/env python3
"""
================================================================================
SIMULACIÓN DE SENSORES
================================================================================

Modelos de sensores:
- IMU (simplificado): Velocidad con drift y ruido gaussiano
- UWB: Distancias inter-robot con modelo IEEE 802.15.4a (NLOS/LOS)
- Detector NLOS: Análisis estadístico de mediciones
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple

from configs.params import UWBParams, IMUParams


class NLOSDetector:
    """
    Detector de condición Non-Line-of-Sight para mediciones UWB
    
    Utiliza análisis estadístico de ventana deslizante para detectar
    mediciones anómalas que indican NLOS.
    
    Criterios de detección:
    - Variación temporal excesiva (velocidad implícita)
    - Alta varianza en ventana de historial
    """
    
    def __init__(self, window_size: int = 10, std_threshold: float = 0.20, 
                 velocity_threshold: float = 8.0):
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.velocity_threshold = velocity_threshold
        self.history: Dict[tuple, deque] = {}
        self.last_results: Dict[tuple, float] = {}

    def detect(self, link_id: tuple, measured_dist: float, dt: float) -> float:
        """
        Calcula probabilidad de NLOS para una medición.
        
        Args:
            link_id: Identificador del enlace (i, j)
            measured_dist: Distancia medida [m]
            dt: Intervalo de tiempo [s]
            
        Returns:
            score: Probabilidad de NLOS [0, 1]
        """
        if link_id not in self.history:
            self.history[link_id] = deque(maxlen=self.window_size)
            self.history[link_id].append(measured_dist)
            self.last_results[link_id] = 0.5
            return 0.5  # Incertidumbre inicial

        hist = self.history[link_id]
        prev_dist = hist[-1]
        
        # Criterio 1: Velocidad implícita excesiva
        measured_vel = abs(measured_dist - prev_dist) / dt if dt > 0 else 0.0
        velocity_violation = measured_vel > self.velocity_threshold

        # Criterio 2: Alta variabilidad
        hist.append(measured_dist)
        sigma = np.std(hist) if len(hist) > 3 else 0.0
        is_noisy = sigma > self.std_threshold

        # Combinar criterios
        score = 0.1
        if velocity_violation:
            score += 0.5
        if is_noisy:
            score += 0.3

        score = float(np.clip(score, 0.05, 0.95))
        self.last_results[link_id] = score
        return score
    
    def update(self, link_id: tuple, measured_dist: float, 
               estimated_dist: float = None, dt: float = 0.02) -> Tuple[bool, float, dict]:
        """
        API alternativa compatible con código anterior.
        
        Returns:
            is_nlos: True si score > 0.5
            score: Probabilidad NLOS [0, 1]
            diagnostics: Diccionario con detalles (vacío)
        """
        score = self.detect(link_id, measured_dist, dt)
        return (score > 0.5, score, {})
    
    def get_score(self, link_id: tuple) -> float:
        """Obtiene el último score NLOS para un enlace."""
        return self.last_results.get(link_id, 0.5)
    
    def reset(self):
        """Reinicia el historial del detector."""
        self.history.clear()
        self.last_results.clear()


class IMUSensor:
    """
    Simulador simplificado de sensor inercial (modelo de velocidad)
    
    NOTA: Este es un modelo simplificado que añade ruido y drift
    directamente a la velocidad, NO simula un IMU real que mediría
    aceleración. Es una aproximación válida para este TFG.
    
    Modela:
    - Ruido blanco gaussiano en velocidad
    - Drift (bias) con random walk
    """
    
    def __init__(self, N: int, params: IMUParams, rng: np.random.RandomState = None):
        self.N = N
        self.params = params
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Bias verdadero del IMU
        self.true_bias = self.rng.normal(
            0, params.initial_bias_std, (N, 2)
        )
        
        # Estimación del bias
        self.est_bias = np.zeros((N, 2))
        
        # Buffer de calibración
        self.calibration_buffer = [[] for _ in range(N)]
    
    def read(self, i: int, true_vel: np.ndarray) -> np.ndarray:
        """
        Simula lectura de IMU con bias y ruido.
        
        Args:
            i: Índice del agente
            true_vel: Velocidad verdadera
            
        Returns:
            vel_measured: Velocidad medida con ruido
        """
        # Evolución del bias (random walk)
        self.true_bias[i] += self.rng.normal(
            0, self.params.bias_instability, 2
        )
        
        # Ruido de medición
        noise = self.rng.normal(0, self.params.vel_noise_density, 2)
        
        return true_vel + self.true_bias[i] + noise
    
    def calibrate(self, i: int, vel_raw: np.ndarray):
        """Agrega muestra al buffer de calibración"""
        self.calibration_buffer[i].append(vel_raw)
        if len(self.calibration_buffer[i]) > 5:
            self.est_bias[i] = np.mean(self.calibration_buffer[i], axis=0)
    
    def correct_bias(self, i: int, correction: np.ndarray, learning_rate: float,
                     decay: float, max_bias: float = 0.3):
        """Corrige el bias estimado"""
        self.est_bias[i] -= learning_rate * correction
        self.est_bias[i] *= decay
        self.est_bias[i] = np.clip(self.est_bias[i], -max_bias, max_bias)
    
    def reset(self, rng: np.random.RandomState = None):
        """Reinicia el sensor con nuevo estado"""
        if rng is not None:
            self.rng = rng
        self.true_bias = self.rng.normal(
            0, self.params.initial_bias_std, (self.N, 2)
        )
        self.est_bias = np.zeros((self.N, 2))
        self.calibration_buffer = [[] for _ in range(self.N)]


class UWBSensor:
    """
    Simulador de sensor UWB
    
    Modela:
    - Line-of-Sight (LOS): ruido gaussiano pequeño
    - Non-Line-of-Sight (NLOS): bias positivo con distribución gamma
    
    Incluye detector NLOS basado en análisis estadístico de residuos.
    """
    
    def __init__(self, N: int, params: UWBParams, dt: float,
                 rng: np.random.RandomState = None):
        self.N = N
        self.params = params
        self.dt = dt
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Detector NLOS heurístico simple
        self.nlos_detector = NLOSDetector(
            window_size=10,
            std_threshold=0.20,
            velocity_threshold=8.0
        )
        
        # Última matriz de scores NLOS
        self.last_nlos_scores = np.zeros((N, N))
        
        # Diagnósticos del detector (para debugging/análisis)
        self.last_diagnostics: Dict[tuple, dict] = {}
    
    def measure(self, positions: np.ndarray, 
                estimated_positions: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula mediciones UWB entre todos los pares de agentes.
        
        Args:
            positions: Posiciones verdaderas [N, 2]
            estimated_positions: Posiciones estimadas [N, 2] (mejora detección NLOS)
            
        Returns:
            dist_meas: Matriz de distancias medidas [N, N]
            nlos_scores: Matriz de scores NLOS [N, N]
        """
        m = self.params
        dist_meas = np.zeros((self.N, self.N))
        nlos_scores = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Distancia verdadera
                d_true = np.linalg.norm(positions[i] - positions[j])
                
                # Probabilidad de NLOS aumenta con distancia
                p_nlos = min(0.8, m.prob_nlos_base + m.dist_decay * max(0, d_true - 2.0))
                is_nlos_true = self.rng.random() < p_nlos

                # Generar error de medición según modelo IEEE 802.15.4a
                if is_nlos_true:
                    # NLOS: ruido gaussiano + bias gamma (siempre positivo)
                    error_los = self.rng.normal(0, m.sigma_los)
                    error_nlos = self.rng.gamma(m.gamma_shape, m.gamma_scale)
                    error = error_los + error_nlos
                else:
                    # LOS: solo ruido gaussiano
                    error = self.rng.normal(0, m.sigma_los)
                
                # Medición final (nunca negativa)
                meas = max(0.05, d_true + error)
                
                # Distancia estimada para el detector (si disponible)
                est_dist = None
                if estimated_positions is not None:
                    est_dist = np.linalg.norm(
                        estimated_positions[i] - estimated_positions[j]
                    )
                
                # Actualizar detector NLOS
                link_id = (min(i,j), max(i,j))
                score = self.nlos_detector.detect(link_id, meas, self.dt)
                
                # Almacenar resultados
                dist_meas[i, j] = dist_meas[j, i] = meas
                nlos_scores[i, j] = nlos_scores[j, i] = score

        self.last_nlos_scores = nlos_scores.copy()
        return dist_meas, nlos_scores
    
    def get_diagnostics(self, i: int, j: int) -> dict:
        """Obtiene diagnósticos del detector para un enlace específico."""
        link_id = (min(i,j), max(i,j))
        return self.last_diagnostics.get(link_id, {})
    
    def reset(self, rng: np.random.RandomState = None):
        """Reinicia el sensor"""
        if rng is not None:
            self.rng = rng
        self.nlos_detector.reset()
        self.last_nlos_scores = np.zeros((self.N, self.N))
        self.last_diagnostics.clear()


class LeaderGNSS:
    """
    Simulador de broadcast de posición del líder con ruido GNSS
    """
    
    def __init__(self, noise_std: float = 0.02, seed: int = 42):
        self.noise_std = noise_std
        # Usamos seed fija para el broadcast del líder
        self.rng = np.random.RandomState(seed)
        self._broadcast = np.zeros(2)
    
    def get_broadcast(self, true_pos: np.ndarray) -> np.ndarray:
        """Obtiene posición broadcast con ruido GNSS"""
        noise = self.rng.normal(0, self.noise_std, 2)
        self._broadcast = true_pos + noise
        return self._broadcast
    
    @property
    def last_broadcast(self) -> np.ndarray:
        return self._broadcast
