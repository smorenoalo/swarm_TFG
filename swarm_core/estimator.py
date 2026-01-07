#!/usr/bin/env python3
"""
================================================================================
ESTIMADOR DE POSICIÓN BASADO EN UKF
================================================================================

Implementa Unscented Kalman Filter para fusión de mediciones UWB.

NOTA: La integración de velocidad (predicción) se realiza externamente
en SwarmController. Este UKF propaga incertidumbre y fusiona mediciones.

Modos de estimación:
- IMU-only: Solo predicción externa (para ablación)
- UWB-only: Solo actualización UWB, sin predicción (para ablación)  
- IMU+UWB: Predicción externa + fusión UWB (baseline)
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from typing import List, Tuple, Optional
from enum import Enum


class EstimatorMode(Enum):
    """Modos de estimación para ablación"""
    IMU_ONLY = "imu_only"
    UWB_ONLY = "uwb_only"
    IMU_UWB = "imu_uwb"


class UKFEstimator:
    """
    Estimador de posición basado en Unscented Kalman Filter
    
    Estado: [x, y] - posición 2D
    
    NOTA: El modelo de transición interno es identidad (fx = x).
    La predicción de posición (integración de velocidad) se realiza
    EXTERNAMENTE en SwarmController y se pasa a predict() como pos_pred.
    Este UKF principalmente propaga incertidumbre y fusiona mediciones UWB.
    
    Medición: Distancias UWB a vecinos
    """
    
    def __init__(self, N: int, dt: float, mode: EstimatorMode = EstimatorMode.IMU_UWB,
                 alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0):
        self.N = N
        self.dt = dt
        self.mode = mode
        self.n = 2  # Dimensión del estado
        
        # Parámetros de sigma points
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Estado y covarianza por agente
        self.means = np.zeros((N, 2))
        self.covs = [np.eye(2) * 0.1 for _ in range(N)]
        
        # Crear UKFs individuales
        self.ukf_list: List[UnscentedKalmanFilter] = []
        self._init_ukfs()
        
        # Estadísticas de actualización
        self.update_counts = np.zeros(N)
        self.rejected_counts = np.zeros(N)
    
    def _init_ukfs(self):
        """Inicializa los filtros UKF para cada agente"""
        self.ukf_list = []
        for i in range(self.N):
            points = MerweScaledSigmaPoints(
                n=self.n, alpha=self.alpha, beta=self.beta, kappa=self.kappa
            )
            ukf = UnscentedKalmanFilter(
                dim_x=self.n, dim_z=1, dt=self.dt,
                fx=lambda x, dt: x,  # Modelo de transición simple
                hx=lambda x: np.array([0.0]),  # Se redefine en update
                points=points
            )
            ukf.x = self.means[i].copy()
            ukf.P = self.covs[i].copy()
            ukf.Q = np.eye(self.n) * 0.005  # Ruido de proceso
            ukf.R = np.array([[0.04]])       # Ruido de medición base
            self.ukf_list.append(ukf)

    def _ensure_positive_definite(self, P: np.ndarray) -> np.ndarray:
        """Asegura que la matriz sea simétrica y positiva definida."""
        P = 0.5 * (P + P.T)
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T + 1e-9 * np.eye(2)

    def predict(self, i: int, pos_pred: np.ndarray, dt: float):
        """
        Paso de predicción UKF.
        
        NOTA: Este método recibe la posición ya predicha externamente
        (integración de velocidad hecha en SwarmController). El UKF
        propaga la incertidumbre pero no hace la integración internamente.
        
        Args:
            i: Índice del agente
            pos_pred: Posición pre-calculada (integración externa)
            dt: Paso de tiempo
        """
        if self.mode == EstimatorMode.UWB_ONLY:
            # Sin predicción IMU, mantener estado anterior
            return
        
        ukf = self.ukf_list[i]
        P = self._ensure_positive_definite(self.covs[i])
        ukf.P = P
        ukf.x = pos_pred.copy()
        ukf.dt = dt

        try:
            ukf.predict()
        except np.linalg.LinAlgError:
            ukf.x = pos_pred.copy()
            ukf.P = P + np.eye(self.n) * 0.1

        self.means[i] = ukf.x.copy()
        self.covs[i] = self._ensure_positive_definite(ukf.P)

    def update_range(self, i: int, z: float, anchor_pos: np.ndarray, R: float):
        """
        Actualización UKF con medición de rango.
        
        Args:
            i: Índice del agente
            z: Distancia medida
            anchor_pos: Posición del ancla/vecino
            R: Varianza de la medición
        """
        if self.mode == EstimatorMode.IMU_ONLY:
            # Sin actualización UWB
            return
        
        ukf = self.ukf_list[i]
        ukf.x = self.means[i].copy()
        ukf.P = self._ensure_positive_definite(self.covs[i])

        def hx(x, anchor=anchor_pos):
            d = np.linalg.norm(x - anchor)
            return np.array([max(d, 0.01)])

        try:
            ukf.update(np.array([z]), R=np.array([[max(R, 1e-4)]]), hx=hx)
            self.update_counts[i] += 1
        except np.linalg.LinAlgError:
            self.rejected_counts[i] += 1
            return

        self.means[i] = ukf.x.copy()
        self.covs[i] = self._ensure_positive_definite(ukf.P)

    def direct_update(self, i: int, target_pos: np.ndarray, R_diag: float = 0.01):
        """
        Actualización directa con posición objetivo (para modo estático).
        
        Args:
            i: Índice del agente
            target_pos: Posición objetivo
            R_diag: Varianza diagonal
        """
        if self.mode == EstimatorMode.IMU_ONLY:
            return
            
        x = self.means[i]
        P = self._ensure_positive_definite(self.covs[i])
        R = np.eye(2) * R_diag
        
        try:
            S = P + R
            K = P @ np.linalg.inv(S)
            y = target_pos - x
            self.means[i] = x + K @ y
            self.covs[i] = self._ensure_positive_definite((np.eye(2) - K) @ P)
        except np.linalg.LinAlgError:
            self.means[i] = 0.9 * x + 0.1 * target_pos

    def set_known_position(self, i: int, pos: np.ndarray):
        """Establece posición conocida (para el líder)"""
        self.means[i] = pos.copy()
        self.covs[i] = np.eye(2) * 1e-9
        self.ukf_list[i].x = pos.copy()
        self.ukf_list[i].P = np.eye(2) * 1e-9

    def reset(self, initial_positions: Optional[np.ndarray] = None):
        """Reinicia el estimador"""
        self.means = np.zeros((self.N, 2)) if initial_positions is None \
                     else initial_positions.copy()
        self.covs = [np.eye(2) * 0.1 for _ in range(self.N)]
        self._init_ukfs()
        self.update_counts = np.zeros(self.N)
        self.rejected_counts = np.zeros(self.N)

    def get_trace(self, i: int) -> float:
        """Obtiene la traza de la covarianza del agente i"""
        return max(np.trace(self.covs[i]), 1e-6)

    def get_stats(self) -> dict:
        """Retorna estadísticas del estimador"""
        return {
            'update_counts': self.update_counts.copy(),
            'rejected_counts': self.rejected_counts.copy(),
            'mean_cov_trace': np.mean([np.trace(c) for c in self.covs])
        }


class SwitchableEstimator:
    """
    Wrapper que permite cambiar entre modos de estimación
    para experimentos de ablación.
    """
    
    def __init__(self, N: int, dt: float, mode: str = "imu_uwb"):
        self.N = N
        self.dt = dt
        self.mode = EstimatorMode(mode)
        self.estimator = UKFEstimator(N, dt, self.mode)
    
    def set_mode(self, mode: str):
        """Cambia el modo de estimación"""
        self.mode = EstimatorMode(mode)
        self.estimator.mode = self.mode
    
    def __getattr__(self, name):
        """Delega llamadas al estimador interno"""
        return getattr(self.estimator, name)
