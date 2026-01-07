#!/usr/bin/env python3
"""
================================================================================
CONFIGURACIÓN DE PARÁMETROS DEL SISTEMA
================================================================================

Centraliza todos los parámetros del experimento para garantizar reproducibilidad.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class UWBParams:
    """Parámetros del modelo de ruido UWB basado en IEEE 802.15.4a"""
    sigma_los: float = 0.02          # Desviación estándar en LOS [m]
    gamma_shape: float = 2.0         # Parámetro k de distribución gamma
    gamma_scale: float = 0.15        # Parámetro θ de distribución gamma
    prob_nlos_base: float = 0.05     # Probabilidad base de NLOS
    dist_decay: float = 0.1          # Incremento de prob NLOS con distancia


@dataclass
class IMUParams:
    """Parámetros del modelo de ruido de velocidad (IMU simplificado)"""
    vel_noise_density: float = 0.05     # Densidad de ruido de velocidad [m/s/√Hz]
    bias_instability: float = 0.0005    # Inestabilidad del drift [m/s por paso]
    initial_bias_std: float = 0.05      # Desviación estándar del drift inicial


@dataclass
class NeuralParams:
    """Parámetros de la red neuronal GAT"""
    gat_hidden_dim: int = 128    # Dimensión oculta
    gat_num_heads: int = 4       # Número de cabezas de atención
    gat_num_layers: int = 2      # Número de capas GAT
    in_features: int = 16        # Dimensiones de features de entrada


@dataclass 
class ControlParams:
    """Parámetros del sistema de control completo"""
    # Configuración del enjambre
    N: int = 10                           # Número de agentes
    dt: float = 0.02                      # Paso de tiempo [s] (50 Hz)
    
    # Sub-configuraciones
    uwb: UWBParams = field(default_factory=UWBParams)
    imu: IMUParams = field(default_factory=IMUParams)
    neural: NeuralParams = field(default_factory=NeuralParams)
    
    # UKF
    max_neighbors: int = 4                # Máximo vecinos para actualización
    mahalanobis_threshold: float = 3.0    # Umbral de rechazo de mediciones
    
    # PID
    k_p: float = 15.0                     # Ganancia proporcional
    k_d: float = 8.0                      # Ganancia derivativa
    k_i: float = 2.0                      # Ganancia integral
    k_consensus: float = 0.5              # Ganancia de consenso
    
    # Límites físicos
    v_max: float = 6.0                    # Velocidad máxima [m/s]
    a_max: float = 12.0                   # Aceleración máxima [m/s²]
    d_safe: float = 0.5                   # Distancia de seguridad [m]
    
    # Calibración IMU
    calibration_time: float = 2.0         # Tiempo de calibración [s]
    bias_learning_rate: float = 0.001     # Tasa de aprendizaje del bias
    bias_decay: float = 0.9995            # Decaimiento del bias
    turn_gate_threshold: float = 0.5      # Umbral para detectar giros
    
    # ControlNet (GAT)
    ctrl_gamma_base: float = 0.5          # Gamma mínimo
    ctrl_gamma_max: float = 0.8           # Gamma máximo  
    ctrl_max_delta: float = 2.0           # Delta máximo permitido
    ctrl_err_gate: float = 2.0            # Gate de error de estimación
    ctrl_cov_gate: float = 2.0            # Gate de covarianza
    ctrl_nlos_gate: float = 0.8           # Gate de probabilidad NLOS
    ctrl_burnin_steps: int = 300          # Pasos de burn-in
    ctrl_leader_acc_gate: float = 4.0     # Gate de aceleración del líder
    ctrl_lp_alpha: float = 0.4            # Alpha del filtro pasa-bajos
    ctrl_v_ref_gate: float = 0.5          # Velocidad de referencia
    ctrl_delta_scale: float = 1.0         # Escala de delta
    ctrl_rmse_gate: float = 1.0           # Gate de RMSE
    ctrl_confidence_thresh: float = 0.2   # Umbral de confianza
    
    # Modo hover/estático
    hover_speed_thresh: float = 0.15      # Velocidad para detectar hover
    hover_pseudo_R: float = 0.02          # R pseudo para hover
    hover_min_time: float = 1.0           # Tiempo mínimo en hover
    hover_k_i_scale: float = 3.0          # Escala K_i en hover
    hover_k_consensus_scale: float = 2.0  # Escala consenso en hover
    hover_k_phys: float = 10.0            # K física en hover
    hover_drag_scale: float = 4.0         # Escala de drag en hover
    
    # ZUPT (Zero Velocity Update)
    zupt_extreme_enabled: bool = True
    zupt_extreme_min_time: float = 2.0
    zupt_extreme_blend: float = 0.9
    
    # Otros
    leader_gnss_noise: float = 0.02       # Ruido GNSS del líder
    base_drag: float = 1.0                # Coeficiente de arrastre base


@dataclass
class ExperimentConfig:
    """Configuración completa de un experimento"""
    # Identificación
    name: str = "default_experiment"
    
    # Escenarios
    scenarios: List[str] = field(default_factory=lambda: [
        'spiral', 'lawnmower', 'snake', 'station_keeping'
    ])
    
    # Seeds para reproducibilidad (30 seeds = 5 runs × 6 seeds)
    seed_ranges: List[tuple] = field(default_factory=lambda: [
        (0, 5), (10, 15), (20, 25), (30, 35), (40, 45)
    ])
    
    # Duración
    mission_duration: float = 100.0        # Duración del test [s]
    
    # Parámetros del sistema
    params: ControlParams = field(default_factory=ControlParams)
    
    # Umbrales para métricas
    loc_threshold_cm: float = 100.0       # Umbral de error localización [cm]
    form_convergence_cm: float = 30.0     # Umbral convergencia formación [cm]
    convergence_hold_time: float = 2.0    # Tiempo para mantener convergencia [s]
    
    def get_all_seeds(self) -> List[int]:
        """Retorna lista plana de todas las seeds"""
        seeds = []
        for start, end in self.seed_ranges:
            seeds.extend(range(start, end + 1))
        return seeds


def get_default_config() -> ExperimentConfig:
    """Retorna configuración por defecto para experimentos"""
    return ExperimentConfig()


def get_ablation_estimator_config() -> ExperimentConfig:
    """Configuración para ablación de estimación"""
    config = ExperimentConfig(name="ablation_estimator")
    return config


def get_ablation_control_config() -> ExperimentConfig:
    """Configuración para ablación de control"""
    config = ExperimentConfig(name="ablation_control")
    return config
