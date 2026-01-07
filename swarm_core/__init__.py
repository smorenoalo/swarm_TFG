#!/usr/bin/env python3
"""
================================================================================
SWARM-BARC-VCA: Core Module
================================================================================

Sistema de Control de Enjambre con Fusión Sensorial y GAT
"""

# Imports básicos (sin dependencias de torch)
from .sensors import IMUSensor, UWBSensor, LeaderGNSS, NLOSDetector
from .estimator import UKFEstimator, EstimatorMode, SwitchableEstimator
from .trajectories import (
    TrajectoryGenerator, RandomTrajectoryGenerator, 
    TrajectoryType, get_formation
)

# Imports que dependen de torch (condicionales)
try:
    import torch
    _HAS_TORCH = True
    from .gat_network import GraphControlNet, GATLayer, create_gat_model, load_gat_model
    from .controller import SwarmController, StepData
except ImportError:
    _HAS_TORCH = False
    GraphControlNet = None
    GATLayer = None
    create_gat_model = None
    load_gat_model = None
    SwarmController = None
    StepData = None

__all__ = [
    # Sensors
    'IMUSensor', 'UWBSensor', 'LeaderGNSS', 'NLOSDetector',
    # Estimator
    'UKFEstimator', 'EstimatorMode', 'SwitchableEstimator',
    # Trajectories
    'TrajectoryGenerator', 'RandomTrajectoryGenerator', 
    'TrajectoryType', 'get_formation',
]

# Añadir exports de torch si está disponible
if _HAS_TORCH:
    __all__.extend([
        # Neural Network
        'GraphControlNet', 'GATLayer', 'create_gat_model', 'load_gat_model',
        # Controller
        'SwarmController', 'StepData',
    ])

__version__ = '1.0.0'
