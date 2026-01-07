#!/usr/bin/env python3
"""
Módulo de configuración del proyecto SWARM-BARC-VCA
"""

from .params import (
    UWBParams,
    IMUParams,
    NeuralParams,
    ControlParams,
    ExperimentConfig,
    get_default_config,
    get_ablation_estimator_config,
    get_ablation_control_config,
)

__all__ = [
    'UWBParams',
    'IMUParams',
    'NeuralParams',
    'ControlParams',
    'ExperimentConfig',
    'get_default_config',
    'get_ablation_estimator_config',
    'get_ablation_control_config',
]
