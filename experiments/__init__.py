#!/usr/bin/env python3
"""
Módulo de experimentos del proyecto SWARM-BARC-VCA
"""

from .harness import (
    RunMetrics,
    RunTimeSeries,
    run_single_experiment,
    run_ablation_estimator,
    run_ablation_control,
    run_full_experiment_pipeline,
)

__all__ = [
    'RunMetrics',
    'RunTimeSeries',
    'run_single_experiment',
    'run_ablation_estimator',
    'run_ablation_control',
    'run_full_experiment_pipeline',
]
