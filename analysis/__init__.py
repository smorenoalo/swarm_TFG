#!/usr/bin/env python3
"""
Módulo de análisis del proyecto 
"""

from .metrics import (
    load_metrics_from_json,
    metrics_to_dataframe,
    aggregate_by_scenario_method,
    paired_analysis,
    generate_localization_table,
    generate_control_table,
    generate_improvement_summary,
    compute_curve_vs_straight_analysis,
    save_tables_to_latex,
    save_tables_to_csv,
    generate_all_tables,
    print_summary,
)

from .visualization import (
    plot_localization_cdf,
    plot_formation_error_vs_time,
    plot_improvement_boxplot,
    plot_trajectory_overlay,
    plot_ia_activation_analysis,
    generate_all_figures,
)

__all__ = [
    # Metrics
    'load_metrics_from_json',
    'metrics_to_dataframe',
    'aggregate_by_scenario_method',
    'paired_analysis',
    'generate_localization_table',
    'generate_control_table',
    'generate_improvement_summary',
    'compute_curve_vs_straight_analysis',
    'save_tables_to_latex',
    'save_tables_to_csv',
    'generate_all_tables',
    'print_summary',
    # Visualization
    'plot_localization_cdf',
    'plot_formation_error_vs_time',
    'plot_improvement_boxplot',
    'plot_trajectory_overlay',
    'plot_ia_activation_analysis',
    'generate_all_figures',
]
