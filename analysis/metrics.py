#!/usr/bin/env python3
"""
================================================================================
ANÁLISIS DE RESULTADOS
================================================================================

Funciones para:
- Agregación de métricas (media, std, mediana, IQR)
- Análisis pareado (diferencias por seed)
- Test estadísticos (Wilcoxon)
- Generación de tablas y resúmenes
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.harness import RunMetrics


def load_metrics_from_json(json_path: str) -> Dict[str, List[dict]]:
    """Carga métricas desde archivo JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def metrics_to_dataframe(metrics_dict: Dict[str, List[dict]]) -> pd.DataFrame:
    """Convierte métricas a DataFrame de pandas"""
    rows = []
    for method, metrics_list in metrics_dict.items():
        for m in metrics_list:
            m['method'] = method
            rows.append(m)
    return pd.DataFrame(rows)


def aggregate_by_scenario_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por escenario y método.
    
    Returns:
        DataFrame con media ± std para cada métrica
    """
    # Columnas numéricas a agregar
    numeric_cols = [
        'loc_rmse', 'loc_median', 'loc_p95', 'loc_worst_follower',
        'loc_frac_above_threshold', 'form_rmse', 'form_p95',
        'effort_mean', 'effort_p95', 'jerk_mean', 'jerk_p95',
        'ia_activation_rate', 'ia_confidence_mean', 'ia_deltaF_mean'
    ]
    
    # Filtrar columnas que existen
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    # Agrupar y calcular estadísticas
    grouped = df.groupby(['scenario', 'method'])[available_cols].agg(['mean', 'std', 'median'])
    
    return grouped


def paired_analysis(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = 'form_rmse'
) -> Dict[str, float]:
    """
    Realiza análisis pareado entre dos métodos.
    
    Calcula:
    - Diferencia media: mean(A) - mean(B)
    - % seeds donde A > B (mejora)
    - Test de Wilcoxon (si hay suficientes muestras)
    
    Args:
        df: DataFrame con métricas
        method_a: Método base (e.g., 'classic')
        method_b: Método a comparar (e.g., 'gat')
        metric: Métrica a comparar
        
    Returns:
        Diccionario con resultados del análisis
    """
    results = {}
    
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        
        # Obtener valores por seed
        values_a = scenario_df[scenario_df['method'] == method_a].set_index('seed')[metric]
        values_b = scenario_df[scenario_df['method'] == method_b].set_index('seed')[metric]
        
        # Alinear por seed
        common_seeds = values_a.index.intersection(values_b.index)
        a = values_a.loc[common_seeds].values
        b = values_b.loc[common_seeds].values
        
        # Diferencias
        diff = a - b  # Positivo = A peor que B
        
        # Estadísticas
        mean_diff = np.mean(diff)
        pct_improvement = np.mean(diff > 0) * 100  # % donde B es mejor
        
        # Wilcoxon (si hay suficientes muestras)
        p_value = None
        if len(diff) >= 5:
            try:
                _, p_value = stats.wilcoxon(a, b, alternative='two-sided')
            except ValueError:
                pass
        
        results[scenario] = {
            'mean_diff': mean_diff,
            'std_diff': np.std(diff),
            'pct_improvement': pct_improvement,
            'p_value': p_value,
            'n_pairs': len(diff),
            'mean_a': np.mean(a),
            'mean_b': np.mean(b)
        }
    
    return results


def generate_localization_table(
    metrics_dict: Dict[str, List[dict]],
    scenarios: List[str] = None,
    to_cm: bool = True
) -> pd.DataFrame:
    """
    Genera Tabla 1: Métricas de localización (IMU vs UWB vs IMU+UWB).
    
    Columnas: RMSE, Mediana, P95, Worst, Frac>100cm, Drift
    """
    df = metrics_to_dataframe(metrics_dict)
    
    if scenarios is None:
        scenarios = df['scenario'].unique()
    
    # Multiplicador para convertir a cm
    mult = 100 if to_cm else 1
    
    rows = []
    for scenario in scenarios:
        for method in ['imu_only', 'uwb_only', 'imu_uwb']:
            subset = df[(df['scenario'] == scenario) & (df['method'] == method)]
            
            if len(subset) == 0:
                continue
            
            row = {
                'Escenario': scenario,
                'Método': method.replace('_', '+').upper(),
                'RMSE (cm)': f"{subset['loc_rmse'].mean() * mult:.1f} ± {subset['loc_rmse'].std() * mult:.1f}",
                'Mediana (cm)': f"{subset['loc_median'].mean() * mult:.1f}",
                'P95 (cm)': f"{subset['loc_p95'].mean() * mult:.1f}",
                'Worst (cm)': f"{subset['loc_worst_follower'].mean() * mult:.1f}",
                'Frac>100cm': f"{subset['loc_frac_above_threshold'].mean() * 100:.1f}%"
            }
            
            # Drift solo para station_keeping
            if scenario == 'station_keeping':
                drift_values = subset['loc_drift'].dropna()
                if len(drift_values) > 0:
                    row['Drift (cm)'] = f"{drift_values.mean() * mult:.1f}"
                else:
                    row['Drift (cm)'] = 'N/A'
            else:
                row['Drift (cm)'] = '-'
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_control_table(
    metrics_dict: Dict[str, List[dict]],
    scenarios: List[str] = None,
    to_cm: bool = True
) -> pd.DataFrame:
    """
    Genera Tabla 2: Métricas de control (Clásico vs GAT).
    
    Columnas: RMSE_form, P95_form, Convergencia, mean|a|, P95|a|, IA_activa%, Confianza
    """
    df = metrics_to_dataframe(metrics_dict)
    
    if scenarios is None:
        scenarios = df['scenario'].unique()
    
    mult = 100 if to_cm else 1
    
    rows = []
    for scenario in scenarios:
        for method in ['classic', 'gat']:
            subset = df[(df['scenario'] == scenario) & (df['method'] == method)]
            
            if len(subset) == 0:
                continue
            
            row = {
                'Escenario': scenario,
                'Método': 'Clásico' if method == 'classic' else 'Clásico+GAT',
                'RMSE (cm)': f"{subset['form_rmse'].mean() * mult:.1f} ± {subset['form_rmse'].std() * mult:.1f}",
                'P95 (cm)': f"{subset['form_p95'].mean() * mult:.1f}",
                'Convergencia (s)': f"{subset['convergence_time'].mean():.1f}" if subset['convergence_time'].notna().any() else 'N/A',
                'Esfuerzo medio': f"{subset['effort_mean'].mean():.2f}",
                'Esfuerzo P95': f"{subset['effort_p95'].mean():.2f}",
            }
            
            # Métricas de IA solo para GAT
            if method == 'gat':
                ia_rate = subset['ia_activation_rate'].dropna()
                ia_conf = subset['ia_confidence_mean'].dropna()
                
                row['IA activa (%)'] = f"{ia_rate.mean() * 100:.1f}%" if len(ia_rate) > 0 else 'N/A'
                row['Confianza'] = f"{ia_conf.mean():.2f}" if len(ia_conf) > 0 else 'N/A'
            else:
                row['IA activa (%)'] = '-'
                row['Confianza'] = '-'
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_improvement_summary(
    metrics_dict: Dict[str, List[dict]],
    scenarios: List[str] = None
) -> pd.DataFrame:
    """
    Genera resumen de mejora GAT vs Clásico.
    
    Incluye diferencia media, % mejora, significancia estadística.
    """
    df = metrics_to_dataframe(metrics_dict)
    
    if scenarios is None:
        scenarios = df['scenario'].unique()
    
    rows = []
    for scenario in scenarios:
        classic = df[(df['scenario'] == scenario) & (df['method'] == 'classic')]
        gat = df[(df['scenario'] == scenario) & (df['method'] == 'gat')]
        
        if len(classic) == 0 or len(gat) == 0:
            continue
        
        # Calcular mejora
        classic_rmse = classic['form_rmse'].mean() * 100
        gat_rmse = gat['form_rmse'].mean() * 100
        improvement = classic_rmse - gat_rmse
        pct_improvement = (improvement / classic_rmse) * 100 if classic_rmse > 0 else 0
        
        # Análisis pareado
        paired = paired_analysis(df, 'classic', 'gat', 'form_rmse')
        scenario_paired = paired.get(scenario, {})
        
        row = {
            'Escenario': scenario,
            'RMSE Clásico (cm)': f"{classic_rmse:.1f}",
            'RMSE GAT (cm)': f"{gat_rmse:.1f}",
            'Mejora (cm)': f"{improvement:+.1f}",
            'Mejora (%)': f"{pct_improvement:+.1f}%",
            '% Seeds mejoran': f"{scenario_paired.get('pct_improvement', 0):.0f}%",
            'p-value': f"{scenario_paired.get('p_value', 'N/A'):.4f}" if scenario_paired.get('p_value') else 'N/A'
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_curve_vs_straight_analysis(
    timeseries_dir: str,
    scenario: str = 'lawnmower'
) -> Dict[str, Dict]:
    """
    Analiza rendimiento en curvas vs rectas.
    
    Muy útil para demostrar dónde ayuda la GAT.
    """
    from swarm_core.trajectories import TrajectoryGenerator
    
    ts_path = Path(timeseries_dir)
    
    results = {'classic': {'curves': [], 'straights': []},
               'gat': {'curves': [], 'straights': []}}
    
    for method in ['classic', 'gat']:
        pattern = f"{scenario}_{method}_seed*.npz"
        
        for file in ts_path.glob(pattern):
            data = np.load(file)
            times = data['times']
            gt_pos = data['gt_pos']
            targets = data['targets']
            
            # Clasificar cada paso como curva o recta
            for t_idx, t in enumerate(times):
                if t < 0:
                    continue
                    
                is_straight = TrajectoryGenerator.is_straight_segment(scenario, t)
                
                # Error de formación en este paso
                form_err = np.sqrt(np.mean([
                    np.linalg.norm(gt_pos[t_idx, i] - targets[t_idx, i])**2
                    for i in range(gt_pos.shape[1])
                ]))
                
                if is_straight:
                    results[method]['straights'].append(form_err)
                else:
                    results[method]['curves'].append(form_err)
    
    # Calcular estadísticas
    summary = {}
    for method in ['classic', 'gat']:
        curves = np.array(results[method]['curves'])
        straights = np.array(results[method]['straights'])
        
        summary[method] = {
            'curves_rmse': np.sqrt(np.mean(curves**2)) * 100 if len(curves) > 0 else None,
            'straights_rmse': np.sqrt(np.mean(straights**2)) * 100 if len(straights) > 0 else None,
            'curves_count': len(curves),
            'straights_count': len(straights)
        }
    
    return summary


def save_tables_to_latex(
    loc_table: pd.DataFrame,
    ctrl_table: pd.DataFrame,
    improvement_table: pd.DataFrame,
    output_dir: str = "results/tables"
):
    """Guarda tablas en formato LaTeX"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tabla de localización
    loc_latex = loc_table.to_latex(index=False, escape=False)
    with open(output_path / "table_localization.tex", 'w') as f:
        f.write(loc_latex)
    
    # Tabla de control
    ctrl_latex = ctrl_table.to_latex(index=False, escape=False)
    with open(output_path / "table_control.tex", 'w') as f:
        f.write(ctrl_latex)
    
    # Tabla de mejora
    imp_latex = improvement_table.to_latex(index=False, escape=False)
    with open(output_path / "table_improvement.tex", 'w') as f:
        f.write(imp_latex)
    
    print(f"Tablas LaTeX guardadas en: {output_path}")


def save_tables_to_csv(
    loc_table: pd.DataFrame,
    ctrl_table: pd.DataFrame,
    improvement_table: pd.DataFrame,
    output_dir: str = "results/tables"
):
    """Guarda tablas en formato CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loc_table.to_csv(output_path / "table_localization.csv", index=False)
    ctrl_table.to_csv(output_path / "table_control.csv", index=False)
    improvement_table.to_csv(output_path / "table_improvement.csv", index=False)
    
    print(f"Tablas CSV guardadas en: {output_path}")


def generate_all_tables(
    results_dir: str = "results",
    output_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera todas las tablas del TFG desde los resultados guardados.
    
    Args:
        results_dir: Directorio con resultados de experimentos
        output_dir: Directorio para guardar tablas (opcional)
        
    Returns:
        loc_table: Tabla de localización
        ctrl_table: Tabla de control
        improvement_table: Tabla de mejora
    """
    results_path = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_path / "tables"
    
    # Cargar métricas de ablación de estimación
    est_metrics_path = results_path / "ablation_estimator" / "metrics.json"
    if est_metrics_path.exists():
        est_metrics = load_metrics_from_json(str(est_metrics_path))
        loc_table = generate_localization_table(est_metrics)
    else:
        print(f"No se encontró: {est_metrics_path}")
        loc_table = pd.DataFrame()
    
    # Cargar métricas de ablación de control
    ctrl_metrics_path = results_path / "ablation_control" / "metrics.json"
    if ctrl_metrics_path.exists():
        ctrl_metrics = load_metrics_from_json(str(ctrl_metrics_path))
        ctrl_table = generate_control_table(ctrl_metrics)
        improvement_table = generate_improvement_summary(ctrl_metrics)
    else:
        print(f"No se encontró: {ctrl_metrics_path}")
        ctrl_table = pd.DataFrame()
        improvement_table = pd.DataFrame()
    
    # Guardar tablas
    if not loc_table.empty and not ctrl_table.empty:
        save_tables_to_csv(loc_table, ctrl_table, improvement_table, str(output_dir))
        save_tables_to_latex(loc_table, ctrl_table, improvement_table, str(output_dir))
    
    return loc_table, ctrl_table, improvement_table


def print_summary(results_dir: str = "results"):
    """Imprime resumen de resultados en consola"""
    loc_table, ctrl_table, improvement_table = generate_all_tables(results_dir)
    
    print("\n" + "="*70)
    print("TABLA 1: MÉTRICAS DE LOCALIZACIÓN")
    print("="*70)
    if not loc_table.empty:
        print(loc_table.to_string(index=False))
    
    print("\n" + "="*70)
    print("TABLA 2: MÉTRICAS DE CONTROL")
    print("="*70)
    if not ctrl_table.empty:
        print(ctrl_table.to_string(index=False))
    
    print("\n" + "="*70)
    print("RESUMEN DE MEJORA")
    print("="*70)
    if not improvement_table.empty:
        print(improvement_table.to_string(index=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de resultados')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directorio con resultados')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio para tablas')
    
    args = parser.parse_args()
    
    print_summary(args.results_dir)
