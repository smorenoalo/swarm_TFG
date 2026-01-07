#!/usr/bin/env python3
"""
================================================================================
VISUALIZACIÓN DE RESULTADOS
================================================================================

Genera las figuras:
- CDF de error de localización
- Error de formación vs tiempo
- Boxplot de mejora por seed
- Overlay de trayectorias GT vs estimada
- Análisis curvas vs rectas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Colores consistentes
COLORS = {
    'imu_only': '#e74c3c',    # Rojo
    'uwb_only': '#3498db',    # Azul
    'imu_uwb': '#2ecc71',     # Verde
    'classic': '#95a5a6',     # Gris
    'gat': '#9b59b6',         # Púrpura
}

LABELS = {
    'imu_only': 'IMU-only',
    'uwb_only': 'UWB-only',
    'imu_uwb': 'IMU+UWB',
    'classic': 'Clásico',
    'gat': 'Clásico+GAT',
}


def plot_localization_cdf(
    metrics_dict: Dict[str, List[dict]],
    scenarios: List[str] = None,
    output_path: str = None,
    figsize: Tuple[float, float] = (12, 5)
) -> plt.Figure:
    """
    Genera figura CDF de error de localización.
    
    Compara IMU vs UWB vs IMU+UWB para escenarios seleccionados.
    """
    # Auto-detectar escenarios disponibles si no se especifican
    if scenarios is None:
        available_scenarios = set()
        for method_metrics in metrics_dict.values():
            for m in method_metrics:
                if isinstance(m, dict) and 'scenario' in m:
                    available_scenarios.add(m['scenario'])
        scenarios = sorted(list(available_scenarios))
        if not scenarios:
            print("No hay escenarios disponibles en los datos")
            return None
    
    # Filtrar solo escenarios con datos
    scenarios_with_data = []
    for scenario in scenarios:
        has_data = False
        for method_metrics in metrics_dict.values():
            for m in method_metrics:
                if isinstance(m, dict) and m.get('scenario') == scenario:
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            scenarios_with_data.append(scenario)
    
    if not scenarios_with_data:
        print("No hay datos para los escenarios especificados")
        return None
    
    scenarios = scenarios_with_data
    
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6*len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]
    
    for ax, scenario in zip(axes, scenarios):
        for method in ['imu_only', 'uwb_only', 'imu_uwb']:
            if method not in metrics_dict:
                continue
            
            # Extraer errores de localización
            errors = []
            for m in metrics_dict[method]:
                if m['scenario'] == scenario:
                    errors.append(m['loc_rmse'] * 100)  # Convertir a cm
            
            if not errors:
                continue
            
            # Calcular CDF
            errors = np.array(errors)
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            
            ax.plot(sorted_errors, cdf, 
                   color=COLORS[method], 
                   label=LABELS[method],
                   linewidth=2)
        
        ax.set_xlabel('Error de Localización RMSE (cm)')
        ax.set_ylabel('Probabilidad Acumulada (CDF)')
        ax.set_title(f'Escenario: {scenario.replace("_", " ").title()}')
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Función de Distribución Acumulada del Error de Localización', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        print(f"Figura guardada: {output_path}")
    
    return fig


def plot_formation_error_vs_time(
    timeseries_dir: str,
    scenario: str = 'lawnmower',
    seeds: List[int] = None,
    output_path: str = None,
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Genera figura de error de formación vs tiempo.
    
    Compara Clásico vs GAT en un escenario.
    """
    ts_path = Path(timeseries_dir)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for method in ['classic', 'gat']:
        pattern = f"{scenario}_{method}_seed*.npz"
        files = list(ts_path.glob(pattern))
        
        if not files:
            continue
        
        # Agregar todas las series
        all_times = []
        all_errors = []
        
        for file in files:
            if seeds is not None:
                seed = int(file.stem.split('seed')[-1])
                if seed not in seeds:
                    continue
            
            data = np.load(file)
            times = data['times']
            gt_pos = data['gt_pos']
            targets = data['targets']
            
            # Calcular RMSE de formación por paso
            form_errors = []
            for t_idx in range(len(times)):
                err = np.sqrt(np.mean([
                    np.linalg.norm(gt_pos[t_idx, i] - targets[t_idx, i])**2
                    for i in range(gt_pos.shape[1])
                ])) * 100  # cm
                form_errors.append(err)
            
            all_times.append(times)
            all_errors.append(form_errors)
        
        if not all_errors:
            continue
        
        # Calcular media y percentiles
        # Primero interpolar a tiempos comunes
        t_common = np.linspace(0, max(t[-1] for t in all_times), 500)
        errors_interp = []
        
        for times, errors in zip(all_times, all_errors):
            interp_err = np.interp(t_common, times, errors)
            errors_interp.append(interp_err)
        
        errors_interp = np.array(errors_interp)
        mean_err = np.mean(errors_interp, axis=0)
        p25 = np.percentile(errors_interp, 25, axis=0)
        p75 = np.percentile(errors_interp, 75, axis=0)
        
        # Plotear
        ax.plot(t_common, mean_err, 
               color=COLORS[method], 
               label=LABELS[method],
               linewidth=2)
        ax.fill_between(t_common, p25, p75, 
                        color=COLORS[method], alpha=0.2)
    
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error de Formación RMSE (cm)')
    ax.set_title(f'Evolución del Error de Formación - {scenario.replace("_", " ").title()}')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        print(f"Figura guardada: {output_path}")
    
    return fig


def plot_improvement_boxplot(
    metrics_dict: Dict[str, List[dict]],
    metric: str = 'form_rmse',
    scenarios: List[str] = None,
    output_path: str = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Genera boxplot de mejora (ΔRMSE) por seed.
    
    Muestra distribución de la mejora para validar consistencia.
    """
    import pandas as pd
    
    # Auto-detectar escenarios disponibles si no se especifican
    if scenarios is None:
        available_scenarios = set()
        for method_metrics in metrics_dict.values():
            for m in method_metrics:
                if isinstance(m, dict) and 'scenario' in m:
                    available_scenarios.add(m['scenario'])
        scenarios = sorted(list(available_scenarios))
    
    if not scenarios:
        print("No hay escenarios disponibles")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    improvements_by_scenario = []
    valid_scenarios = []
    
    for scenario in scenarios:
        # Obtener valores por seed
        classic_by_seed = {}
        gat_by_seed = {}
        
        for m in metrics_dict.get('classic', []):
            if isinstance(m, dict) and m.get('scenario') == scenario:
                classic_by_seed[m['seed']] = m[metric] * 100  # cm
        
        for m in metrics_dict.get('gat', []):
            if isinstance(m, dict) and m.get('scenario') == scenario:
                gat_by_seed[m['seed']] = m[metric] * 100  # cm
        
        # Calcular diferencias pareadas
        improvements = []
        for seed in classic_by_seed:
            if seed in gat_by_seed:
                diff = classic_by_seed[seed] - gat_by_seed[seed]
                improvements.append(diff)
        
        # Solo añadir si hay datos
        if improvements:
            improvements_by_scenario.append(improvements)
            valid_scenarios.append(scenario)
    
    if not valid_scenarios:
        print("No hay datos de mejora para ningún escenario")
        return None
    
    # Crear boxplot solo con escenarios válidos
    bp = ax.boxplot(improvements_by_scenario, 
                    labels=[s.replace('_', '\n') for s in valid_scenarios],
                    patch_artist=True)
    
    # Colorear boxplots
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['gat'])
        patch.set_alpha(0.6)
    
    # Línea en y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Escenario de Trayectoria')
    ax.set_ylabel('Mejora ΔRMSE (cm)\n(Positivo = GAT mejor)')
    ax.set_title('Distribución de Mejora del Control GAT por Escenario')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir estadísticas
    for i, improvements in enumerate(improvements_by_scenario):
        if improvements:
            mean_imp = np.mean(improvements)
            pct_positive = np.mean(np.array(improvements) > 0) * 100
            ax.annotate(f'{mean_imp:+.1f}cm\n({pct_positive:.0f}%+)',
                       xy=(i + 1, mean_imp),
                       xytext=(i + 1.3, mean_imp),
                       fontsize=9,
                       ha='left')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        print(f"Figura guardada: {output_path}")
    
    return fig


def plot_trajectory_overlay(
    timeseries_path: str,
    follower_ids: List[int] = None,
    output_path: str = None,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """
    Genera overlay de trayectorias GT vs estimada.
    
    Muestra líder y seguidores seleccionados.
    """
    if follower_ids is None:
        follower_ids = [1, 5]  # Dos seguidores representativos
    
    data = np.load(timeseries_path)
    times = data['times']
    gt_pos = data['gt_pos']
    est_pos = data['est_pos']
    targets = data['targets']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Subplot 1: Vista XY de trayectorias
    ax1 = axes[0]
    
    # Líder
    ax1.plot(gt_pos[:, 0, 0], gt_pos[:, 0, 1], 
            'k-', linewidth=2, label='Líder (GT)')
    ax1.scatter(gt_pos[0, 0, 0], gt_pos[0, 0, 1], 
               color='green', s=100, marker='o', zorder=5)
    ax1.scatter(gt_pos[-1, 0, 0], gt_pos[-1, 0, 1], 
               color='red', s=100, marker='s', zorder=5)
    
    # Seguidores
    colors = plt.cm.tab10(np.linspace(0, 1, len(follower_ids)))
    
    for idx, (fid, color) in enumerate(zip(follower_ids, colors)):
        ax1.plot(gt_pos[:, fid, 0], gt_pos[:, fid, 1], 
                '-', color=color, linewidth=1.5, 
                label=f'Seguidor {fid} (GT)')
        ax1.plot(est_pos[:, fid, 0], est_pos[:, fid, 1], 
                '--', color=color, linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('Posición X (m)')
    ax1.set_ylabel('Posición Y (m)')
    ax1.set_title('Trayectorias en el Plano XY')
    ax1.legend(loc='best', fontsize=9)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Error de estimación vs tiempo
    ax2 = axes[1]
    
    for idx, (fid, color) in enumerate(zip(follower_ids, colors)):
        errors = np.linalg.norm(gt_pos[:, fid, :] - est_pos[:, fid, :], axis=1) * 100
        ax2.plot(times, errors, '-', color=color, linewidth=1.5,
                label=f'Seguidor {fid}')
    
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Error de Localización (cm)')
    ax2.set_title('Error de Estimación por Seguidor')
    ax2.legend(loc='upper right')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        print(f"Figura guardada: {output_path}")
    
    return fig


def plot_ia_activation_analysis(
    timeseries_dir: str,
    scenario: str = 'lawnmower',
    output_path: str = None,
    figsize: Tuple[float, float] = (14, 8)
) -> plt.Figure:
    """
    Analiza cuándo se activa la IA y su impacto.
    
    Muestra:
    - Tasa de activación vs tiempo
    - Confianza media vs tiempo
    - Correlación con curvatura
    """
    ts_path = Path(timeseries_dir)
    pattern = f"{scenario}_gat_seed*.npz"
    files = list(ts_path.glob(pattern))
    
    if not files:
        print(f"No se encontraron archivos para {scenario}")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Recolectar datos
    all_times = []
    all_ia_allowed = []
    all_confidence = []
    all_deltaF_norms = []
    
    for file in files[:5]:  # Limitar a 5 seeds para claridad
        data = np.load(file)
        times = data['times']
        ia_allowed = data['ia_allowed']
        confidence = data['confidence']
        deltaF = data['deltaF']
        
        all_times.append(times)
        all_ia_allowed.append(ia_allowed)
        all_confidence.append(np.mean(confidence[:, 1:], axis=1))  # Media sobre seguidores
        
        deltaF_norm = np.mean(np.linalg.norm(deltaF[:, 1:, :], axis=2), axis=1)
        all_deltaF_norms.append(deltaF_norm)
    
    # Interpolar a tiempos comunes
    t_common = np.linspace(0, 60, 500)
    
    # Plot 1: Tasa de activación
    ax1 = axes[0, 0]
    for times, ia in zip(all_times, all_ia_allowed):
        ia_interp = np.interp(t_common, times, ia.astype(float))
        ax1.plot(t_common, ia_interp, alpha=0.5)
    
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Estado de Activación (0/1)')
    ax1.set_title('Activación de la Red GAT')
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confianza media
    ax2 = axes[0, 1]
    for times, conf in zip(all_times, all_confidence):
        conf_interp = np.interp(t_common, times, conf)
        ax2.plot(t_common, conf_interp, alpha=0.5)
    
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Confianza Media')
    ax2.set_title('Confianza de la Red GAT')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Magnitud del delta
    ax3 = axes[1, 0]
    for times, df_norm in zip(all_times, all_deltaF_norms):
        df_interp = np.interp(t_common, times, df_norm)
        ax3.plot(t_common, df_interp, alpha=0.5)
    
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Magnitud ||ΔF|| (m/s²)')
    ax3.set_title('Corrección Aplicada por la IA')
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histograma de confianza cuando IA activa
    ax4 = axes[1, 1]
    active_confs = []
    for ia, conf in zip(all_ia_allowed, all_confidence):
        active_confs.extend(conf[ia])
    
    if active_confs:
        ax4.hist(active_confs, bins=30, color=COLORS['gat'], alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Nivel de Confianza')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Confianza (IA Activa)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Análisis del Control GAT - {scenario.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
        print(f"Figura guardada: {output_path}")
    
    return fig


def generate_all_figures(
    results_dir: str = "results",
    output_dir: str = None
):
    """
    Genera todas las figuras del TFG.
    """
    results_path = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_path / "figures"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERANDO FIGURAS")
    print("="*60)
    
    # Cargar métricas
    est_metrics_path = results_path / "ablation_estimator" / "metrics.json"
    ctrl_metrics_path = results_path / "ablation_control" / "metrics.json"
    
    # Auto-detectar escenarios disponibles
    available_scenarios = set()
    if est_metrics_path.exists():
        with open(est_metrics_path) as f:
            est_metrics = json.load(f)
        for method_metrics in est_metrics.values():
            for m in method_metrics:
                if isinstance(m, dict) and 'scenario' in m:
                    available_scenarios.add(m['scenario'])
    
    if ctrl_metrics_path.exists():
        with open(ctrl_metrics_path) as f:
            ctrl_metrics = json.load(f)
        for method_metrics in ctrl_metrics.values():
            for m in method_metrics:
                if isinstance(m, dict) and 'scenario' in m:
                    available_scenarios.add(m['scenario'])
    
    scenarios = sorted(list(available_scenarios))
    print(f"Escenarios detectados: {scenarios}")
    
    # Figura 1: CDF de localización
    if est_metrics_path.exists():
        print("\n[1/5] CDF de localización...")
        plot_localization_cdf(
            est_metrics,
            scenarios=None,  # Auto-detectar
            output_path=str(output_path / "fig1_localization_cdf.png")
        )
    
    # Figura 2: Error de formación vs tiempo
    ctrl_ts_dir = results_path / "ablation_control"
    if ctrl_ts_dir.exists() and list(ctrl_ts_dir.glob("*.npz")):
        print("\n[2/5] Error de formación vs tiempo...")
        for scenario in scenarios:
            ts_files = list(ctrl_ts_dir.glob(f"{scenario}_*.npz"))
            if ts_files:
                plot_formation_error_vs_time(
                    str(ctrl_ts_dir),
                    scenario=scenario,
                    output_path=str(output_path / f"fig2_formation_error_{scenario}.png")
                )
    
    # Figura 3: Boxplot de mejora
    if ctrl_metrics_path.exists():
        print("\n[3/5] Boxplot de mejora...")
        plot_improvement_boxplot(
            ctrl_metrics,
            scenarios=None,  # Auto-detectar
            output_path=str(output_path / "fig3_improvement_boxplot.png")
        )
    
    # Figura 4: Overlay de trayectoria (buscar cualquier escenario disponible)
    sample_ts = None
    for scenario in scenarios:
        candidates = list(ctrl_ts_dir.glob(f"{scenario}_imu_uwb_seed*.npz"))
        if candidates:
            sample_ts = candidates[0]
            break
    
    if sample_ts:
        print("\n[4/5] Overlay de trayectoria...")
        plot_trajectory_overlay(
            str(sample_ts),
            output_path=str(output_path / "fig4_trajectory_overlay.png")
        )
    
    # Figura 5: Análisis de IA (buscar cualquier escenario disponible)
    gat_files = list(ctrl_ts_dir.glob("*_gat_*.npz"))
    if gat_files:
        print("\n[5/5] Análisis de IA...")
        # Detectar escenario del primer archivo
        first_gat = gat_files[0].stem
        scenario_for_ia = first_gat.split('_gat_')[0]
        plot_ia_activation_analysis(
            str(ctrl_ts_dir),
            scenario=scenario_for_ia,
            output_path=str(output_path / "fig5_ia_analysis.png")
        )
    
    print(f"\nFiguras guardadas en: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generación de figuras')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directorio con resultados')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio para figuras')
    
    args = parser.parse_args()
    
    generate_all_figures(args.results_dir, args.output_dir)
