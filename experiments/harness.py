#!/usr/bin/env python3
"""
================================================================================
HARNESS DE EXPERIMENTOS
================================================================================

Ejecuta experimentos de ablación según el plan:
- Bloque 1: Ablación de estimación (IMU vs UWB vs IMU+UWB)
- Bloque 2: Ablación de control (Clásico vs Clásico+GAT)
"""

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm_core import (
    SwarmController, TrajectoryGenerator, get_formation, StepData
)
from swarm_core.training import load_trained_model
from configs.params import ControlParams, ExperimentConfig


@dataclass
class RunMetrics:
    """Métricas de una ejecución individual"""
    # Identificación (requeridos)
    scenario: str
    seed: int
    method: str
    estimator_mode: str
    
    # Métricas de localización (requeridos, en metros)
    loc_rmse: float
    loc_median: float
    loc_p95: float
    loc_worst_follower: float
    loc_frac_above_threshold: float
    
    # Métricas de formación (requeridos, en metros)
    form_rmse: float
    form_p95: float
    
    # Esfuerzo de control (requeridos)
    effort_mean: float
    effort_p95: float
    
    # --- Campos opcionales (con defaults) ---
    
    # Localización opcional
    loc_drift: Optional[float] = None  # Solo para station_keeping
    
    # Formación opcional
    convergence_time: Optional[float] = None  # Tiempo hasta convergencia
    
    # Esfuerzo opcional
    jerk_mean: Optional[float] = None
    jerk_p95: Optional[float] = None
    
    # Métricas de IA (solo si use_ia=True)
    ia_activation_rate: Optional[float] = None
    ia_confidence_mean: Optional[float] = None
    ia_deltaF_mean: Optional[float] = None
    ia_deltaF_p95: Optional[float] = None
    
    # Metadatos
    duration: float = 0.0
    total_steps: int = 0


@dataclass 
class RunTimeSeries:
    """Series temporales de una ejecución"""
    times: np.ndarray
    gt_pos: np.ndarray      # [T, N, 2]
    est_pos: np.ndarray     # [T, N, 2]
    targets: np.ndarray     # [T, N, 2]
    velocities: np.ndarray  # [T, N, 2]
    accelerations: np.ndarray  # [T, N, 2]
    ia_allowed: np.ndarray  # [T]
    confidence: np.ndarray  # [T, N]
    deltaF_applied: np.ndarray  # [T, N, 2]


def run_single_experiment(
    scenario: str,
    seed: int,
    params: ControlParams,
    duration: float = 60.0,
    estimator_mode: str = "imu_uwb",
    use_ia: bool = False,
    ctrl_state_dict: Optional[dict] = None,
    convergence_threshold: float = 0.3,
    convergence_hold: float = 2.0,
    loc_threshold: float = 1.0,
    save_timeseries: bool = True
) -> Tuple[RunMetrics, Optional[RunTimeSeries]]:
    """
    Ejecuta un experimento individual.
    
    Args:
        scenario: Tipo de trayectoria
        seed: Semilla para reproducibilidad
        params: Parámetros del sistema
        duration: Duración del test [s]
        estimator_mode: Modo del estimador
        use_ia: Si usar corrección GAT
        ctrl_state_dict: Pesos de la red
        convergence_threshold: Umbral de convergencia [m]
        convergence_hold: Tiempo para mantener convergencia [s]
        loc_threshold: Umbral de error de localización [m]
        save_timeseries: Si guardar series temporales
        
    Returns:
        metrics: Métricas calculadas
        timeseries: Series temporales (si save_timeseries=True)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Formación circular
    form = get_formation(params.N)
    
    # Crear controlador
    swarm = SwarmController(
        params, form,
        use_ia=use_ia,
        ctrl_state_dict=ctrl_state_dict,
        training_mode=False,
        estimator_mode=estimator_mode,
        seed=seed
    )
    
    # Posición inicial
    p0, _ = TrajectoryGenerator.get_ref(0, scenario)
    swarm.init(p0)
    
    # Preparar almacenamiento
    steps = int((duration + params.calibration_time) / params.dt)
    
    times = []
    gt_pos_list = []
    est_pos_list = []
    targets_list = []
    vel_list = []
    acc_list = []
    ia_allowed_list = []
    conf_list = []
    deltaF_list = []
    
    loc_errors_list = []  # Error de localización por paso
    form_errors_list = []  # Error de formación por paso
    
    # Simulación
    for k in range(steps):
        t_abs = k * params.dt
        t_mission = t_abs - params.calibration_time
        
        if t_mission < 0:
            ref_pos, ref_vel = p0, np.zeros(2)
        else:
            ref_pos, ref_vel = TrajectoryGenerator.get_ref(t_mission, scenario)
            if scenario != 'station_keeping':
                ref_pos = ref_pos + p0
        
        step_data = swarm.step(ref_pos, ref_vel, t_abs)
        
        if step_data is None:
            continue
        
        # Almacenar datos
        times.append(t_mission)
        gt_pos_list.append(step_data.gt_pos.copy())
        est_pos_list.append(step_data.est_pos.copy())
        targets_list.append(step_data.targets.copy())
        vel_list.append(step_data.velocities.copy())
        acc_list.append(step_data.accelerations.copy())
        ia_allowed_list.append(step_data.ia_allowed)
        conf_list.append(step_data.confidence.copy())
        deltaF_list.append(step_data.deltaF_applied.copy())
        
        # Calcular errores instantáneos
        # Error de localización (solo seguidores)
        loc_err = [
            np.linalg.norm(step_data.gt_pos[i] - step_data.est_pos[i])
            for i in range(1, params.N)
        ]
        loc_errors_list.append(loc_err)
        
        # Error de formación
        form_err = [
            np.linalg.norm(step_data.gt_pos[i] - step_data.targets[i])
            for i in range(params.N)
        ]
        form_errors_list.append(form_err)
    
    # Convertir a arrays
    times = np.array(times)
    gt_pos = np.stack(gt_pos_list)
    est_pos = np.stack(est_pos_list)
    targets = np.stack(targets_list)
    velocities = np.stack(vel_list)
    accelerations = np.stack(acc_list)
    ia_allowed = np.array(ia_allowed_list)
    confidence = np.stack(conf_list)
    deltaF_applied = np.stack(deltaF_list)
    
    loc_errors = np.array(loc_errors_list)  # [T, N-1]
    form_errors = np.array(form_errors_list)  # [T, N]
    
    # =========== CALCULAR MÉTRICAS ===========
    
    # --- Métricas de localización ---
    loc_all = loc_errors.flatten()
    loc_rmse = np.sqrt(np.mean(loc_all ** 2))
    loc_median = np.median(loc_all)
    loc_p95 = np.percentile(loc_all, 95)
    
    # Peor seguidor (RMSE por seguidor)
    loc_rmse_per_follower = [
        np.sqrt(np.mean(loc_errors[:, i] ** 2))
        for i in range(params.N - 1)
    ]
    loc_worst = max(loc_rmse_per_follower)
    
    # Fracción de tiempo sobre umbral
    loc_frac_above = np.mean(loc_all > loc_threshold)
    
    # Drift (solo para station_keeping)
    loc_drift = None
    if scenario == 'station_keeping' and len(times) > 0:
        # Drift = error final - error inicial
        initial_loc = np.mean(loc_errors[:50]) if len(loc_errors) > 50 else np.mean(loc_errors)
        final_loc = np.mean(loc_errors[-50:]) if len(loc_errors) > 50 else np.mean(loc_errors)
        loc_drift = final_loc - initial_loc
    
    # --- Métricas de formación ---
    form_all = form_errors.flatten()
    form_rmse = np.sqrt(np.mean(form_all ** 2))
    form_p95 = np.percentile(form_all, 95)
    
    # Tiempo de convergencia
    convergence_time = None
    form_rmse_per_step = np.sqrt(np.mean(form_errors ** 2, axis=1))
    
    below_threshold = form_rmse_per_step < convergence_threshold
    hold_steps = int(convergence_hold / params.dt)
    
    for t_idx in range(len(below_threshold) - hold_steps):
        if np.all(below_threshold[t_idx:t_idx + hold_steps]):
            convergence_time = times[t_idx]
            break
    
    # --- Esfuerzo de control ---
    acc_norms = np.linalg.norm(accelerations[:, 1:, :], axis=2).flatten()
    effort_mean = np.mean(acc_norms)
    effort_p95 = np.percentile(acc_norms, 95)
    
    # Jerk (suavidad)
    if len(accelerations) > 1:
        jerk = np.diff(accelerations, axis=0) / params.dt
        jerk_norms = np.linalg.norm(jerk[:, 1:, :], axis=2).flatten()
        jerk_mean = np.mean(jerk_norms)
        jerk_p95 = np.percentile(jerk_norms, 95)
    else:
        jerk_mean = None
        jerk_p95 = None
    
    # --- Métricas de IA ---
    ia_activation_rate = None
    ia_confidence_mean = None
    ia_deltaF_mean = None
    ia_deltaF_p95 = None
    
    if use_ia:
        ia_activation_rate = np.mean(ia_allowed)
        
        if np.any(ia_allowed):
            ia_steps = np.where(ia_allowed)[0]
            conf_during_ia = confidence[ia_steps, 1:]  # Solo seguidores
            ia_confidence_mean = np.mean(conf_during_ia)
            
            deltaF_norms = np.linalg.norm(deltaF_applied[ia_steps, 1:, :], axis=2).flatten()
            ia_deltaF_mean = np.mean(deltaF_norms)
            ia_deltaF_p95 = np.percentile(deltaF_norms, 95) if len(deltaF_norms) > 0 else 0.0
    
    # Construir objeto de métricas
    method = f"{'ia' if use_ia else 'classic'}_{estimator_mode}"
    
    metrics = RunMetrics(
        scenario=scenario,
        seed=seed,
        method=method,
        estimator_mode=estimator_mode,
        loc_rmse=loc_rmse,
        loc_median=loc_median,
        loc_p95=loc_p95,
        loc_worst_follower=loc_worst,
        loc_frac_above_threshold=loc_frac_above,
        loc_drift=loc_drift,
        form_rmse=form_rmse,
        form_p95=form_p95,
        convergence_time=convergence_time,
        effort_mean=effort_mean,
        effort_p95=effort_p95,
        jerk_mean=jerk_mean,
        jerk_p95=jerk_p95,
        ia_activation_rate=ia_activation_rate,
        ia_confidence_mean=ia_confidence_mean,
        ia_deltaF_mean=ia_deltaF_mean,
        ia_deltaF_p95=ia_deltaF_p95,
        duration=duration,
        total_steps=len(times)
    )
    
    # Series temporales
    timeseries = None
    if save_timeseries:
        timeseries = RunTimeSeries(
            times=times,
            gt_pos=gt_pos,
            est_pos=est_pos,
            targets=targets,
            velocities=velocities,
            accelerations=accelerations,
            ia_allowed=ia_allowed,
            confidence=confidence,
            deltaF_applied=deltaF_applied
        )
    
    return metrics, timeseries


def _run_single_experiment_wrapper(args):
    """Wrapper para ejecutar experimento en proceso separado."""
    (scenario, seed, params, duration, estimator_mode, use_ia, 
     ctrl_state_dict, convergence_threshold, convergence_hold, 
     loc_threshold, save_timeseries, method_name) = args
    
    try:
        metrics, timeseries = run_single_experiment(
            scenario=scenario,
            seed=seed,
            params=params,
            duration=duration,
            estimator_mode=estimator_mode,
            use_ia=use_ia,
            ctrl_state_dict=ctrl_state_dict,
            convergence_threshold=convergence_threshold,
            convergence_hold=convergence_hold,
            loc_threshold=loc_threshold,
            save_timeseries=save_timeseries
        )
        return (scenario, seed, method_name, use_ia, metrics, timeseries, None)
    except Exception as e:
        import traceback
        return (scenario, seed, method_name, use_ia, None, None, str(e))


def run_ablation_estimator(
    config: ExperimentConfig,
    output_dir: str = "results/ablation_estimator",
    save_timeseries: bool = False,
    n_workers: int = None
) -> Dict[str, List[RunMetrics]]:
    """
    Ejecuta Bloque 1: Ablación de estimación.
    
    Compara: IMU-only vs UWB-only vs IMU+UWB
    (mismo controlador clásico, sin IA)
    
    Args:
        config: Configuración del experimento
        output_dir: Directorio de salida
        save_timeseries: Si guardar series temporales
        n_workers: Número de procesos paralelos (None=automático)
        
    Returns:
        results: Diccionario con métricas por método
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    estimator_modes = ["imu_only", "uwb_only", "imu_uwb"]
    seeds = config.get_all_seeds()
    
    results = {mode: [] for mode in estimator_modes}
    
    print("="*70)
    print("BLOQUE 1: ABLACIÓN DE ESTIMACIÓN")
    print("="*70)
    print(f"Escenarios: {config.scenarios}")
    print(f"Seeds: {len(seeds)}")
    print(f"Modos de estimación: {estimator_modes}")
    
    # Determinar número de workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    total_runs = len(config.scenarios) * len(seeds) * len(estimator_modes)
    print(f"Total runs: {total_runs} | Workers: {n_workers}")
    print()
    
    # Preparar argumentos para todos los experimentos
    all_args = []
    for scenario in config.scenarios:
        for mode in estimator_modes:
            for seed in seeds:
                all_args.append((
                    scenario, seed, config.params, config.mission_duration,
                    mode, False, None,  # use_ia=False, ctrl_state_dict=None
                    config.form_convergence_cm / 100, config.convergence_hold_time,
                    config.loc_threshold_cm / 100, save_timeseries,
                    mode  # method_name = estimator_mode
                ))
    
    # Ejecutar en paralelo o secuencial
    if n_workers > 1:
        print(f"Ejecutando {total_runs} experimentos en paralelo...")
        start_time = time.time()
        
        with Pool(n_workers) as pool:
            results_list = []
            for i, result in enumerate(pool.imap_unordered(_run_single_experiment_wrapper, all_args)):
                results_list.append(result)
                if (i + 1) % 10 == 0 or (i + 1) == total_runs:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total_runs - i - 1) / rate if rate > 0 else 0
                    print(f"  Progreso: {i+1}/{total_runs} ({rate:.1f} runs/s, ETA: {eta:.0f}s)")
    else:
        print("Ejecutando en modo secuencial...")
        results_list = [_run_single_experiment_wrapper(args) for args in all_args]
    
    # Procesar resultados
    errors = []
    for scenario, seed, mode, use_ia, metrics, timeseries, error in results_list:
        if error:
            errors.append(f"{scenario}/{mode}/seed{seed}: {error}")
            continue
        
        results[mode].append(metrics)
        
        # Guardar timeseries si se requiere
        if save_timeseries and timeseries is not None:
            ts_path = output_path / f"{scenario}_{mode}_seed{seed}.npz"
            np.savez_compressed(
                ts_path,
                times=timeseries.times,
                gt_pos=timeseries.gt_pos,
                est_pos=timeseries.est_pos,
                targets=timeseries.targets
            )
    
    if errors:
        print(f"\n⚠️ {len(errors)} errores durante ejecución:")
        for e in errors[:5]:
            print(f"  - {e}")
    
    # Guardar métricas
    metrics_path = output_path / "metrics.json"
    metrics_dict = {
        mode: [asdict(m) for m in metrics_list]
        for mode, metrics_list in results.items()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return results


def run_ablation_control(
    config: ExperimentConfig,
    ctrl_state_dict: dict,
    output_dir: str = "results/ablation_control",
    save_timeseries: bool = False,
    n_workers: int = None
) -> Dict[str, List[RunMetrics]]:
    """
    Ejecuta Bloque 2: Ablación de control.
    
    Compara: Control clásico vs Control clásico + GAT
    (estimador IMU+UWB idéntico en ambos)
    
    Args:
        config: Configuración del experimento
        ctrl_state_dict: Pesos de la red GAT
        output_dir: Directorio de salida
        save_timeseries: Si guardar series temporales
        n_workers: Número de procesos paralelos (None=automático)
        
    Returns:
        results: Diccionario con métricas por método
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    methods = [("classic", False), ("gat", True)]
    seeds = config.get_all_seeds()
    
    results = {name: [] for name, _ in methods}
    
    print("="*70)
    print("BLOQUE 2: ABLACIÓN DE CONTROL")
    print("="*70)
    print(f"Escenarios: {config.scenarios}")
    print(f"Seeds: {len(seeds)}")
    print(f"Métodos: {[m[0] for m in methods]}")
    
    # Determinar número de workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    total_runs = len(config.scenarios) * len(seeds) * len(methods)
    print(f"Total runs: {total_runs} | Workers: {n_workers}")
    print()
    
    # Preparar argumentos para todos los experimentos
    all_args = []
    for scenario in config.scenarios:
        for method_name, use_ia in methods:
            for seed in seeds:
                all_args.append((
                    scenario, seed, config.params, config.mission_duration,
                    "imu_uwb", use_ia, 
                    ctrl_state_dict if use_ia else None,
                    config.form_convergence_cm / 100, config.convergence_hold_time,
                    config.loc_threshold_cm / 100, save_timeseries,
                    method_name  # "classic" o "gat"
                ))
    
    # Ejecutar en paralelo o secuencial
    if n_workers > 1:
        print(f"Ejecutando {total_runs} experimentos en paralelo...")
        start_time = time.time()
        
        with Pool(n_workers) as pool:
            results_list = []
            for i, result in enumerate(pool.imap_unordered(_run_single_experiment_wrapper, all_args)):
                results_list.append(result)
                if (i + 1) % 10 == 0 or (i + 1) == total_runs:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total_runs - i - 1) / rate if rate > 0 else 0
                    print(f"  Progreso: {i+1}/{total_runs} ({rate:.1f} runs/s, ETA: {eta:.0f}s)")
    else:
        print("Ejecutando en modo secuencial...")
        results_list = [_run_single_experiment_wrapper(args) for args in all_args]
    
    # Procesar resultados
    errors = []
    for scenario, seed, method_name, use_ia, metrics, timeseries, error in results_list:
        if error:
            errors.append(f"{scenario}/{method_name}/seed{seed}: {error}")
            continue
        
        results[method_name].append(metrics)
        
        # Guardar timeseries si se requiere
        if save_timeseries and timeseries is not None:
            ts_path = output_path / f"{scenario}_{method_name}_seed{seed}.npz"
            np.savez_compressed(
                ts_path,
                times=timeseries.times,
                gt_pos=timeseries.gt_pos,
                est_pos=timeseries.est_pos,
                targets=timeseries.targets,
                ia_allowed=timeseries.ia_allowed,
                confidence=timeseries.confidence,
                deltaF=timeseries.deltaF_applied
            )
    
    if errors:
        print(f"\n⚠️ {len(errors)} errores durante ejecución:")
        for e in errors[:5]:
            print(f"  - {e}")
    
    # Guardar métricas
    metrics_path = output_path / "metrics.json"
    metrics_dict = {
        name: [asdict(m) for m in metrics_list]
        for name, metrics_list in results.items()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return results


def run_full_experiment_pipeline(
    config: ExperimentConfig = None,
    model_path: str = None,
    output_dir: str = "results",
    train_if_needed: bool = True,
    save_timeseries: bool = False,
    verbose: bool = True,
    n_workers: int = None
) -> Dict[str, Dict]:
    """
    Ejecuta pipeline completo de experimentos.
    
    1. Entrena GAT si es necesario
    2. Ejecuta ablación de estimación
    3. Ejecuta ablación de control
    
    Args:
        config: Configuración del experimento
        model_path: Ruta del modelo (si existe)
        output_dir: Directorio de salida
        train_if_needed: Si entrenar si no hay modelo
        save_timeseries: Si guardar series temporales
        verbose: Si mostrar progreso detallado
        n_workers: Número de procesos paralelos (None=automático)
        
    Returns:
        all_results: Diccionario con todos los resultados
    """
    from swarm_core.training import full_training_pipeline
    
    if config is None:
        config = ExperimentConfig()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*70)
        print("PIPELINE COMPLETO DE EXPERIMENTOS")
        print("="*70)
        print(f"Configuración: {config.name}")
        print(f"Directorio de salida: {output_path}")
        if n_workers:
            print(f"Workers paralelos: {n_workers}")
        print()
    
    # Paso 1: Obtener modelo GAT
    if model_path and Path(model_path).exists():
        if verbose:
            print("[1/3] Cargando modelo existente...")
        ctrl_state_dict, delta_scale = load_trained_model(model_path, config.params)
    elif train_if_needed:
        if verbose:
            print("[1/3] Entrenando modelo GAT...")
        model_save_path = output_path / "models" / "gat_model.pt"
        ctrl_state_dict, delta_scale = full_training_pipeline(
            params=config.params,
            output_path=str(model_save_path),
            seed=0
        )
    else:
        raise ValueError("No hay modelo y train_if_needed=False")
    
    # Paso 2: Ablación de estimación
    if verbose:
        print("\n[2/3] Ejecutando ablación de estimación...")
    estimator_results = run_ablation_estimator(
        config,
        output_dir=str(output_path / "ablation_estimator"),
        save_timeseries=save_timeseries,
        n_workers=n_workers
    )
    
    # Paso 3: Ablación de control
    if verbose:
        print("\n[3/3] Ejecutando ablación de control...")
    control_results = run_ablation_control(
        config,
        ctrl_state_dict,
        output_dir=str(output_path / "ablation_control"),
        save_timeseries=save_timeseries,
        n_workers=n_workers
    )
    
    # Guardar configuración
    config_path = output_path / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'name': config.name,
            'scenarios': config.scenarios,
            'seeds': config.get_all_seeds(),
            'duration': config.mission_duration,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    if verbose:
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO")
        print("="*70)
    
    return {
        'estimator': estimator_results,
        'control': control_results
    }
