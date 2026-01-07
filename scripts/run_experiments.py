#!/usr/bin/env python3
"""
================================================================================
SCRIPT PRINCIPAL DE EJECUCIÓN DE EXPERIMENTOS
================================================================================

Ejecuta el pipeline completo de experimentos para el TFG:
1. Bloque 1: Ablación de estimación (IMU vs UWB vs IMU+UWB)
2. Bloque 2: Ablación de control (Clásico vs GAT)
3. Generación de tablas y figuras

Uso:
    python scripts/run_experiments.py --full           # Pipeline completo
    python scripts/run_experiments.py --estimator     # Solo ablación estimación
    python scripts/run_experiments.py --control       # Solo ablación control
    python scripts/run_experiments.py --analysis      # Solo análisis
    python scripts/run_experiments.py --quick         # Test rápido (1 seed)
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.params import ExperimentConfig, ControlParams
from experiments.harness import (
    run_ablation_estimator,
    run_ablation_control,
    run_full_experiment_pipeline
)
from analysis.metrics import generate_all_tables, print_summary
from analysis.visualization import generate_all_figures


def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Ejecuta experimentos de ablación para TFG de control de enjambre'
    )
    
    # Modos de ejecución
    parser.add_argument('--full', action='store_true',
                        help='Ejecutar pipeline completo')
    parser.add_argument('--estimator', action='store_true',
                        help='Solo ablación de estimación')
    parser.add_argument('--control', action='store_true',
                        help='Solo ablación de control')
    parser.add_argument('--train', action='store_true',
                        help='Solo entrenar GAT')
    parser.add_argument('--analysis', action='store_true',
                        help='Solo generar tablas y figuras')
    parser.add_argument('--quick', action='store_true',
                        help='Test rápido con 1 seed')
    
    # Configuración
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directorio de salida')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['spiral', 'lawnmower', 'snake', 'station_keeping'],
                        help='Escenarios a ejecutar')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds específicas (override config)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duración de cada experimento [s]')
    parser.add_argument('--save-timeseries', action='store_true',
                        help='Guardar series temporales completas')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Modo verbose')
    parser.add_argument('--workers', '-j', type=int, default=None,
                        help='Número de procesos paralelos (default: CPUs-1)')
    parser.add_argument('--sequential', action='store_true',
                        help='Forzar ejecución secuencial (sin paralelismo)')
    
    return parser.parse_args()


def create_config(args) -> ExperimentConfig:
    """Crea configuración basada en argumentos"""
    config = ExperimentConfig(
        scenarios=args.scenarios,
        mission_duration=args.duration
    )
    
    # Override seeds si se especifican
    if args.seeds:
        # Crear rangos ficticios que contengan las seeds
        config.seed_ranges = [(s, s) for s in args.seeds]
    
    # Modo quick: solo 1 seed
    if args.quick:
        config.seed_ranges = [(0, 0)]
        config.scenarios = ['spiral']  # Solo un escenario
        config.mission_duration = 20.0  # Más corto
    
    return config


def run_estimator_ablation(config: ExperimentConfig, output_dir: str,
                           save_timeseries: bool, n_workers: int = None):
    """Ejecuta ablación de estimación"""
    print("\n" + "="*70)
    print("BLOQUE 1: ABLACIÓN DE ESTIMACIÓN")
    print("="*70)
    print(f"Modos: IMU-only, UWB-only, IMU+UWB")
    print(f"Escenarios: {config.scenarios}")
    print(f"Seeds: {config.get_all_seeds()}")
    print(f"Total ejecuciones: {3 * len(config.scenarios) * len(config.get_all_seeds())}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    results = run_ablation_estimator(
        config=config,
        output_dir=output_dir,
        save_timeseries=save_timeseries,
        n_workers=n_workers
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Bloque 1 completado en {elapsed/60:.1f} minutos")
    print(f"  Resultados guardados en: {output_dir}/ablation_estimator/")
    
    return results


def run_control_ablation(config: ExperimentConfig, output_dir: str,
                         save_timeseries: bool, n_workers: int = None,
                         train_if_needed: bool = True):
    """Ejecuta ablación de control"""
    from swarm_core.training import full_training_pipeline
    
    print("\n" + "="*70)
    print("BLOQUE 2: ABLACIÓN DE CONTROL")
    print("="*70)
    print(f"Métodos: Clásico, Clásico+GAT")
    print(f"Escenarios: {config.scenarios}")
    print(f"Seeds: {config.get_all_seeds()}")
    print(f"Total ejecuciones: {2 * len(config.scenarios) * len(config.get_all_seeds())}")
    print("="*70 + "\n")
    
    # Entrenar o cargar modelo GAT
    model_path = Path(output_dir) / "models" / "gat_model.pt"
    if model_path.exists():
        print("Cargando modelo GAT existente...")
        ctrl_state_dict, _ = load_trained_model(str(model_path), config.params)
    elif train_if_needed:
        print("Entrenando modelo GAT...")
        ctrl_state_dict, _ = full_training_pipeline(
            params=config.params,
            output_path=str(model_path),
            seed=0
        )
    else:
        raise ValueError("No hay modelo y train_if_needed=False")
    
    start_time = time.time()
    
    results = run_ablation_control(
        config=config,
        ctrl_state_dict=ctrl_state_dict,
        output_dir=output_dir,
        save_timeseries=save_timeseries,
        n_workers=n_workers
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Bloque 2 completado en {elapsed/60:.1f} minutos")
    print(f"  Resultados guardados en: {output_dir}/ablation_control/")
    
    return results


def run_training_only(config: ExperimentConfig, output_dir: str, verbose: bool):
    """Solo entrena el modelo GAT"""
    from swarm_core.training import full_training_pipeline
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO DE RED GAT")
    print("="*70)
    
    start_time = time.time()
    
    model_path = full_training_pipeline(
        params=config.params,
        output_dir=output_dir,
        n_graphs=8000,
        epochs=100,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Entrenamiento completado en {elapsed/60:.1f} minutos")
    print(f"  Modelo guardado en: {model_path}")
    
    return model_path


def run_analysis_only(output_dir: str):
    """Solo genera tablas y figuras"""
    print("\n" + "="*70)
    print("ANÁLISIS Y GENERACIÓN DE RESULTADOS")
    print("="*70)
    
    # Verificar que existen datos
    est_dir = Path(output_dir) / 'ablation_estimator'
    ctrl_dir = Path(output_dir) / 'ablation_control'
    
    if not est_dir.exists() and not ctrl_dir.exists():
        print("❌ Error: No se encontraron resultados de experimentos")
        print(f"   Buscando en: {output_dir}")
        return
    
    # Generar tablas
    print("\n Generando tablas...")
    generate_all_tables(output_dir)
    print("   ✓ Tablas generadas en: {}/tables/".format(output_dir))
    
    # Generar figuras
    print("\n Generando figuras...")
    generate_all_figures(output_dir)
    print("   ✓ Figuras generadas en: {}/figures/".format(output_dir))
    
    # Imprimir resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    print_summary(output_dir)


def main():
    args = parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear configuración
    config = create_config(args)
    
    # Guardar configuración
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'scenarios': config.scenarios,
            'seeds': config.get_all_seeds(),
            'duration': config.mission_duration,
            'params': {
                'N': config.params.N,
                'dt': config.params.dt,
                'k_p': config.params.k_p,
                'k_d': config.params.k_d,
                'k_i': config.params.k_i
            }
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("Sistema de Control de Enjambre")
    print("Experimentos de Ablación para TFG")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output: {output_dir.absolute()}")
    print("="*70)
    
    start_total = time.time()
    
    # Determinar número de workers
    n_workers = 1 if args.sequential else args.workers
    
    try:
        if args.analysis:
            run_analysis_only(str(output_dir))
            
        elif args.train:
            run_training_only(config, str(output_dir), args.verbose)
            
        elif args.estimator:
            run_estimator_ablation(
                config, str(output_dir), args.save_timeseries, n_workers
            )
            run_analysis_only(str(output_dir))
            
        elif args.control:
            run_control_ablation(
                config, str(output_dir), args.save_timeseries, n_workers
            )
            run_analysis_only(str(output_dir))
            
        elif args.full or args.quick:
            # Pipeline completo
            run_full_experiment_pipeline(
                config=config,
                output_dir=str(output_dir),
                train_if_needed=True,
                save_timeseries=args.save_timeseries,
                verbose=args.verbose,
                n_workers=n_workers
            )
            run_analysis_only(str(output_dir))
            
        else:
            # Por defecto, mostrar ayuda
            print("\n No se especificó modo de ejecución.")
            print("   Usa --help para ver opciones disponibles.")
            print("\nEjemplos:")
            print("  python run_experiments.py --quick    # Test rápido")
            print("  python run_experiments.py --full     # Pipeline completo")
            print("  python run_experiments.py --analysis # Solo análisis")
            return
    
    except KeyboardInterrupt:
        print("\n\n Ejecución interrumpida por usuario")
        return
    except Exception as e:
        print(f"\n\n❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()
        return
    
    elapsed_total = time.time() - start_total
    print("\n" + "="*70)
    print(f"✓ EJECUCIÓN COMPLETADA")
    print(f"  Tiempo total: {elapsed_total/60:.1f} minutos")
    print(f"  Resultados en: {output_dir.absolute()}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
