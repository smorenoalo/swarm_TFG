#!/usr/bin/env python3
"""
================================================================================
VALIDACIÓN DEL PROYECTO SWARM-BARC-VCA
================================================================================

Script para verificar que todas las dependencias están instaladas y que
los módulos funcionan correctamente antes de ejecutar experimentos.

Uso:
    python scripts/validate_setup.py
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check_dependencies():
    """Verifica que todas las dependencias están instaladas"""
    print("\n📦 Verificando dependencias...")
    
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'torch': 'torch',
        'filterpy': 'filterpy',
        'tqdm': 'tqdm',
    }
    
    missing = []
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - NO INSTALADO")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Faltan dependencias: {', '.join(missing)}")
        print("   Instala con: pip install -r requirements.txt")
        return False
    
    print("   Todas las dependencias instaladas")
    return True


def check_modules():
    """Verifica que todos los módulos del proyecto se importan correctamente"""
    print("\n Verificando módulos del proyecto...")
    
    modules = [
        ('configs.params', 'Configuración'),
        ('swarm_core.sensors', 'Sensores'),
        ('swarm_core.estimator', 'Estimador UKF'),
        ('swarm_core.gat_network', 'Red GAT'),
        ('swarm_core.trajectories', 'Trayectorias'),
        ('swarm_core.controller', 'Controlador'),
        ('swarm_core.training', 'Entrenamiento'),
        ('experiments.harness', 'Harness experimentos'),
        ('analysis.metrics', 'Métricas'),
        ('analysis.visualization', 'Visualización'),
    ]
    
    errors = []
    for module, name in modules:
        try:
            __import__(module)
            print(f"   ✓ {name} ({module})")
        except Exception as e:
            print(f"   ✗ {name} ({module}) - ERROR: {e}")
            errors.append((module, str(e)))
    
    if errors:
        print(f"\n⚠️  {len(errors)} módulo(s) con errores")
        return False
    
    print("   Todos los módulos importados correctamente")
    return True


def check_basic_functionality():
    """Verifica funcionalidad básica de los componentes"""
    print("\n Verificando funcionalidad básica...")
    
    import numpy as np
    
    # 1. Config
    try:
        from configs.params import get_default_config
        config = get_default_config()
        assert config.N > 0
        assert config.dt > 0
        print("   ✓ Configuración carga correctamente")
    except Exception as e:
        print(f"   ✗ Configuración - ERROR: {e}")
        return False
    
    # 2. Sensores
    try:
        from swarm_core.sensors import IMUSensor, UWBSensor
        imu = IMUSensor(config, agent_id=0, seed=42)
        meas = imu.measure(np.array([0.0, 0.0]))
        assert meas.shape == (2,)
        print("   ✓ Sensores funcionan correctamente")
    except Exception as e:
        print(f"   ✗ Sensores - ERROR: {e}")
        return False
    
    # 3. Estimador
    try:
        from swarm_core.estimator import UKFEstimator
        ukf = UKFEstimator(config, agent_id=1, seed=42)
        ukf.set_known_position(np.array([1.0, 0.0]))
        assert ukf.state is not None
        print("   ✓ Estimador UKF funciona correctamente")
    except Exception as e:
        print(f"   ✗ Estimador - ERROR: {e}")
        return False
    
    # 4. Trayectorias
    try:
        from swarm_core.trajectories import TrajectoryGenerator
        traj = TrajectoryGenerator(scenario='spiral', seed=42)
        pos = traj.get_position(0.0)
        assert pos.shape == (2,)
        print("   ✓ Generador de trayectorias funciona")
    except Exception as e:
        print(f"   ✗ Trayectorias - ERROR: {e}")
        return False
    
    # 5. Red GAT
    try:
        from swarm_core.gat_network import create_gat_model
        model = create_gat_model(config)
        assert model is not None
        print("   ✓ Red GAT se crea correctamente")
    except Exception as e:
        print(f"   ✗ Red GAT - ERROR: {e}")
        return False
    
    # 6. Controlador
    try:
        from swarm_core.controller import SwarmController
        ctrl = SwarmController(
            params=config,
            scenario='station_keeping',
            seed=42
        )
        step = ctrl.step()
        assert step is not None
        print("   ✓ Controlador ejecuta pasos correctamente")
    except Exception as e:
        print(f"   ✗ Controlador - ERROR: {e}")
        return False
    
    print("   Funcionalidad básica verificada")
    return True


def check_mini_experiment():
    """Ejecuta un mini-experimento de prueba"""
    print("\n Ejecutando mini-experimento...")
    
    try:
        from swarm_core.controller import SwarmController
        from swarm_core.estimator import EstimatorMode
        from configs.params import get_default_config
        import numpy as np
        
        config = get_default_config()
        
        # Crear controlador
        ctrl = SwarmController(
            params=config,
            scenario='spiral',
            seed=42,
            estimator_mode=EstimatorMode.IMU_UWB,
            use_ia=False
        )
        
        # Ejecutar 100 pasos (1 segundo)
        loc_errors = []
        form_errors = []
        
        for _ in range(100):
            step = ctrl.step()
            loc_errors.extend(step.localization_errors[1:])  # Excluir líder
            form_errors.append(step.formation_error)
        
        mean_loc = np.mean(loc_errors) * 100  # en cm
        mean_form = np.mean(form_errors) * 100  # en cm
        
        print(f"   • 100 pasos ejecutados")
        print(f"   • Error localización medio: {mean_loc:.1f} cm")
        print(f"   • Error formación medio: {mean_form:.1f} cm")
        print("   ✓ Mini-experimento completado")
        return True
        
    except Exception as e:
        print(f"   ✗ Mini-experimento fallido: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("Validación del Proyecto")
    print("="*70)
    
    all_ok = True
    
    # Verificar dependencias
    if not check_dependencies():
        all_ok = False
    
    # Verificar módulos
    if not check_modules():
        all_ok = False
    
    # Verificar funcionalidad (solo si módulos OK)
    if all_ok:
        if not check_basic_functionality():
            all_ok = False
    
    # Mini-experimento (solo si todo OK)
    if all_ok:
        if not check_mini_experiment():
            all_ok = False
    
    # Resumen final
    print("\n" + "="*70)
    if all_ok:
        print("✓ VALIDACIÓN COMPLETADA - Todo funciona correctamente")
        print("\nSiguiente paso:")
        print("  python scripts/run_experiments.py --quick")
        print("\nPara experimentos completos:")
        print("  python scripts/run_experiments.py --full --save-timeseries")
    else:
        print("✗ VALIDACIÓN FALLIDA - Revisar errores arriba")
        print("\nPasos recomendados:")
        print("  1. pip install -r requirements.txt")
        print("  2. Verificar instalación de PyTorch")
        print("  3. Ejecutar este script de nuevo")
    print("="*70 + "\n")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
