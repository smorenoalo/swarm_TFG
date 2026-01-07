# Control de Enjambre con fusion IMU (simplificada)+ UWB y GAT

## Descripción

Este proyecto implementa un framework experimental para estudiar:

1. **Ablación de Estimación**: Comparación de modos de localización (IMU-only, UWB-only, IMU+UWB fusionado con UKF)
2. **Ablación de Control**: Evaluación de mejora de control clásico PID con redes Graph Attention Networks (GAT)

## Instalación

### Requisitos

- Python 3.8+
- PyTorch 1.10+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- FilterPy (para UKF)

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/smorenoalo/swarm_tfg.git
cd swarm_tfg

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Test Rápido

```bash
# Verificar instalación con test rápido
python scripts/run_experiments.py --quick
```

### Experimentos Completos

```bash
# Pipeline completo (Bloque 1 + Bloque 2 + Análisis)
python scripts/run_experiments.py --full --save-timeseries

# Solo ablación de estimación
python scripts/run_experiments.py --estimator

# Solo ablación de control
python scripts/run_experiments.py --control

# Solo generar tablas y figuras
python scripts/run_experiments.py --analysis
```

### Opciones Disponibles

| Opción | Descripción |
|--------|-------------|
| `--full` | Pipeline completo |
| `--quick` | Test rápido (1 seed, 1 escenario) |
| `--estimator` | Solo ablación de estimación |
| `--control` | Solo ablación de control |
| `--train` | Solo entrenar red GAT |
| `--analysis` | Solo análisis de resultados |
| `--output-dir DIR` | Directorio de salida |
| `--scenarios LIST` | Escenarios a ejecutar |
| `--seeds LIST` | Seeds específicas |
| `--duration SECS` | Duración de misiones |
| `--save-timeseries` | Guardar series temporales |
| `-v, --verbose` | Modo verbose |

## Experimentos

### Bloque 1: Ablación de Estimación

Compara tres modos de localización:
- **IMU-only**: Solo integración inercial (con drift)
- **UWB-only**: Solo ranging UWB (sin predicción)
- **IMU+UWB**: Fusión completa con UKF

### Bloque 2: Ablación de Control

Compara dos métodos de control:
- **Clásico**: Control PID estándar
- **Clásico+GAT**: PID con corrección neuronal

### Escenarios

| Escenario | Descripción |
|-----------|-------------|
| `spiral` | Espiral de Arquímedes, curvas continuas |
| `lawnmower` | Barrido rectangular con giros |
| `snake` | Serpenteo sinusoidal |
| `station_keeping` | Modo estáticon |

### Seeds

30 seeds pareadas:
- 5 runs × 6 bloques: [0-5], [10-15], [20-25], [30-35], [40-45]

## Métricas

### Localización
- RMSE, mediana, P95, worst follower
- Fracción >100cm
- Drift rate

### Formación
- RMSE, P95
- Tiempo de convergencia (<30cm durante 2s)

### Esfuerzo de Control
- Aceleración media y P95
- Jerk (suavidad)

### IA
- Tasa de activación
- Confianza media
- Magnitud de corrección

## Resultados

Estructura de resultados:
```
results/
├── ablation_estimator/
│   ├── metrics.json           # Métricas agregadas
│   ├── spiral_imu_only_seed42.npz  # Series temporales
│   └── ...
├── ablation_control/
│   ├── metrics.json
│   └── ...
├── models/
│   └── gat_model.pth          # Modelo entrenado
├── tables/
│   ├── localization.tex       # Tabla 1
│   ├── control.tex            # Tabla 2
│   └── improvement.tex        # Tabla 3
└── figures/
    ├── cdf_localization.pdf   # Figura 1
    ├── formation_vs_time.pdf  # Figura 2
    └── ...
```


## Configuración

Modificar parámetros en `configs/params.py`:

```python
# Control PID
k_p = 1.5    # Ganancia proporcional
k_d = 0.8    # Ganancia derivativa
k_i = 0.1    # Ganancia integral

# UWB
uwb_los_std = 0.05      # 5cm en LOS
uwb_nlos_bias = 0.5     # 50cm bias NLOS

# IMU
imu_accel_noise = 0.1   # m/s²
imu_bias_std = 0.05     # m/s²

# GAT
gat_hidden_dim = 64
gat_heads = 4
gat_layers = 2
```

## Licencia

Copyright 2026 Samuel Moreno Alonso

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Autor

Samuel Moreno Alonso, 
Grado en Ingenieria de Tecnologías y Servicios de Telecomunicación, UOC

