#!/usr/bin/env python3
"""
================================================================================
ENTRENAMIENTO DE LA RED GAT
================================================================================

Funciones para:
- Recolección de datos de entrenamiento
- Entrenamiento de la red
- Validación y evaluación
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm_core import (
    SwarmController, RandomTrajectoryGenerator, 
    GraphControlNet, get_formation
)
from configs.params import ControlParams


def collect_training_data(params: ControlParams, num_graphs: int = 8000, 
                         seed: int = 0, verbose: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Recolecta datos de entrenamiento para la red GAT.
    
    Genera episodios con trayectorias aleatorias y recolecta:
    - Features de estado
    - Matriz de adyacencia
    - Delta de fuerza (F_ideal - F_base)
    
    Args:
        params: Parámetros del sistema
        num_graphs: Número de grafos a recolectar
        seed: Semilla para reproducibilidad
        verbose: Si mostrar progreso
        
    Returns:
        H: Features [G, N, 16]
        A: Adyacencias [G, N, N]
        dF: Deltas de fuerza [G, N, 2]
        rmse_arr: RMSEs por grafo [G]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    graphs_h, graphs_adj, graphs_dF, graphs_rmse = [], [], [], []
    graphs_collected = 0
    rmse_reset_threshold = 1.2
    store_every = 10
    max_graphs_per_episode = 150
    ep_idx = 0
    
    if verbose:
        print("  Recolectando datos...")

    while graphs_collected < num_graphs:
        ep_idx += 1
        ep_graphs = 0

        # Formación circular
        form = get_formation(params.N)

        swarm = SwarmController(
            params, form, use_ia=False, training_mode=True, seed=seed + ep_idx
        )

        # Trayectoria aleatoria
        v_max = np.random.uniform(1.0, 4.0)
        a_max = np.random.uniform(0.3, 2.0)
        change_T = np.random.uniform(0.5, 4.0)
        
        traj = RandomTrajectoryGenerator(
            v_max=v_max, a_max=a_max, change_T=change_T,
            seed=seed + ep_idx * 1000
        )
        
        p0, _ = traj.step(0.0, params.dt)
        swarm.init(p0)

        steps = int(40.0 / params.dt)

        for k in range(steps):
            t_abs = k * params.dt
            t_mission = t_abs - params.calibration_time

            if t_mission < 0:
                ref_pos, ref_vel = p0, np.zeros(2)
            else:
                ref_pos, ref_vel = traj.step(t_mission, params.dt)

            swarm.step(ref_pos, ref_vel, t_abs)

            if t_mission < 0:
                continue

            if len(swarm.hist_rmse) > 0:
                rmse = swarm.hist_rmse[-1]
                if rmse > rmse_reset_threshold:
                    break

            if k % store_every != 0:
                continue

            # Recolectar datos
            adj = swarm.last_adj.copy()
            targets = swarm.last_targets.copy()
            F_base = swarm.last_F_base.copy()
            F_ideal = swarm._compute_ideal_forces(targets, adj)

            deltaF = F_ideal - F_base
            deltaF = np.clip(deltaF, -params.ctrl_max_delta, 
                           params.ctrl_max_delta).astype(np.float32)

            h, adj_t = swarm._build_features(adj, targets, ref_vel)

            graphs_h.append(h.numpy())
            graphs_adj.append(adj_t.numpy())
            graphs_dF.append(deltaF)
            graphs_rmse.append(rmse if len(swarm.hist_rmse) > 0 else 0.0)

            graphs_collected += 1
            ep_graphs += 1
            
            if ep_graphs >= max_graphs_per_episode:
                break
                
            if graphs_collected >= num_graphs:
                break

    if verbose:
        print(f"  Recolectados {graphs_collected} graphs de {ep_idx} episodios")
        print(f"  Promedio graphs/episodio: {graphs_collected/ep_idx:.1f}")
    
    return (np.stack(graphs_h), np.stack(graphs_adj), 
            np.stack(graphs_dF), np.array(graphs_rmse))


def train_gat(params: ControlParams, H: np.ndarray, A: np.ndarray, 
              dF: np.ndarray, rmse_arr: np.ndarray,
              epochs: int = 40, lr: float = 1e-3,
              verbose: bool = True) -> Tuple[dict, float]:
    """
    Entrena la red GAT.
    
    Args:
        params: Parámetros del sistema
        H: Features [G, N, F]
        A: Adyacencias [G, N, N]
        dF: Deltas de fuerza [G, N, 2]
        rmse_arr: RMSEs por grafo
        epochs: Número de épocas
        lr: Learning rate
        verbose: Si mostrar progreso
    
    Returns:
        state_dict: Pesos entrenados de la red
        delta_scale: Escala de delta calculada
    """
    G, N, Fdim = H.shape
    
    net = GraphControlNet(
        Fdim, params.neural.gat_hidden_dim,
        num_heads=params.neural.gat_num_heads,
        num_layers=params.neural.gat_num_layers
    )
    
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr/10
    )

    H_t = torch.from_numpy(H).float()
    A_t = torch.from_numpy(A).float()
    dF_t = torch.from_numpy(dF).float()

    # Calcular escala de delta
    dF_norms = np.linalg.norm(dF.reshape(G, -1), axis=1)
    raw_scale = float(np.mean(dF_norms))
    scale = min(max(raw_scale, 0.5), 3.0)
    
    if verbose:
        print(f"  Delta scale: {scale:.3f} (raw: {raw_scale:.3f})")
    
    # Pesos de importancia
    importance_weights = 1.0 + np.clip(
        dF_norms / np.mean(dF_norms), 0.5, 3.0
    )

    best_loss = float('inf')
    best_state = None
    loss_history = []

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(G)
        total_loss = 0.0
        
        for idx in perm:
            h_g, a_g, y_g = H_t[idx], A_t[idx], dF_t[idx] / scale
            w = importance_weights[int(idx)]

            optimizer.zero_grad()
            y_pred, conf = net(h_g, a_g, return_confidence=True)
            
            # Loss principal
            loss_main = F.mse_loss(y_pred, y_g, reduction='none').mean() * w
            
            # Loss de confianza
            pred_error = ((y_pred - y_g) ** 2).sum(dim=1, keepdim=True).detach()
            target_conf = torch.exp(-pred_error * 2)
            loss_conf = F.mse_loss(conf, target_conf) * 0.1
            
            loss = loss_main + loss_conf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / G
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if verbose and (ep % 5 == 0 or ep == epochs):
            print(f"  Epoch {ep:2d} | Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return best_state, scale


def save_trained_model(state_dict: dict, delta_scale: float, 
                       output_path: str, params: ControlParams = None):
    """
    Guarda el modelo entrenado junto con metadatos.
    
    Args:
        state_dict: Pesos del modelo
        delta_scale: Escala de delta
        output_path: Ruta de salida
        params: Parámetros (para guardar configuración)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar pesos
    torch.save(state_dict, output_path)
    
    # Guardar metadatos
    meta = {
        'delta_scale': delta_scale,
        'gat_hidden_dim': params.neural.gat_hidden_dim if params else 128,
        'gat_num_heads': params.neural.gat_num_heads if params else 4,
        'gat_num_layers': params.neural.gat_num_layers if params else 2,
        'in_features': params.neural.in_features if params else 16,
    }
    
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Modelo guardado en: {output_path}")
    print(f"  Metadatos en: {meta_path}")


def load_trained_model(model_path: str, params: ControlParams = None) -> Tuple[dict, float]:
    """
    Carga un modelo entrenado con sus metadatos.
    
    Args:
        model_path: Ruta del modelo
        params: Parámetros (se actualizan con delta_scale)
        
    Returns:
        state_dict: Pesos del modelo
        delta_scale: Escala de delta
    """
    model_path = Path(model_path)
    
    # Cargar pesos
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Cargar metadatos
    meta_path = model_path.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        delta_scale = meta.get('delta_scale', 1.0)
    else:
        delta_scale = 1.0
    
    # Actualizar parámetros si se proporcionan
    if params is not None:
        params.ctrl_delta_scale = delta_scale
    
    return state_dict, delta_scale


def full_training_pipeline(params: ControlParams = None, 
                           num_graphs: int = 8000,
                           epochs: int = 40,
                           seed: int = 0,
                           output_path: str = None) -> Tuple[dict, float]:
    """
    Pipeline completo de entrenamiento.
    
    Args:
        params: Parámetros del sistema
        num_graphs: Número de grafos para entrenamiento
        epochs: Número de épocas
        seed: Semilla para reproducibilidad
        output_path: Ruta para guardar el modelo (opcional)
        
    Returns:
        state_dict: Pesos entrenados
        delta_scale: Escala de delta
    """
    if params is None:
        params = ControlParams()
    
    print("="*60)
    print("ENTRENAMIENTO GAT")
    print("="*60)
    
    # Recolectar datos
    print("\n[1/2] Recolectando datos de entrenamiento...")
    H, A, dF, rmse_arr = collect_training_data(params, num_graphs, seed)
    print(f"  Dataset: {len(H)} graphs")
    
    # Entrenar
    print("\n[2/2] Entrenando red...")
    state_dict, delta_scale = train_gat(
        params, H, A, dF, rmse_arr, epochs=epochs
    )
    
    # Actualizar parámetros
    params.ctrl_delta_scale = delta_scale
    
    # Guardar si se especifica ruta
    if output_path:
        save_trained_model(state_dict, delta_scale, output_path, params)
    
    print("\n" + "="*60)
    print("Entrenamiento completado")
    print("="*60)
    
    return state_dict, delta_scale


if __name__ == "__main__":
    # Ejemplo de uso
    params = ControlParams()
    state_dict, scale = full_training_pipeline(
        params,
        num_graphs=1000,
        epochs=10,
        seed=0,
        output_path="results/models/gat_model.pt"
    )
