#!/usr/bin/env python3
"""
================================================================================
CONTROLADOR PRINCIPAL DEL ENJAMBRE
================================================================================

Integra:
- Simulación de sensores (IMU, UWB)
- Estimación de estado (UKF)
- Control PID distribuido con consenso
- Corrección GAT (opcional)

Modos de operación:
- training_mode: Para recolección de datos (sin IA)
- use_ia: Con corrección GAT activa
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .sensors import IMUSensor, UWBSensor, LeaderGNSS
from .estimator import UKFEstimator, EstimatorMode
from .gat_network import GraphControlNet, load_gat_model
from .trajectories import get_formation


@dataclass
class StepData:
    """Datos de un paso de simulación para logging"""
    gt_pos: np.ndarray          # Posición ground truth [N, 2]
    est_pos: np.ndarray         # Posición estimada [N, 2]
    targets: np.ndarray         # Targets de formación [N, 2]
    velocities: np.ndarray      # Velocidades [N, 2]
    accelerations: np.ndarray   # Aceleraciones [N, 2]
    ia_allowed: bool            # Si la IA estaba permitida
    confidence: np.ndarray      # Confianza por agente [N]
    deltaF_applied: np.ndarray  # Delta aplicado [N, 2]
    formation_error: float      # RMSE de formación
    localization_error: float   # Error de localización medio


class SwarmController:
    """
    Controlador principal del enjambre de robots.
    
    Integra:
    - Simulación de sensores (IMU, UWB)
    - Estimación de estado (UKF)
    - Control PID distribuido
    - Corrección GAT (opcional)
    """
    
    def __init__(self, params, formation: Dict[int, np.ndarray] = None,
                 use_ia: bool = False, ctrl_state_dict: Optional[dict] = None,
                 training_mode: bool = False, estimator_mode: str = "imu_uwb",
                 seed: int = None):
        self.p = params
        self.N = params.N
        self.use_ia = use_ia
        self.training_mode = training_mode
        
        # Semilla para reproducibilidad
        self.rng = np.random.RandomState(seed) if seed is not None \
                   else np.random.RandomState()
        
        # Formación
        self.form = formation if formation is not None \
                    else get_formation(self.N)
        
        # Estado de los agentes
        self.pos = np.zeros((self.N, 2))
        self.vel = np.zeros((self.N, 2))
        self.acc = np.zeros((self.N, 2))
        self.step_count = 0

        # Sensores
        self.imu = IMUSensor(self.N, params.imu, self.rng)
        self.uwb = UWBSensor(self.N, params.uwb, params.dt, self.rng)
        self.leader_gnss = LeaderGNSS(params.leader_gnss_noise, seed=42)

        # Estimador UKF
        self.ukf = UKFEstimator(self.N, params.dt, EstimatorMode(estimator_mode))

        # Red de control GAT
        if use_ia and ctrl_state_dict is not None:
            self.ctrl_net = load_gat_model(ctrl_state_dict, params)
        else:
            self.ctrl_net = None

        # Variables de control PID
        self.integral = np.zeros((self.N, 2))
        self.prev_err = np.zeros((self.N, 2))
        self.hist_rmse = []
        
        # Historiales
        self.error_history = deque(maxlen=10)
        self.vel_leader_history = deque(maxlen=5)

        # Estado anterior
        self.last_vel = np.zeros((self.N, 2))
        self.last_adj = np.zeros((self.N, self.N))
        self.last_targets = np.zeros((self.N, 2))
        self.last_F_base = np.zeros((self.N, 2))
        self.last_leader_vel = np.zeros(2)
        self.prev_deltaF_applied = np.zeros((self.N, 2))
        self.last_rmse = 0.0
        self.last_confidence = np.zeros(self.N)
        self.last_ia_allowed = False

        # Control de modos
        self.hover_timer = 0.0
        self.hover_enabled = False
        self.is_static_mode = False
        self.straight_line_timer = 0.0

    def init(self, pos_leader: np.ndarray):
        """Inicializa posiciones del enjambre."""
        for i in range(self.N):
            if i == 0:
                self.pos[i] = pos_leader.copy()
            else:
                self.pos[i] = pos_leader + self.form[i] + self.rng.randn(2) * 0.05
            self.ukf.means[i] = self.pos[i].copy()

    def _run_calibration(self):
        """Ejecuta calibración de bias IMU durante fase inicial."""
        for i in range(1, self.N):
            vel_raw = self.imu.read(i, self.vel[i])
            self.imu.calibrate(i, vel_raw)

    def _update_estimation(self, measurements: np.ndarray, 
                          nlos_scores: np.ndarray,
                          pred_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Actualiza estimación de estado con mediciones UWB.
        
        Returns:
            adj_matrix: Matriz de adyacencia de mediciones válidas
            panic_mask: Máscara de agentes en pánico (sin mediciones)
        """
        leader_broadcast = self.leader_gnss.get_broadcast(self.pos[0])
        
        # Predicción UKF
        for i in range(1, self.N):
            self.ukf.predict(i, pred_pos[i], self.p.dt)

        # Líder tiene posición conocida
        self.ukf.set_known_position(0, self.pos[0])

        adj_matrix = np.zeros((self.N, self.N))
        panic_mask = np.zeros(self.N, dtype=bool)

        for i in range(1, self.N):
            old_mean = self.ukf.means[i].copy()

            # Encontrar vecinos más cercanos
            est_dists = []
            for j in range(self.N):
                if i != j:
                    d = np.linalg.norm(self.ukf.means[i] - self.ukf.means[j])
                    est_dists.append((d, j))
            est_dists.sort(key=lambda x: x[0])
            neighbors = [x[1] for x in est_dists[:self.p.max_neighbors]]

            valid_measurements = 0

            for j in neighbors:
                z = measurements[i, j]
                if z <= 0:
                    continue

                diff_est = self.ukf.means[j] - self.ukf.means[i]
                dist_est = np.linalg.norm(diff_est)
                if dist_est < 0.01:
                    continue

                # Test de Mahalanobis
                h_factor = 0.1 if j == 0 else 1.0
                tr_i = self.ukf.get_trace(i)
                tr_j = self.ukf.get_trace(j)
                var_base = max((tr_i + tr_j + 0.2) * h_factor, 1e-4)

                innov = z - dist_est
                if abs(innov) / np.sqrt(var_base) >= self.p.mahalanobis_threshold:
                    continue

                # Actualización UKF
                nlos_score = nlos_scores[i, j]
                R = var_base * (1.0 + 4.0 * nlos_score)
                self.ukf.update_range(i, z, self.ukf.means[j].copy(), R)

                adj_matrix[i, j] = 1.0
                valid_measurements += 1

            # Manejo de pánico (sin mediciones válidas)
            if valid_measurements == 0:
                panic_mask[i] = True
                # Buscar mejor medición disponible
                best_j, min_inn = -1, 1000.0
                for j in neighbors:
                    z = measurements[i, j]
                    if z > 0:
                        diff = self.ukf.means[j] - self.ukf.means[i]
                        inn = abs(z - np.linalg.norm(diff))
                        if inn < min_inn:
                            min_inn, best_j = inn, j
                            
                if best_j != -1 and min_inn < 6.0:
                    z = measurements[i, best_j]
                    diff = self.ukf.means[best_j] - self.ukf.means[i]
                    d_est = np.linalg.norm(diff)
                    if d_est > 1e-3:
                        corr = (z - d_est) * (-diff / d_est)
                        self.ukf.means[i] += 0.5 * corr
                        self.ukf.covs[i] = np.eye(2) * 0.5
                        adj_matrix[i, best_j] = 1.0

            # Actualización directa en modo estático
            if self.is_static_mode and not self.training_mode:
                target_i = leader_broadcast + self.form[i]
                self.ukf.direct_update(i, target_i, R_diag=0.01)

            # Corrección de bias IMU
            pos_correction = self.ukf.means[i] - old_mean
            mag = np.linalg.norm(pos_correction)
            
            v = self.vel[i]
            a = (self.vel[i] - self.last_vel[i]) / max(self.p.dt, 1e-6)
            centrifugal = v[0] * a[1] - v[1] * a[0]
            turning = abs(centrifugal) > self.p.turn_gate_threshold

            if not turning and mag < 0.05:
                vel_error = pos_correction / max(self.p.dt, 1e-6)
                self.imu.correct_bias(i, vel_error, self.p.bias_learning_rate,
                                      self.p.bias_decay)

            # Corrección en hover
            if self.hover_enabled and not self.training_mode and not self.is_static_mode:
                target_i = leader_broadcast + self.form[i]
                x = self.ukf.means[i]
                P = self.ukf.covs[i]
                R_hover = (self.p.hover_pseudo_R ** 2) * np.eye(2)
                
                try:
                    P_inv = np.linalg.inv(P + 1e-6 * np.eye(2))
                    R_inv = np.linalg.inv(R_hover + 1e-6 * np.eye(2))
                    Y_new = P_inv + R_inv
                    P_new = np.linalg.inv(Y_new)
                    mu_new = P_new @ (P_inv @ x + R_inv @ target_i)
                    self.ukf.means[i] = mu_new
                    self.ukf.covs[i] = 0.5 * (P_new + P_new.T) + 1e-9 * np.eye(2)
                except np.linalg.LinAlgError:
                    pass

        return adj_matrix, panic_mask

    def _control_pid(self, vel_leader: np.ndarray, targets: np.ndarray,
                     adj_matrix: np.ndarray) -> np.ndarray:
        """Control PID distribuido con consenso."""
        F = np.zeros((self.N, 2))
        
        for i in range(1, self.N):
            est_pos = self.ukf.means[i]
            err = targets[i] - est_pos
            dist_err = np.linalg.norm(err)

            # Ganancias (ajustadas en hover)
            k_p, k_i, k_d = self.p.k_p, self.p.k_i, self.p.k_d
            k_cons = self.p.k_consensus

            if self.hover_enabled and not self.training_mode:
                k_i *= self.p.hover_k_i_scale
                k_cons *= self.p.hover_k_consensus_scale

            # Término proporcional
            P_term = k_p * err

            # Término integral (con anti-windup)
            if dist_err < 0.5:
                self.integral[i] += err * self.p.dt
                int_mag = np.linalg.norm(self.integral[i])
                if int_mag > 1.0:
                    self.integral[i] *= 1.0 / int_mag
            else:
                self.integral[i] *= 0.95
            I_term = k_i * self.integral[i]

            # Término derivativo
            d_err = (err - self.prev_err[i]) / max(self.p.dt, 1e-6)
            d_err = np.clip(d_err, -3.0, 3.0)
            D_term = k_d * d_err
            self.prev_err[i] = err.copy()

            # Término de consenso
            F_con = np.zeros(2)
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if neighbors.size > 0:
                for j in neighbors:
                    if j != i:
                        diff = (self.ukf.means[j] - self.form[j]) - \
                               (self.ukf.means[i] - self.form[i])
                        F_con += diff
                F_con /= max(1, neighbors.size)
            
            con_mag = np.linalg.norm(F_con)
            if con_mag > 1.5:
                F_con *= 1.5 / con_mag

            # Término de separación (evitar colisiones)
            F_sep = np.zeros(2)
            for j in range(self.N):
                if i != j:
                    diff = est_pos - self.ukf.means[j]
                    dist = np.linalg.norm(diff)
                    if 0.05 < dist < self.p.d_safe:
                        F_sep += 5.0 * (1.0/dist - 1.0/self.p.d_safe) * (diff/dist)

            # Feedforward
            F_ff = vel_leader

            # Fuerza total
            F[i] = F_ff + P_term + I_term + D_term + k_cons * F_con + F_sep

        return F

    def _compute_ideal_forces(self, targets: np.ndarray, 
                              adj_matrix: np.ndarray) -> np.ndarray:
        """Calcula fuerzas ideales usando posiciones REALES (para entrenamiento)."""
        F = np.zeros((self.N, 2))
        
        for i in range(1, self.N):
            pos_true = self.pos[i]
            err_true = targets[i] - pos_true
            P_term = self.p.k_p * err_true
            D_like = -0.5 * self.vel[i]

            F_con = np.zeros(2)
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if neighbors.size > 0:
                for j in neighbors:
                    if j != i:
                        diff = (self.pos[j] - self.form[j]) - (self.pos[i] - self.form[i])
                        F_con += diff
                F_con /= max(1, neighbors.size)

            con_mag = np.linalg.norm(F_con)
            if con_mag > 1.5:
                F_con *= 1.5 / con_mag

            F_sep = np.zeros(2)
            for j in range(self.N):
                if i != j:
                    diff = pos_true - self.pos[j]
                    dist = np.linalg.norm(diff)
                    if 0.05 < dist < self.p.d_safe:
                        F_sep += 5.0 * (1.0/dist - 1.0/self.p.d_safe) * (diff/dist)

            F[i] = P_term + D_like + self.p.k_consensus * F_con + F_sep

        return F

    def _build_features(self, adj_matrix: np.ndarray, targets: np.ndarray,
                        vel_leader: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construye features para la red GAT (16 dimensiones)."""
        N = self.N
        mu = self.ukf.means
        vel = self.vel
        form_err = mu - targets
        
        cov_traces = np.array([
            self.ukf.get_trace(i) for i in range(N)
        ]).reshape(N, 1)
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)

        nlos_local = np.zeros((N, 1))
        for i in range(N):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if neighbors.size > 0:
                nlos_local[i, 0] = np.mean(self.uwb.last_nlos_scores[i, neighbors])
            else:
                nlos_local[i, 0] = 0.5

        vel_rel = vel - vel_leader
        acc_est = self.acc
        
        if len(self.error_history) > 0:
            hist_err = np.mean(self.error_history, axis=0)
        else:
            hist_err = np.zeros((N, 2))
        hist_err_norm = np.linalg.norm(hist_err, axis=1, keepdims=True)
        
        num_neighbors = np.sum(adj_matrix, axis=1, keepdims=True) / self.p.max_neighbors
        
        # Aceleración del líder
        if len(self.vel_leader_history) >= 2:
            prev_vel = self.vel_leader_history[-2]
            leader_acc = (vel_leader - prev_vel) / self.p.dt
            leader_acc_norm = np.linalg.norm(leader_acc) / self.p.a_max
        else:
            leader_acc_norm = 0.0
        
        leader_acc_arr = np.full((N, 1), leader_acc_norm)

        # Concatenar features [N, 16]
        h_np = np.hstack([
            mu,               # 2: posición estimada
            vel,              # 2: velocidad
            form_err,         # 2: error de formación
            cov_traces,       # 1: traza de covarianza
            speeds,           # 1: velocidad escalar
            nlos_local,       # 1: prob NLOS local
            vel_rel,          # 2: velocidad relativa
            acc_est,          # 2: aceleración estimada
            hist_err_norm,    # 1: error histórico
            num_neighbors,    # 1: número de vecinos
            leader_acc_arr    # 1: aceleración del líder
        ])
        
        h_np = np.clip(h_np, -100, 100)
        
        h = torch.FloatTensor(h_np)
        adj = torch.FloatTensor(adj_matrix.astype(np.float32))
        return h, adj

    def _apply_physics(self, F: np.ndarray, panic_mask: np.ndarray,
                       pos_leader: np.ndarray, vel_leader: np.ndarray):
        """Aplica física a los agentes."""
        self.pos[0] = pos_leader.copy()
        self.vel[0] = vel_leader.copy()
        
        for i in range(1, self.N):
            if panic_mask[i]:
                self.vel[i] *= 0.5

            drag = self.p.base_drag
            spring = np.zeros(2)

            if self.hover_enabled and not self.training_mode:
                drag = self.p.base_drag * self.p.hover_drag_scale
                target = self.leader_gnss.last_broadcast + self.form[i]
                spring = self.p.hover_k_phys * (target - self.ukf.means[i])

            accel = F[i] + spring - drag * self.vel[i]
            accel = np.clip(accel, -self.p.a_max, self.p.a_max)
            
            self.acc[i] = accel
            self.vel[i] += accel * self.p.dt

            speed = np.linalg.norm(self.vel[i])
            if speed > self.p.v_max:
                self.vel[i] *= self.p.v_max / speed

            self.pos[i] += self.vel[i] * self.p.dt

        self.last_vel = self.vel.copy()

    def step(self, pos_leader: np.ndarray, vel_leader: np.ndarray, 
             time_now: float) -> Optional[StepData]:
        """
        Ejecuta un paso de simulación del enjambre.
        
        Args:
            pos_leader: Posición del líder
            vel_leader: Velocidad del líder
            time_now: Tiempo actual
            
        Returns:
            step_data: Datos del paso para logging (None durante calibración)
        """
        self.step_count += 1

        # Fase de calibración
        if time_now < self.p.calibration_time:
            self._run_calibration()
            self.pos[0] = pos_leader.copy()
            self.vel[0] = vel_leader.copy()
            return None

        # Calcular targets de formación
        targets = np.array([
            pos_leader if i == 0 else pos_leader + self.form[i]
            for i in range(self.N)
        ])
        self.last_targets = targets.copy()
        
        # Actualizar historiales
        current_errors = self.ukf.means - targets
        self.error_history.append(current_errors.copy())
        self.vel_leader_history.append(vel_leader.copy())

        # Detectar modo estático/hover
        leader_speed = np.linalg.norm(vel_leader)
        self.is_static_mode = leader_speed < self.p.hover_speed_thresh
        
        if self.training_mode:
            self.hover_timer = 0.0
            self.hover_enabled = False
        else:
            if leader_speed < self.p.hover_speed_thresh:
                self.hover_timer += self.p.dt
            else:
                self.hover_timer = 0.0
            self.hover_enabled = self.hover_timer >= self.p.hover_min_time

        # Mediciones UWB (pasamos estimaciones para mejorar detección NLOS)
        meas, nlos_scores = self.uwb.measure(self.pos, self.ukf.means)

        # Predicción IMU
        pred_pos = np.zeros((self.N, 2))
        pred_pos[0] = pos_leader
        
        zupt_active = (
            self.p.zupt_extreme_enabled and
            not self.training_mode and
            self.hover_timer >= self.p.zupt_extreme_min_time
        )
        
        for i in range(1, self.N):
            imu_reading = self.imu.read(i, self.vel[i])
            imu_pred = self.ukf.means[i] + \
                      (imu_reading - self.imu.est_bias[i]) * self.p.dt
            
            if zupt_active:
                target = self.leader_gnss.last_broadcast + self.form[i]
                alpha = self.p.zupt_extreme_blend
                pred_pos[i] = (1 - alpha) * imu_pred + alpha * target
            else:
                pred_pos[i] = imu_pred

        # Actualización de estimación
        adj, panic = self._update_estimation(meas, nlos_scores, pred_pos)
        self.last_adj = adj.copy()

        # Control base PID
        F_base = self._control_pid(vel_leader, targets, adj)
        self.last_F_base = F_base.copy()

        # Detección de rectas largas (para gate de IA)
        a_leader = (vel_leader - self.last_leader_vel) / max(self.p.dt, 1e-6)
        leader_acc_norm = np.linalg.norm(a_leader)
        self.last_leader_vel = vel_leader.copy()
        
        if leader_speed > 0.5 and leader_acc_norm < 0.3:
            self.straight_line_timer += self.p.dt
        else:
            self.straight_line_timer = 0.0
        
        in_long_straight = self.straight_line_timer > 1.5

        # Decidir si aplicar IA
        ia_allowed = (
            self.use_ia and 
            self.ctrl_net is not None and
            not self.is_static_mode and
            not in_long_straight and
            leader_acc_norm < self.p.ctrl_leader_acc_gate and
            self.step_count > self.p.ctrl_burnin_steps and
            self.last_rmse < self.p.ctrl_rmse_gate
        )
        self.last_ia_allowed = ia_allowed

        F_total = F_base.copy()
        deltaF_applied = np.zeros_like(F_base)
        confidence = np.zeros(self.N)

        # Aplicar corrección GAT
        if ia_allowed:
            h, adj_t = self._build_features(adj, targets, vel_leader)
            with torch.no_grad():
                deltaF_pred, conf = self.ctrl_net(h, adj_t, return_confidence=True)
                deltaF_pred = deltaF_pred.numpy()
                confidence = conf.numpy().flatten()

            deltaF_pred *= self.p.ctrl_delta_scale

            # Limitar magnitud
            max_delta = self.p.ctrl_max_delta
            norms = np.linalg.norm(deltaF_pred, axis=1, keepdims=True) + 1e-9
            mask_big = norms > max_delta
            if np.any(mask_big):
                deltaF_pred[mask_big[:, 0]] *= (max_delta / norms[mask_big[:, 0]])

            speed_factor = min(1.0, leader_speed / self.p.ctrl_v_ref_gate) \
                          if leader_speed > 0.1 else 0.5

            for i in range(1, self.N):
                # Gates de aplicación
                err_est = np.linalg.norm(self.ukf.means[i] - targets[i])
                if err_est > self.p.ctrl_err_gate:
                    continue

                cov_tr = self.ukf.get_trace(i)
                if cov_tr > self.p.ctrl_cov_gate:
                    continue

                neighbors = np.where(adj[i] > 0)[0]
                nlos_loc = float(np.mean(self.uwb.last_nlos_scores[i, neighbors])) \
                          if neighbors.size > 0 else 0.5
                if nlos_loc > self.p.ctrl_nlos_gate:
                    continue
                
                conf_i = confidence[i]
                if conf_i < self.p.ctrl_confidence_thresh:
                    continue

                # Aplicar con gamma adaptativo
                gamma = self.p.ctrl_gamma_base + \
                       (self.p.ctrl_gamma_max - self.p.ctrl_gamma_base) * conf_i

                proposed = gamma * deltaF_pred[i] * speed_factor
                deltaF_applied[i] = (1.0 - self.p.ctrl_lp_alpha) * \
                                    self.prev_deltaF_applied[i] + \
                                    self.p.ctrl_lp_alpha * proposed
                F_total[i] += deltaF_applied[i]
        else:
            self.prev_deltaF_applied *= 0.9

        self.prev_deltaF_applied = deltaF_applied
        self.last_confidence = confidence

        # Aplicar física
        self._apply_physics(F_total, panic, pos_leader, vel_leader)

        # Calcular métricas
        formation_error = np.sqrt(np.mean([
            np.linalg.norm(self.pos[i] - targets[i])**2 
            for i in range(self.N)
        ]))
        self.hist_rmse.append(formation_error)
        self.last_rmse = formation_error
        
        localization_error = np.mean([
            np.linalg.norm(self.pos[i] - self.ukf.means[i])
            for i in range(1, self.N)
        ])

        # Construir datos del paso
        step_data = StepData(
            gt_pos=self.pos.copy(),
            est_pos=self.ukf.means.copy(),
            targets=targets.copy(),
            velocities=self.vel.copy(),
            accelerations=self.acc.copy(),
            ia_allowed=ia_allowed,
            confidence=confidence.copy(),
            deltaF_applied=deltaF_applied.copy(),
            formation_error=formation_error,
            localization_error=localization_error
        )
        
        return step_data
    
    def reset(self, seed: int = None):
        """Reinicia el controlador"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self.pos = np.zeros((self.N, 2))
        self.vel = np.zeros((self.N, 2))
        self.acc = np.zeros((self.N, 2))
        self.step_count = 0
        
        self.imu.reset(self.rng)
        self.uwb.reset(self.rng)
        self.ukf.reset()
        
        self.integral = np.zeros((self.N, 2))
        self.prev_err = np.zeros((self.N, 2))
        self.hist_rmse = []
        self.error_history.clear()
        self.vel_leader_history.clear()
        
        self.last_vel = np.zeros((self.N, 2))
        self.last_adj = np.zeros((self.N, self.N))
        self.last_targets = np.zeros((self.N, 2))
        self.last_F_base = np.zeros((self.N, 2))
        self.last_leader_vel = np.zeros(2)
        self.prev_deltaF_applied = np.zeros((self.N, 2))
        self.last_rmse = 0.0
        
        self.hover_timer = 0.0
        self.hover_enabled = False
        self.is_static_mode = False
        self.straight_line_timer = 0.0
