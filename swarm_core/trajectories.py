#!/usr/bin/env python3
"""
================================================================================
GENERADORES DE TRAYECTORIA
================================================================================

Trayectorias para benchmark:
- spiral: Espiral expansiva (curvas suaves)
- lawnmower: Patrón de barrido (rectas + giros bruscos)
- snake: Serpenteo sinusoidal (oscilaciones continuas)
- station_keeping: Posición fija (modo hover)

También incluye generador aleatorio para entrenamiento.
"""

import numpy as np
from typing import Tuple
from enum import Enum


class TrajectoryType(Enum):
    """Tipos de trayectoria disponibles"""
    SPIRAL = "spiral"
    LAWNMOWER = "lawnmower"
    SNAKE = "snake"
    STATION_KEEPING = "station_keeping"


class RandomTrajectoryGenerator:
    """Genera trayectorias aleatorias para entrenamiento."""
    
    def __init__(self, v_max: float = 3.0, a_max: float = 1.0, 
                 change_T: float = 2.0, seed: int = 0):
        self.v_max = v_max
        self.a_max = a_max
        self.change_T = change_T
        self.rng = np.random.RandomState(seed)
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.t_last_change = 0.0

    def step(self, t: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Avanza la trayectoria un paso."""
        if (t - self.t_last_change) >= self.change_T:
            theta = self.rng.uniform(0, 2 * np.pi)
            a_mag = self.rng.uniform(0, self.a_max)
            self.acc = a_mag * np.array([np.cos(theta), np.sin(theta)])
            self.t_last_change = t

        self.vel += self.acc * dt
        speed = np.linalg.norm(self.vel)
        if speed > self.v_max:
            self.vel *= (self.v_max / (speed + 1e-9))

        self.pos += self.vel * dt
        return self.pos.copy(), self.vel.copy()
    
    def reset(self, seed: int = None):
        """Reinicia el generador"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.t_last_change = 0.0


class TrajectoryGenerator:
    """
    Genera trayectorias predefinidas para benchmark.
    
    Trayectorias disponibles:
    - spiral: Espiral de Arquímedes
    - lawnmower: Patrón de barrido bidireccional
    - snake: Serpenteo sinusoidal
    - station_keeping: Posición fija
    """
    
    # Parámetros por tipo de trayectoria
    SPIRAL_PARAMS = {'a': 0.5, 'b': 0.2, 'w': 0.2}
    LAWNMOWER_PARAMS = {'cycle': 20.0, 'width': 10.0, 'advance_vel': 0.5}
    SNAKE_PARAMS = {'amplitude': 5.0, 'period': 10.0, 'advance_vel': 0.6}
    STATION_PARAMS = {'x': 5.0, 'y': 5.0}
    
    @staticmethod
    def get_ref(t: float, mode: str = 'spiral') -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene posición y velocidad de referencia.
        
        Args:
            t: Tiempo [s]
            mode: Tipo de trayectoria
            
        Returns:
            pos: Posición [x, y]
            vel: Velocidad [vx, vy]
        """
        if t < 0:
            return np.zeros(2), np.zeros(2)

        if mode == 'spiral':
            return TrajectoryGenerator._spiral(t)
        elif mode == 'lawnmower':
            return TrajectoryGenerator._lawnmower(t)
        elif mode == 'snake':
            return TrajectoryGenerator._snake(t)
        elif mode == 'station_keeping':
            return TrajectoryGenerator._station_keeping(t)
        
        return np.zeros(2), np.zeros(2)
    
    @staticmethod
    def _spiral(t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Espiral de Arquímedes"""
        p = TrajectoryGenerator.SPIRAL_PARAMS
        r = p['a'] + p['b'] * t
        angle = p['w'] * t
        pos = np.array([r * np.cos(angle), r * np.sin(angle)])
        vx = p['b'] * np.cos(angle) - r * p['w'] * np.sin(angle)
        vy = p['b'] * np.sin(angle) + r * p['w'] * np.cos(angle)
        return pos, np.array([vx, vy])
    
    @staticmethod
    def _lawnmower(t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Patrón de barrido"""
        p = TrajectoryGenerator.LAWNMOWER_PARAMS
        scan_speed = p['width'] / (p['cycle'] / 2.0)
        tau = t % p['cycle']
        
        if tau < (p['cycle'] / 2.0):
            x = -p['width'] / 2 + scan_speed * tau
            vx = scan_speed
        else:
            tau_return = tau - (p['cycle'] / 2.0)
            x = p['width'] / 2 - scan_speed * tau_return
            vx = -scan_speed
            
        y = p['advance_vel'] * t
        return np.array([x, y]), np.array([vx, p['advance_vel']])
    
    @staticmethod
    def _snake(t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Serpenteo sinusoidal"""
        p = TrajectoryGenerator.SNAKE_PARAMS
        w = 2 * np.pi / p['period']
        x = p['amplitude'] * np.sin(w * t)
        y = p['advance_vel'] * t
        vx = p['amplitude'] * w * np.cos(w * t)
        return np.array([x, y]), np.array([vx, p['advance_vel']])
    
    @staticmethod
    def _station_keeping(t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Posición fija"""
        p = TrajectoryGenerator.STATION_PARAMS
        return np.array([p['x'], p['y']]), np.zeros(2)
    
    @staticmethod
    def is_straight_segment(mode: str, t: float, threshold: float = 0.3) -> bool:
        """
        Determina si el segmento actual es recto (poca aceleración).
        Útil para análisis de curvas vs rectas.
        """
        _, vel = TrajectoryGenerator.get_ref(t, mode)
        _, vel_next = TrajectoryGenerator.get_ref(t + 0.1, mode)
        acc = np.linalg.norm(vel_next - vel) / 0.1
        return acc < threshold
    
    @staticmethod
    def get_curvature_profile(mode: str, duration: float, 
                              dt: float = 0.1) -> np.ndarray:
        """
        Calcula perfil de curvatura/aceleración para una trayectoria.
        Útil para análisis de curvas vs rectas.
        """
        times = np.arange(0, duration, dt)
        curvatures = []
        
        for t in times:
            _, vel = TrajectoryGenerator.get_ref(t, mode)
            _, vel_next = TrajectoryGenerator.get_ref(t + dt, mode)
            acc = np.linalg.norm(vel_next - vel) / dt
            curvatures.append(acc)
        
        return np.array(curvatures)


def get_formation(N: int, radius: float = 2.0) -> dict:
    """
    Genera formación circular para N agentes.
    
    Args:
        N: Número de agentes
        radius: Radio de la formación [m]
        
    Returns:
        formation: Diccionario {id: offset}
    """
    form = {0: np.array([0., 0.])}  # Líder en el centro
    for i in range(1, N):
        angle = i * (2 * np.pi / (N - 1))
        form[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    return form
