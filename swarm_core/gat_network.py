#!/usr/bin/env python3
"""
================================================================================
RED GRAPH ATTENTION NETWORK (GAT) PARA CONTROL
================================================================================

Arquitectura:
- 2 capas GAT con 4 cabezas de atención
- Conexiones residuales
- Cabeza de confianza para auto-evaluación
- Features: 16 dimensiones

Referencia: Veličković et al., "Graph Attention Networks", ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GATLayer(nn.Module):
    """
    Capa de Graph Attention Network
    
    Implementa el mecanismo de atención para grafos donde cada nodo
    agrega información de sus vecinos ponderada por coeficientes de atención.
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, 
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.head_dim = out_features // num_heads if concat else out_features
        
        # Proyección lineal
        self.W = nn.Linear(in_features, self.head_dim * num_heads, bias=False)
        
        # Parámetros de atención
        self.att = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la capa GAT.
        
        Args:
            h: Features de nodos [N, in_features]
            adj_mask: Matriz de adyacencia [N, N]
            
        Returns:
            out: Features actualizadas [N, out_features]
        """
        N = h.size(0)
        device = h.device
        
        # Proyección: [N, heads, head_dim]
        h_proj = self.W(h).view(N, self.num_heads, self.head_dim)
        
        # Expandir para pares de nodos
        h_i = h_proj.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)
        h_j = h_proj.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)
        
        # Calcular coeficientes de atención
        cat_ij = torch.cat([h_i, h_j], dim=-1)
        e = (cat_ij * self.att).sum(dim=-1)
        e = self.leaky_relu(e).permute(2, 0, 1)  # [heads, N, N]
        
        # Aplicar máscara de adyacencia (incluir self-loops)
        adj = adj_mask.unsqueeze(0).bool()
        adj = adj | torch.eye(N, device=device).bool().unsqueeze(0)
        e = e.masked_fill(~adj, float('-inf'))
        
        # Softmax sobre vecinos
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        
        # Agregación ponderada
        h_proj_t = h_proj.permute(1, 0, 2)  # [heads, N, head_dim]
        out = torch.bmm(alpha, h_proj_t)     # [heads, N, head_dim]
        
        # Concatenar o promediar cabezas
        if self.concat:
            out = out.permute(1, 0, 2).reshape(N, -1)
        else:
            out = out.mean(dim=0)
        
        return out


class GraphControlNet(nn.Module):
    """
    Red de control basada en GAT para corrección de fuerzas
    
    Arquitectura:
    - Input normalization
    - 2 capas GAT con conexiones residuales
    - MLP para predicción de delta_F
    - Cabeza de confianza para auto-evaluación
    
    Features de entrada (16 dimensiones):
    - Posición estimada (2)
    - Velocidad (2)  
    - Error de formación (2)
    - Traza de covarianza (1)
    - Velocidad escalar (1)
    - Probabilidad NLOS local (1)
    - Velocidad relativa al líder (2)
    - Aceleración estimada (2)
    - Error histórico normalizado (1)
    - Número de vecinos normalizado (1)
    - Aceleración del líder normalizada (1)
    """
    
    def __init__(self, in_features: int = 16, hidden_dim: int = 128, 
                 out_dim: int = 2, num_heads: int = 4, num_layers: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # Normalización de entrada
        self.input_norm = nn.LayerNorm(in_features)
        
        # Capas GAT con normalización
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        self.gat_layers.append(
            GATLayer(in_features, hidden_dim, num_heads, dropout, concat=True)
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATLayer(hidden_dim, hidden_dim, num_heads, dropout, concat=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Proyección residual
        self.input_proj = nn.Linear(in_features, hidden_dim) \
                         if in_features != hidden_dim else nn.Identity()
        
        # MLP de salida
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Cabeza de confianza
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.confidence_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, 
                return_confidence: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass de la red de control.
        
        Args:
            h: Features de nodos [N, in_features]
            adj_mask: Matriz de adyacencia [N, N]
            return_confidence: Si retornar también confianza
            
        Returns:
            deltaF: Corrección de fuerza [N, 2]
            confidence: (opcional) Confianza por nodo [N, 1]
        """
        # Normalizar entrada
        h = self.input_norm(h)
        h_res = self.input_proj(h)
        
        # Capas GAT con residuales
        h_out = None
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h if i == 0 else h_out, adj_mask)
            h_new = norm(h_new)
            
            if i == 0:
                h_out = F.gelu(h_new + h_res)
            else:
                h_out = F.gelu(h_new + h_out)
        
        # Predicción
        deltaF = self.mlp(h_out)
        
        if return_confidence:
            confidence = self.confidence_head(h_out)
            return deltaF, confidence
        
        return deltaF


def create_gat_model(params) -> GraphControlNet:
    """Crea una instancia del modelo GAT con los parámetros dados"""
    return GraphControlNet(
        in_features=params.neural.in_features,
        hidden_dim=params.neural.gat_hidden_dim,
        num_heads=params.neural.gat_num_heads,
        num_layers=params.neural.gat_num_layers
    )


def load_gat_model(state_dict: dict, params) -> GraphControlNet:
    """Carga un modelo GAT desde un state_dict"""
    model = create_gat_model(params)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_gat_model(model: GraphControlNet, path: str):
    """Guarda el modelo GAT"""
    torch.save(model.state_dict(), path)
