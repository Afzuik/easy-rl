from __future__ import annotations
import random
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
        )
