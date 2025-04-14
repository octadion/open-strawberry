import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from loguru import logger
from typing import List, Tuple

logger.add("training.log", rotation="500 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerPolicyNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerPolicyNetwork, self).__init__()
        self.model_type = "Transformer"

        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        self.fc_out = nn.Linear(dim_feedforward, action_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]
        action_logits = self.fc_out(output)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs
    
class TransformerValueNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super(TransformerValueNetwork, self).__init__()
        self.model_type = "Transformer"

        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        self.fc_out = nn.Linear(dim_feedforward, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]
        state_value = self.fc_out(output)
        return state_value
    
class TransformerRewardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerRewardModel, self).__init__()
        self.model_type = "Transformer"
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        self.fc_out = nn.Linear(dim_feedforward, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]
        reward = self.fc_out(output)
        return reward
    
class ThoughtTree:

    def __init__(self, root_state: torch.Tensor):
        self.root = {"state": root_state, "children": [], "reward": 0}

    def add_child(
        self, parent: dict, child_state: torch.Tensor, reward: float
    ):
        child = {
            "state": child_state,
            "children": [],
            "reward": reward
        }
        parent["children"].append(child)
        return child
    
def transition(
    state: torch.Tensor,
    action: torch.Tensor
) -> torch.Tensor:
    next_state = state + action.float()
    return next_state

def reward_function(
    state: torch.Tensor
) -> float:
    reward = -torch.sum(state**2).item()
    return reward
    
def monte_carlo_rollout(
    policy_net: TransformerPolicyNetwork,
    state_sequence: torch.Tensor,
    depth: int,
    max_depth: int,
    sequence_length: int,
) -> List[Tuple[torch.Tensor, float]]:
    
    trajectory = []
    current_sequence = state_sequence.clone()
    for _ in range(depth, max_depth):
        action_probs = policy_net(current_sequence)
        m = Categorical(action_probs)
        action = m.sample()
        next_state = transition(current_sequence[-1], action)
        next_sequence = torch.cat(
            [current_sequence, next_state.unsqueeze(0)], dim=0
        )
        if next_sequence.size(0) > sequence_length:
            next_sequence = next_sequence[1:, :]
        reward = reward_function(next_state)
        trajectory.append((next_sequence, reward))
        current_sequence = next_sequence
    
    return trajectory

def train():
    pass