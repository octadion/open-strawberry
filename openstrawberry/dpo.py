from collections.abc import Iterator
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn import Module
from x_transformers.x_transformers import TransformerWrapper

from einops import rearrange

def exists(v):
    return v is not None

def freeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False

def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_prob = logits.log_softmax(dim=-1)
    indices = rearrange(seq, "... -> ... 1")
    log_probs = log_prob.gather(-1, indices)
    return rearrange(log_probs, "... 1 -> ...")

def masked_mean(log_probs, mask=None):
    if not exists(mask):
        return log_probs.mean(dim=-1)
    
    log_probs = log_probs.masked_fill(~mask, 0.0)
    num = log_probs.sum(dim=-1)
    den = mask.sum(dim=-1)
    return num / den.clamp(min=1e-5)

def maybe_and_mask(*masks):
    masks = [*filter(exists, masks)]
    if len(masks) == 0:
        return None
    
    mask, *rest_masks = masks
    for rest_mask in rest_masks:
        mask = mask & rest_mask
    
    return mask

class DPO(Module):
    
    def __init__(
        self,
        model: TransformerWrapper,
        *,
        beta: float = 0.1,
        pad_id: int = None,
    ):
        super().__init__()
        self.policy_model = model
        self.ref_model = deepcopy(model)
        
        freeze_all_layers_(self.ref_model)

        self.beta = beta
        self.pad_id = pad_id

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.policy_model.parameters()
    
    def forward(
        self,
        preferred_seq: torch.Tensor,
        unpreferred_seq: torch.Tensor,
        *,
        prompt_mask: torch.Tensor,
        preferred_seq_mask: torch.Tensor = None,
        unpreferred_seq_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        assert preferred_seq.ndim == 2
        assert preferred_seq.shape == unpreferred_seq.shape

        if exists(self.pad_id):
            if not exists(preferred_seq_mask):
                preferred_seq_mask = preferred_seq != self.pad_id

            if not exists(unpreferred_seq_mask):
                unpreferred_seq_mask = unpreferred_seq != self.pad_id

        with torch.no_grad():
            self.ref_model.eval()
            ref_preferred_logprob = log_prob_from_model_and_seq(
                self.ref_model, preferred_seq
            )

            ref_unpreferred_logprob = log_prob_from_model_and_seq(
                self.ref_model, unpreferred_seq
            )

        policy_preferred_logprob = log_prob_from_model_and_seq(
            self.policy_model, preferred_seq
        )

        policy_unpreferred_logprob = log_prob_from_model_and_seq(
            self.policy_model, unpreferred_seq
        )

        preferred_seq_mask = maybe_and_mask(
            ~prompt_mask, preferred_seq_mask
        )

        unpreferred_seq_mask = maybe_and_mask(
            ~prompt_mask, unpreferred_seq_mask
        )

        ref_preferred_logprob, policy_preferred_logprob = map(
            lambda t: masked_mean(t, preferred_seq_mask),
            (ref_preferred_logprob, policy_preferred_logprob),
        )

        ref_unpreferred_logprob, policy_unpreferred_logprob = map(
            lambda t: masked_mean(t, unpreferred_seq_mask),
            (ref_unpreferred_logprob, policy_unpreferred_logprob),
        )

        policy_logratios = (
            policy_preferred_logprob - policy_unpreferred_logprob
        )

        ref_logratios = (
            ref_preferred_logprob - ref_unpreferred_logprob
        )

        losses = -F.logsigmoid(
            self.beta * (policy_logratios - ref_logratios)
        )

        return losses.mean()