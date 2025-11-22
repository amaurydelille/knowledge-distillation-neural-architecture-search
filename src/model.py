import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Tuple, List
import torch.nn.functional as F
MAX_LAYERS = 12


class ElasticBERT(nn.Module):
    """
    BERT student model capable of dynamically adjust its architecture based on
    the Controller's output actions
    """
    def __init__(self, teacher_config: BertConfig, max_layer: int = MAX_LAYERS) -> None:
        super().__init__()
        self.config = teacher_config
        self.max_layer = max_layer

        self.student_config = BertConfig(
            vocab_size=teacher_config.vocab_size,
            hidden_size=teacher_config.hidden_size,
            num_hidden_layers=max_layer,
            num_attention_heads=teacher_config.num_attention_heads,
            intermediate_size=teacher_config.intermediate_size
        ) # could be even more customized with dropout prob, etc.

        self.bert = BertModel(config=self.student_config)

        # learnable projection to match teacher hidden size if we reduce student size
        self.fit_dense = nn.Linear(self.student_config.hidden_size, teacher_config.hidden_size)

    def forward(self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        active_layers =  None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # while search phase we only use the first n layers.
        if active_layers is not None and active_layers < self.student_config.num_hidden_layers:
            final_hidden = outputs.hidden_states[active_layers]
        else:
            final_hidden = outputs.last_hidden_state

        return final_hidden, outputs.hidden_states

class KDLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, layer_mapping: List[int]) -> torch.Tensor:
        p_s = F.log_softmax(student_logits / self.T, dim=-1)
        p_t = F.softmax(teacher_logits / self.T, dim=-1)
        loss_kd = self.kl_div(p_s, p_t) * (self.T ** 2)

        loss_hidden = 0.0
        for s_idx, t_idx in enumerate(layer_mapping):
            loss_hidden += self.mse(student_hidden[s_idx], teacher_hidden[t_idx])

        return (self.alpha * loss_kd) + (1 - self.alpha) * loss_hidden

def get_layer_mapping(num_student_layers: int, num_teacher_layers: int) -> List[int]:
    """
    Uniformly maps student layers to teacher layers
    """
    return [int(i * (num_teacher_layers / num_student_layers)) for i in range(1, num_student_layers + 1)]