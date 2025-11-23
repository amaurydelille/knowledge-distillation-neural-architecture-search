import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Tuple, List, Dict
import torch.nn.functional as F

from utils import evaluate_accuracy, measure_latency
MAX_LAYERS = 12
HIDDEN_DIM = 128


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

class Controller(nn.Module):
    def __init__(self, search_space_dims: Dict[str, int], hidden_dim=HIDDEN_DIM) -> None:
        super().__init__()
        self.search_space = search_space_dims
        self.num_layers_options = len(search_space_dims['num_layers'])

        self.lstm = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.decoder_layers = nn.Linear(in_features=hidden_dim, out_features=self.num_layers_options)
        self.embedding = nn.Embedding(
            num_embeddings=self.num_layers_options, embedding_dim=hidden_dim)

        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.decoder_layers.weight, -0.1, 0.1)

    def forward(self, prev_action_idx: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(prev_action_idx)
        h_t, c_t = self.lstm(embedded, (h_t, c_t))
        decoder_logits = self.decoder_layers(h_t)
        return decoder_logits, h_t, c_t

    def sample_architecture(self) -> Tuple[Dict[str, int], torch.Tensor]:
        h_t = torch.zeros(1, self.hidden_dim)
        c_t = torch.zeros(1, self.hidden_dim)

        prev_action = torch.zeros(1, dtype=torch.long)

        logits, h_t, c_t = self(prev_action, h_t, c_t)
        probs = F.softmax(logits, dim=-1)
        action_idx = probs.multinomial(num_samples=1)

        num_layers = self.search_space['num_layers'][action_idx.items()]

        log_prob = torch.log(probs.gather(1, action_idx))

        return { 'num_layers': num_layers }, log_prob

def train_kd_nas(teacher_model, train_loader, val_loader, episodes=100, device: str = 'mps' | 'cuda' | 'cpu'):
    teacher_model.eval()
    search_space = {
        'num_layers': [4, 6, 8, 10, 12],
        'hidden_dim': [128, 256, 512],
        'dropout_prob': [0.1, 0.2, 0.3],
        'attention_heads': [1, 2, 4],
        'intermediate_size': [256, 512, 1024],
        'max_position_embeddings': [512, 1024, 2048],
    }

    controller = Controller(search_space).to(device)
    controller_optim = torch.optim.Adam(controller.parameters(), lr=0.001)
    
    kd_criterion = KDLoss().to(device)
    
    baseline_reward = 0.0
    
    for episode in range(episodes):
        arch_config, log_prob = controller.sample_architecture()
        num_layers = arch_config['num_layers']
        
        student_model = ElasticBERT(teacher_model.config, max_layers=12).to(device)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)
        
        print(f"Episode {episode}: Training Student with {num_layers} layers...")
        student_model.train().to(device)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 50:
                break
            
            inputs, masks = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            
            with torch.no_grad():
                t_out = teacher_model(inputs, attention_mask=masks, output_hidden_states=True)
                t_logits, t_hidden = t_out.logits, t_out.hidden_states
            
            s_out, s_hidden = student_model(inputs, attention_mask=masks, active_layers=num_layers)
            
            mapping = get_layer_mapping(num_layers, 12)
            loss = kd_criterion(s_out, t_logits, s_hidden, t_hidden, mapping)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        acc = evaluate_accuracy(student_model, val_loader)
        latency = measure_latency(student_model, inputs, masks)
        
        reward = acc - 0.1 * latency
        
        print(f"Result: Acc={acc:.2f}, Reward={reward:.4f}")
        
        policy_loss = -log_prob * (reward - baseline_reward)
        
        controller_optim.zero_grad()
        policy_loss.backward()
        controller_optim.step()
        
        baseline_reward = 0.9 * baseline_reward + 0.1 * reward