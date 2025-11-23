import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def evaluate_accuracy(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating accuracy"):
            inputs, masks = batch['input_ids'], batch['attention_mask']
            outputs = model(inputs, attention_mask=masks)
            correct += (outputs.argmax(dim=1) == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    return correct / total

def measure_latency(model: nn.Module, inputs: torch.Tensor, attention_mask: torch.Tensor) -> int:
    start_time = time.time()
    model.forward(inputs, attention_mask=attention_mask)
    end_time = time.time()
    return (end_time - start_time) * 1000 # in milliseconds