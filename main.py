import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from typing import Dict, Any, Optional, List, Tuple
from datasets import load_dataset
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import re
import json
import tempfile
import ast
from radon.complexity import cc_visit
from pylint.lint import Run
from pylint.reporters import JSONReporter
from io import StringIO

# === Model Setup and Initialization ===
login(token="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")

# Load and prepare datasets
def prepare_datasets():
    code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k")
    human_eval = load_dataset("openai/human-eval")
    mbpp = load_dataset("mbpp")
    
    train_dataset = concatenate_datasets([
        code_alpaca["train"], 
        human_eval["train"],
        mbpp["train"]
    ])
    
    eval_dataset = concatenate_datasets([
        human_eval["test"],
        mbpp["test"]
    ])
    
    return train_dataset, eval_dataset

# Enhanced reward model for code generation
class CodeRewardModelWithConfidence(nn.Module):
    def __init__(self, model, tokenizer, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, use_flash_attention_2=False):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Quality prediction heads
        self.reward_head = nn.Sequential(
            nn.Linear(model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        ).to(self.device)
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, input_ids, attention_mask=None, return_conf=True):
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        last_hidden = outputs.hidden_states[-1]
        
        # Pool the hidden states (mean pooling)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            last_hidden = last_hidden * attention_mask
            pooled = last_hidden.sum(1) / attention_mask.sum(1)
        else:
            pooled = last_hidden.mean(1)
        
        # Get reward prediction
        reward = self.reward_head(pooled)
        
        if return_conf:
            # Get confidence score
            confidence = self.confidence_head(pooled)
            return reward, confidence
        
        return reward

def calculate_code_quality(code: str) -> float:
    """Calculate code quality using pylint."""
    try:
        # Save code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Create a StringIO to capture the output
        output = StringIO()
        reporter = JSONReporter(output)

        # Run pylint with JSON reporter
        Run([temp_path], reporter=reporter, exit=False)
        
        # Get the score from JSON output
        result = json.loads(output.getvalue())
        if result:
            score = result[0].get('score', 0.0)
        else:
            score = 0.0
        
        # Clean up
        os.unlink(temp_path)
        
        # Normalize score to 0-1 range
        normalized_score = score / 10.0  # pylint scores are now 0-10
        return max(0.0, min(1.0, normalized_score))  # Clamp between 0 and 1
    except Exception as e:
        print(f"Error calculating code quality: {e}")
        return 0.5  # Default score for failed analysis

# Model initialization
policy_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_flash_attention_2=False
)
policy_model.gradient_checkpointing_enable()

# === Training Functions ===
def pretrain_reward_model(reward_model, train_dataset, num_epochs=3):
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
    
    for epoch in range(num_epochs):
        for batch in DataLoader(train_dataset, batch_size=8):
            code = batch['code']
            
            # Calculate ground truth metrics
            syntax, complexity, lint = calculate_code_metrics(code)
            
            # Get model predictions
            inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(reward_model.device)
            predicted_reward, predicted_confidence = reward_model(inputs.input_ids, inputs.attention_mask)
            
            # Compute loss
            metrics_tensor = torch.tensor([syntax, complexity, lint]).mean().to(reward_model.device)
            loss = F.mse_loss(predicted_reward, metrics_tensor)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log training progress
            if wandb.run is not None:
                wandb.log({
                    'train_loss': loss.item(),
                    'predicted_reward': predicted_reward.mean().item(),
                    'predicted_confidence': predicted_confidence.mean().item()
                })