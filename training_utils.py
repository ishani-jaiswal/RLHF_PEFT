import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import numpy as np
from typing import Dict, Any, Optional
import wandb
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any, Optional
import ast
import re

def calculate_code_metrics(code: str) -> Dict[str, float]:
    """Calculate code quality metrics for a given piece of code.
    
    Args:
        code (str): The source code to analyze
        
    Returns:
        Dict[str, float]: A dictionary containing various code quality metrics
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Initialize metrics
        metrics = {
            'cyclomatic_complexity': 0.0,  # Number of decision points + 1
            'lines_of_code': len(code.split('\n')),  # Raw LOC
            'doc_coverage': 0.0,  # Percentage of documented functions/classes
            'naming_quality': 0.0,  # Score based on naming conventions
            'code_to_comment_ratio': 0.0  # Ratio of code to comments
        }
        
        # Count decision points for cyclomatic complexity
        decision_points = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                decision_points += 1
        metrics['cyclomatic_complexity'] = 1.0 / (decision_points + 1)  # Normalize
        
        # Calculate documentation coverage
        total_funcs = 0
        documented_funcs = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total_funcs += 1
                if ast.get_docstring(node):
                    documented_funcs += 1
        metrics['doc_coverage'] = documented_funcs / max(total_funcs, 1)
        
        # Evaluate naming quality
        good_names = 0
        total_names = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Name)):
                total_names += 1
                name = node.name if hasattr(node, 'name') else node.id
                if re.match(r'^[a-z][a-z0-9_]*$', name) or re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                    good_names += 1
        metrics['naming_quality'] = good_names / max(total_names, 1)
        
        # Calculate code to comment ratio
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        code_lines = metrics['lines_of_code'] - comment_lines
        metrics['code_to_comment_ratio'] = min(comment_lines / max(code_lines, 1), 1.0)
        
        # Normalize lines of code
        metrics['lines_of_code'] = 1.0 / max(metrics['lines_of_code'], 1)  # Prefer shorter code
        
        return metrics
    except Exception as e:
        # Return default metrics if parsing fails
        return {
            'cyclomatic_complexity': 0.5,
            'lines_of_code': 0.5,
            'doc_coverage': 0.5,
            'naming_quality': 0.5,
            'code_to_comment_ratio': 0.5
        }

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        # Handle invalid values
        if val_loss is None or np.isnan(val_loss) or np.isinf(val_loss):
            return False
            
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop

class ATRLHFTrainingManager:
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        num_epochs: int,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.1,
        scheduler_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=learning_rate)
        
        # Initialize scheduler
        if scheduler_kwargs is None:
            scheduler_kwargs = {"T_max": num_epochs, "eta_min": 1e-6}
        else:
            # Remove warmup steps if present as it's not used by CosineAnnealingLR
            scheduler_kwargs.pop('num_warmup_steps', None)
            scheduler_kwargs.pop('num_training_steps', None)
            if 'T_max' not in scheduler_kwargs:
                scheduler_kwargs['T_max'] = num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * last_gae
        
        return advantages

    def train_step(self, prompts: List[str]) -> Tuple[float, float, float]:
        print("\n=== Starting Response Generation Phase ===")
        # Generate responses using current policy
        all_responses = []
        all_rewards = []
        all_confidences = []
        all_feedback_rewards = []
        
        # Progress bar for response generation
        gen_pbar = tqdm(prompts, desc="Generating responses", leave=False)
        for prompt_idx, prompt in enumerate(gen_pbar):
            print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}")
            
            # CHECKPOINT A: Tokenization
            print("\nCHECKPOINT A - Prompt Tokenization:")
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.policy_model.device)
                print(f"Input shape: {inputs['input_ids'].shape}")
                print(f"Max token value: {inputs['input_ids'].max().item()}")
                print(f"Contains NaN: {torch.isnan(inputs['input_ids'].float()).any()}")
            except Exception as e:
                print(f"ERROR in tokenization: {str(e)}")
                continue
            
            # First pass: get baseline response
            try:
                print("\nCHECKPOINT B - Baseline Generation:")
                with torch.no_grad():
                    base_outputs = self.policy_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=200,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    print(f"Base output shape: {base_outputs.shape}")
                    print(f"Base output contains NaN: {torch.isnan(base_outputs.float()).any()}")
                    
                    base_response = self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)
                    print(f"Base response length: {len(base_response)}")
                    if not base_response.strip():  # Skip if empty response
                        print("WARNING: Empty base response generated, skipping")
                        continue
                    
                    print("\nCHECKPOINT C - Policy Generation:")
                    outputs = self.policy_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=200,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    print(f"Policy output shape: {outputs.shape}")
                    print(f"Policy output contains NaN: {torch.isnan(outputs.float()).any()}")
                    
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Response length: {len(response)}")
                    if not response.strip():  # Skip if empty response
                        print("WARNING: Empty policy response generated, skipping")
                        continue
            except Exception as e:
                print(f"ERROR in generation: {str(e)}")
                continue
            
            # Get rewards from reward model
            try:
                print("\nCHECKPOINT D - Reward Calculation:")
                with torch.no_grad():
                    # Get reward for current response
                    print("D1. Processing policy response:")
                    reward_inputs = self.tokenizer(
                        response,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.policy_model.device)
                    print(f"Reward input shape: {reward_inputs['input_ids'].shape}")
                    
                    reward_outputs = self.reward_model(
                        input_ids=reward_inputs["input_ids"],
                        attention_mask=reward_inputs["attention_mask"]
                    )
                    print(f"Raw reward output: {reward_outputs['reward']}")
                    print(f"Raw confidence output: {reward_outputs['confidence']}")
                    
                    reward = reward_outputs["reward"].mean().item()
                    confidence = reward_outputs["confidence"].mean().item()
                    print(f"Mean reward: {reward:.4f}, Mean confidence: {confidence:.4f}")
                    
                    # Get baseline reward
                    print("\nD2. Processing baseline response:")
                    base_reward_inputs = self.tokenizer(
                        base_response,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.policy_model.device)
                    print(f"Base reward input shape: {base_reward_inputs['input_ids'].shape}")
                    
                    base_reward_outputs = self.reward_model(
                        input_ids=base_reward_inputs["input_ids"],
                        attention_mask=base_reward_inputs["attention_mask"]
                    )
                    print(f"Raw base reward output: {base_reward_outputs['reward']}")
                    
                    base_reward = base_reward_outputs["reward"].mean().item()
                    print(f"Mean base reward: {base_reward:.4f}")
                    
                    # Calculate feedback reward
                    feedback_reward = reward - base_reward
                    print(f"Feedback reward: {feedback_reward:.4f}")
                    
                    # Skip if any rewards are invalid
                    if any(np.isnan([reward, base_reward, feedback_reward, confidence])):
                        print("WARNING: Invalid rewards detected:")
                        print(f"  Reward: {reward}")
                        print(f"  Base reward: {base_reward}")
                        print(f"  Feedback reward: {feedback_reward}")
                        print(f"  Confidence: {confidence}")
                        continue
                    
                    # Store valid results
                    all_responses.append(response)
                    all_rewards.append(reward)
                    all_confidences.append(confidence)
                    all_feedback_rewards.append(feedback_reward)
                    
                    gen_pbar.set_postfix({
                        'reward': f'{reward:.3f}',
                        'feedback': f'{feedback_reward:.3f}'
                    })
            except Exception as e:
                print(f"Error calculating rewards: {e}")
                continue
        
        # Convert lists to tensors
        rewards = torch.tensor(all_rewards, dtype=torch.float32).to(self.policy_model.device)
        feedback_rewards = torch.tensor(all_feedback_rewards, dtype=torch.float32).to(self.policy_model.device)
        confidences = torch.tensor(all_confidences, dtype=torch.float32).to(self.policy_model.device)
        
        # Handle empty or invalid rewards
        if len(rewards) == 0 or torch.isnan(rewards).any():
            rewards = torch.zeros(1, device=self.policy_model.device)
        if len(feedback_rewards) == 0 or torch.isnan(feedback_rewards).any():
            feedback_rewards = torch.zeros(1, device=self.policy_model.device)
        if len(confidences) == 0 or torch.isnan(confidences).any():
            confidences = torch.ones(1, device=self.policy_model.device)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards + feedback_rewards, confidences)
        
        # PPO training loop
        policy_loss = 0.0
        value_loss = 0.0
        
        # Progress bar for PPO updates
        ppo_pbar = tqdm(range(self.ppo_epochs), desc="PPO iterations", leave=False)
        for _ in ppo_pbar:
            batch_losses = []
            # Inner loop progress bar
            inner_pbar = tqdm(enumerate(prompts), total=len(prompts), desc="Processing batches", leave=False)
            for i, prompt in inner_pbar:
                try:
                    # CHECKPOINT 1: Input data
                    print(f"\nCHECKPOINT 1 - Input Data for batch {i}:")
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.policy_model.device)
                    print(f"Input shape: {inputs['input_ids'].shape}")
                    print(f"Contains NaN: {torch.isnan(inputs['input_ids'].float()).any()}")
                    
                    # CHECKPOINT 2: Model outputs
                    outputs = self.policy_model(**inputs)
                    logits = outputs.logits
                    print(f"\nCHECKPOINT 2 - Model Outputs:")
                    print(f"Logits shape: {logits.shape}")
                    print(f"Logits NaN: {torch.isnan(logits).any()}")
                    
                    # CHECKPOINT 3: Policy computation
                    old_policy = F.softmax(logits.detach(), dim=-1)
                    new_policy = F.softmax(logits, dim=-1)
                    ratio = (new_policy / (old_policy + 1e-8)).mean()
                    print(f"\nCHECKPOINT 3 - Policy Values:")
                    print(f"Old policy NaN: {torch.isnan(old_policy).any()}")
                    print(f"New policy NaN: {torch.isnan(new_policy).any()}")
                    print(f"Ratio NaN: {torch.isnan(ratio).any()}")
                    
                    # CHECKPOINT 4: Advantage and reward check
                    print(f"\nCHECKPOINT 4 - Advantages and Rewards:")
                    print(f"Advantage[{i}] shape: {advantages[i].shape}")
                    print(f"Advantage NaN: {torch.isnan(advantages[i]).any()}")
                    print(f"Rewards[{i}] NaN: {torch.isnan(rewards[i]).any() if i < len(rewards) else 'Index out of range'}")
                    
                    # Compute losses with NaN checks
                    policy_loss = -torch.min(
                        ratio * advantages[i],
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                    ).mean()
                    
                    value_loss = F.mse_loss(confidences[i], rewards[i] + feedback_rewards[i])
                    
                    # CHECKPOINT 5: Loss computation
                    print(f"\nCHECKPOINT 5 - Loss Values:")
                    print(f"Policy loss: {policy_loss.item():.4f} (NaN: {torch.isnan(policy_loss).any()})")
                    print(f"Value loss: {value_loss.item():.4f} (NaN: {torch.isnan(value_loss).any()})")
                    
                    # Combined loss
                    loss = policy_loss + self.vf_coef * value_loss
                    
                    # Skip if loss is NaN
                    if torch.isnan(loss).any():
                        print(f"WARNING: NaN loss detected in batch {i}, skipping update")
                        continue
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # CHECKPOINT 6: Gradient check
                    print(f"\nCHECKPOINT 6 - Gradients:")
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                    print(f"Gradient norm before clipping: {grad_norm:.4f}")
                    
                    self.optimizer.step()
                    
                    # Update progress bars
                    batch_loss = loss.item()
                    batch_losses.append(batch_loss)
                    inner_pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'avg_loss': f'{np.mean(batch_losses):.4f}'
                    })
                except Exception as e:
                    print(f"\nERROR in batch {i}: {str(e)}")
                    continue
            
            ppo_pbar.set_postfix({
                'avg_loss': f'{np.mean(batch_losses):.4f}'
            })
        
        # Ensure we have valid values to return
        try:
            p_loss = policy_loss.item() if not torch.isnan(policy_loss).any() else 0.0
            v_loss = value_loss.item() if not torch.isnan(value_loss).any() else 0.0
            r_mean = (rewards + feedback_rewards).mean().item() if not torch.isnan(rewards + feedback_rewards).any() else 0.0
            return p_loss, v_loss, r_mean
        except:
            return 0.0, 0.0, 0.0

    def validate(self, val_prompts: List[str]) -> Tuple[float, float]:
        self.policy_model.eval()
        val_rewards = []
        val_confidences = []
        
        with torch.no_grad():
            for prompt in tqdm(val_prompts, desc="Validating", leave=False):
                try:
                    # Generate response
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.policy_model.device)
                    outputs = self.policy_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=200,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Get reward
                    reward_inputs = self.tokenizer(
                        response,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.policy_model.device)
                    
                    reward_outputs = self.reward_model(
                        input_ids=reward_inputs["input_ids"],
                        attention_mask=reward_inputs["attention_mask"]
                    )
                    
                    val_rewards.append(reward_outputs["reward"].mean().item())
                    val_confidences.append(reward_outputs["confidence"].mean().item())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
        
        self.policy_model.train()
        
        # Return mean reward and confidence
        if len(val_rewards) > 0:
            return np.mean(val_rewards), np.mean(val_confidences)
        return 0.0, 0.0

    def train(self, train_prompts: List[str], val_prompts: List[str]):
        print(f"Starting training for {self.num_epochs} epochs...")
        global_step = 0
        total_steps = self.num_epochs * len(train_prompts)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            self.policy_model.train()
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_rewards = []
            
            # Process prompts in batches
            for i in range(0, len(train_prompts), self.batch_size):
                batch_prompts = train_prompts[i:i + self.batch_size]
                policy_loss, value_loss, reward = self.train_step(batch_prompts)
                
                epoch_policy_losses.append(policy_loss)
                epoch_value_losses.append(value_loss)
                epoch_rewards.append(reward)
                
                global_step += 1
                
                # Log every 10 steps
                if global_step % 10 == 0:
                    print(f"Step {global_step}/{total_steps}:")
                    print(f"  Policy Loss: {policy_loss:.4f}")
                    print(f"  Value Loss: {value_loss:.4f}")
                    print(f"  Mean Reward: {reward:.4f}")
            
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            avg_reward = np.mean(epoch_rewards)
            
            # Validation phase
            print("\nRunning validation...")
            val_reward, val_confidence = self.validate(val_prompts)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/policy_loss': avg_policy_loss,
                    'train/value_loss': avg_value_loss,
                    'train/mean_reward': avg_reward,
                    'val/reward': val_reward,
                    'val/confidence': val_confidence,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Print epoch summary
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Train Value Loss: {avg_value_loss:.4f}")
            print(f"  Train Mean Reward: {avg_reward:.4f}")
            print(f"  Val Reward: {val_reward:.4f}")
            print(f"  Val Confidence: {val_confidence:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model based on validation reward
            if val_reward > self.best_val_loss:  # Using reward instead of loss
                self.best_val_loss = val_reward
                self.best_model_state = {
                    'policy_model': self.policy_model.state_dict(),
                    'reward_model': self.reward_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'val_reward': val_reward,
                    'val_confidence': val_confidence
                }
                print("  Saved new best model!")
            
            # Early stopping check
            if self.early_stopping(-val_reward):  # Negative since we want to maximize reward
                print("Early stopping triggered!")
                break
        
        return self.best_model_state