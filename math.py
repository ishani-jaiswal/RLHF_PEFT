import torch
import random
import os
import numpy as np
import re
from tqdm import tqdm
import json
import time
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for model checkpoints
checkpoint_dir = "llama3_math_reasoning_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Define functions for data preparation
def prepare_gsm8k_sample(sample):
    """Enhanced prompt template for LLaMA 3 format"""
    return {
        "prompt": f"""[INST] Solve this step-by-step:
Problem: {sample['question']}

Follow these steps:
1. Identify the key information
2. Break down the problem
3. Show all calculations clearly
4. Verify your answer [/INST]

""",
        "solution": sample['answer']
    }

def extract_answer(solution):
    """Extract the final numerical answer from a solution with improved pattern matching"""
    if isinstance(solution, str):        
        # For GSM8K format with "The answer is" pattern
        answer_pattern = r"[Tt]he\s+answer\s+is\s+([-+]?\d*\.?\d+)"
        match = re.search(answer_pattern, solution)
        if match:
            return match.group(1)
        
        # Try finding answer after "=" at the end of text
        equal_pattern = r"=\s*([-+]?\d*\.?\d+)(?:[^\d]|$)"
        matches = re.findall(equal_pattern, solution)
        if matches:
            return matches[-1]
        
        # Last resort - try to find the last number in the text
        numbers = re.findall(r'[-+]?\d*\.?\d+', solution)
        if numbers:
            return numbers[-1]
    
    return None

def get_generation_config():
    return {
        'max_new_tokens': 512,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 50,
        'repetition_penalty': 1.2,
        'eos_token_id': None,
    }

def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, best_idpo_score, 
                   patience_counter, epoch_scores, checkpoint_dir):
    """Save a checkpoint of the model and training state"""
    # Create a new directory for this training run
    training_run_dir = os.path.join(checkpoint_dir, f"training_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(training_run_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "best_idpo_score": best_idpo_score,
        "patience_counter": patience_counter,
        "epoch_scores": epoch_scores,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "training_params": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay
        }
    }
    
    # Save the checkpoint with timestamp
    checkpoint_path = os.path.join(training_run_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save model with better organization
    model_dir = os.path.join(training_run_dir, f"model_epoch_{epoch}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(
        model_dir,
        save_function=torch.save,
        max_shard_size="5GB"  # Increased shard size for LLaMA 3
    )
    tokenizer.save_pretrained(model_dir)
    
    print(f"Checkpoint saved at {model_dir}")

class MathReasoningDataset(Dataset):
    """Dataset for math reasoning tasks with LLaMA 3 format"""
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-process all examples to reduce training time
        self.processed_data = []
        for item in data:
            # Create input tensors with proper padding
            inputs = tokenizer(
                item["prompt"], 
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create output tensors with proper padding
            full_sequence = item["prompt"] + item["solution"]
            outputs = tokenizer(
                full_sequence,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create attention mask that masks the prompt tokens
            prompt_len = len(inputs["input_ids"][0])
            labels = outputs["input_ids"][0].clone()
            # Set prompt tokens to -100 (ignored in loss)
            labels[:prompt_len] = -100
            
            self.processed_data.append({
                "input_ids": outputs["input_ids"][0],
                "attention_mask": outputs["attention_mask"][0],
                "labels": labels,
                "raw_prompt": item["prompt"],
                "raw_solution": item["solution"]
            })
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def collate_fn(batch):
    """Collate function for the dataloader with better handling of tensors"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Also return raw text for evaluation
    raw_prompts = [item["raw_prompt"] for item in batch]
    raw_solutions = [item["raw_solution"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "raw_prompts": raw_prompts,
        "raw_solutions": raw_solutions
    }

# IDPO Implementation
class IDPOTrainer:
    """IDPO (Implicit Direct Preference Optimization) Trainer class"""
    def __init__(self, model, tokenizer, beta=0.1, margin=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta  # KL penalty coefficient
        self.margin = margin  # Preference margin
        
    def compute_idpo_loss(self, model_outputs, input_ids, attention_mask, labels, 
                         chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        """
        Compute IDPO loss based on the paper: 
        "Direct Preference Optimization without Preference Data"
        """
        # Standard supervised loss
        supervised_loss = model_outputs.loss
        
        # Get model logits for chosen and rejected completions
        with torch.no_grad():
            chosen_outputs = self.model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
                labels=chosen_ids  # We'll mask unnecessary tokens later
            )
            rejected_outputs = self.model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
                labels=rejected_ids  # We'll mask unnecessary tokens later
            )
        
        # Extract logprobs for chosen and rejected
        chosen_logprobs = self.get_sequence_logprobs(chosen_outputs, chosen_ids)
        rejected_logprobs = self.get_sequence_logprobs(rejected_outputs, rejected_ids)
        
        # Compute the preference loss (simplified DPO loss)
        # log(σ(β(r_chosen - r_rejected - margin)))
        preference_loss = -torch.nn.functional.logsigmoid(
            self.beta * (chosen_logprobs - rejected_logprobs - self.margin)
        ).mean()
        
        # Total loss is a combination of supervised and preference loss
        total_loss = supervised_loss + preference_loss
        
        return {
            "loss": total_loss,
            "supervised_loss": supervised_loss,
            "preference_loss": preference_loss,
            "chosen_logprobs": chosen_logprobs.mean(),
            "rejected_logprobs": rejected_logprobs.mean()
        }
    
    def get_sequence_logprobs(self, outputs, input_ids):
        """Extract log probabilities for a sequence"""
        logits = outputs.logits[:, :-1, :]  # Remove last position
        targets = input_ids[:, 1:]  # Shift right to get targets
        
        # Create a mask to focus only on non-padding tokens
        mask = targets != self.tokenizer.pad_token_id
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probs of the targets
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # Apply mask and normalize
        token_log_probs = token_log_probs * mask.float()
        sequence_log_probs = token_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return sequence_log_probs
    
    def generate_paired_samples(self, batch):
        """Generate chosen and rejected samples from batch"""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Generate two samples with different temperatures
        with torch.no_grad():
            chosen_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.5,  # Lower temperature for chosen (higher quality)
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1
            )
            
            rejected_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=1.2,  # Higher temperature for rejected (lower quality)
                max_new_tokens=256,
                do_sample=True,
                top_p=0.98,
                top_k=80,
                repetition_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1
            )
        
        # Process the outputs
        chosen_ids = chosen_outputs
        chosen_mask = torch.ones_like(chosen_ids)
        rejected_ids = rejected_outputs
        rejected_mask = torch.ones_like(rejected_ids)
        
        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

def prepare_test_set():
    """Prepare the test dataset with better sampling"""
    # Load test data
    gsm8k_test = load_dataset("gsm8k", "main", split="test")
    
    # Balanced sampling across difficulty levels
    # We use a proxy for difficulty: solution length
    difficulty_buckets = {}
    
    for i, sample in enumerate(gsm8k_test):
        solution_length = len(sample['answer'])
        if solution_length < 100:
            bucket = "easy"
        elif solution_length < 200:
            bucket = "medium"
        else:
            bucket = "hard"
        
        if bucket not in difficulty_buckets:
            difficulty_buckets[bucket] = []
        difficulty_buckets[bucket].append(i)
    
    # Sample equally from each bucket
    max_test_samples = 100
    samples_per_bucket = max_test_samples // len(difficulty_buckets)
    
    balanced_indices = []
    for bucket, indices in difficulty_buckets.items():
        if len(indices) > samples_per_bucket:
            selected = random.sample(indices, samples_per_bucket)
        else:
            selected = indices
        balanced_indices.extend(selected)
    
    # If we need more samples to reach max_test_samples
    remaining = max_test_samples - len(balanced_indices)
    if remaining > 0:
        all_indices = list(range(len(gsm8k_test)))
        additional_indices = random.sample([i for i in all_indices if i not in balanced_indices], remaining)
        balanced_indices.extend(additional_indices)
    
    # Select test samples
    gsm8k_test = gsm8k_test.select(balanced_indices[:max_test_samples])
    
    # Prepare test data
    test_data = []
    for sample in gsm8k_test:
        test_data.append(prepare_gsm8k_sample(sample))
    
    return test_data

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # Load GSM8K dataset
    print("Loading datasets...")
    # Increase training samples
    datasets = {
        "gsm8k": load_dataset("gsm8k", "main", split="train"),  # Use full dataset
    }
    
    # Improve model configuration
    peft_config = LoraConfig(
        r=64,  # Increase LoRA rank
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",  # Changed to CAUSAL_LM for LLaMA 3
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]  # Target modules for LLaMA architecture
    )
    
    print(f"Loaded {len(datasets['gsm8k'])} GSM8K examples")
    
    # Sample a subset for training with stratified sampling by problem difficulty
    max_samples_per_dataset = 4000  # Adjusted for LLaMA 3 which may be more efficient
    if len(datasets['gsm8k']) > max_samples_per_dataset:
        # Use stratified sampling based on solution length as a proxy for difficulty
        all_samples = []
        for i, sample in enumerate(datasets['gsm8k']):
            solution_length = len(sample['answer'])
            all_samples.append((i, solution_length))
        
        # Sort by solution length and bucket
        all_samples.sort(key=lambda x: x[1])
        bucket_size = len(all_samples) // 3
        
        # Extract indices from each bucket (easy, medium, hard)
        easy_indices = [x[0] for x in all_samples[:bucket_size]]
        medium_indices = [x[0] for x in all_samples[bucket_size:2*bucket_size]]
        hard_indices = [x[0] for x in all_samples[2*bucket_size:]]
        
        # Sample from each bucket
        samples_per_bucket = max_samples_per_dataset // 3
        selected_easy = random.sample(easy_indices, samples_per_bucket)
        selected_medium = random.sample(medium_indices, samples_per_bucket)
        selected_hard = random.sample(hard_indices, samples_per_bucket)
        
        # Combine all selected indices
        gsm8k_indices = selected_easy + selected_medium + selected_hard
        datasets['gsm8k'] = datasets['gsm8k'].select(gsm8k_indices)
    
    print(f"Using {len(datasets['gsm8k'])} GSM8K examples for training")
    
    # Prepare the training data
    training_data = []
    for sample in datasets["gsm8k"]:
        prepared = prepare_gsm8k_sample(sample)
        training_data.append(prepared)
    
    print(f"Total training samples: {len(training_data)}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "meta-llama/Meta-Llama-3-8B"  # Updated to use HF model
    hf_token = "[enter access token]"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure model loading with optimizations for LLaMA 3
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with proper LLaMA configuration
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Configure training parameters optimized for LLaMA 3
    MAX_LENGTH = 1024  # Increased for LLaMA 3
    batch_size = 2  # Reduced for LLaMA 3 due to size
    gradient_accumulation_steps = 8  # Increased for LLaMA 3
    num_epochs = 5
    learning_rate = 5e-5
    warmup_ratio = 0.15
    weight_decay = 0.01
    patience = 3
    max_train_samples = min(2000, len(training_data))
    
    # Create IDPO trainer
    idpo_trainer = IDPOTrainer(model, tokenizer, beta=0.1, margin=0.0)
    
    # Prepare dataset with improved structure
    train_dataset = MathReasoningDataset(training_data[:max_train_samples], tokenizer, MAX_LENGTH)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Set up optimizer with better parameters for large models
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.95),
    )
    
    # Set up learning rate scheduler with proper warmup
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize test data before training loop
    test_data = prepare_test_set()
    print(f"Prepared {len(test_data)} test samples")
    
    # Training loop with gradient accumulation and mixed precision
    print("Starting training...")
    best_idpo_score = -float("inf")
    patience_counter = 0
    all_epoch_scores = []
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_scores = []
        total_loss = 0
        total_supervised_loss = 0
        total_preference_loss = 0
        
        # Training loop with progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda') if scaler else torch.no_grad():
                # Get model outputs for supervised learning
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Generate chosen and rejected samples for IDPO
                chosen_ids, chosen_mask, rejected_ids, rejected_mask = idpo_trainer.generate_paired_samples(batch)
                
                # Compute IDPO loss
                loss_dict = idpo_trainer.compute_idpo_loss(
                    outputs, input_ids, attention_mask, labels,
                    chosen_ids, chosen_mask, rejected_ids, rejected_mask
                )
                
                # Get losses
                loss = loss_dict["loss"] / gradient_accumulation_steps
                supervised_loss = loss_dict["supervised_loss"]
                preference_loss = loss_dict["preference_loss"]
                
                # Track chosen vs rejected logprobs for monitoring
                chosen_logprobs = loss_dict["chosen_logprobs"]
                rejected_logprobs = loss_dict["rejected_logprobs"]
                
                # Track preference margin
                preference_margin = chosen_logprobs - rejected_logprobs
                epoch_scores.append(preference_margin.item())
            
            # Backward pass with gradient scaling
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Update total loss statistics
            total_loss += loss.item() * gradient_accumulation_steps
            total_supervised_loss += supervised_loss.item()
            total_preference_loss += preference_loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent explosion
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                avg_supervised = total_supervised_loss / (batch_idx + 1)
                avg_preference = total_preference_loss / (batch_idx + 1)
                avg_margin = sum(epoch_scores[-8:]) / min(8, len(epoch_scores[-8:]))
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "sup_loss": f"{avg_supervised:.4f}",
                    "pref_loss": f"{avg_preference:.4f}",
                    "margin": f"{avg_margin:.4f}"
                })
        
        # Calculate average metrics for this epoch
        avg_epoch_loss = total_loss / len(train_dataloader)
        avg_epoch_score = sum(epoch_scores) / len(epoch_scores)
        all_epoch_scores.append(avg_epoch_score)
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_epoch_loss:.4f}, Average Preference Margin = {avg_epoch_score:.4f}")
        
        # Save checkpoint after every epoch
        save_checkpoint(model, tokenizer, optimizer, lr_scheduler, epoch, best_idpo_score, 
                       patience_counter, all_epoch_scores, checkpoint_dir)
        
        # Early stopping check based on preference margin
        if avg_epoch_score > best_idpo_score:
            best_idpo_score = avg_epoch_score
            patience_counter = 0
            # Save best model
            best_model_dir = os.path.join(checkpoint_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Save final model
    final_model_dir = "llama3_math_reasoning_final"
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Update final model saving with timestamp
    final_model_dir = os.path.join("llama3_math_reasoning_final", f"version_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        "model_name": model_name,
        "training_params": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "idpo_beta": idpo_trainer.beta,
            "idpo_margin": idpo_trainer.margin
        },
        "best_idpo_score": best_idpo_score,
        "final_epoch": epoch + 1,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(final_model_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Save the model
    model.save_pretrained(
        final_model_dir,
        save_function=torch.save,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(final_model_dir)
    
    # Save LoRA adapter separately
    adapter_dir = os.path.join(final_model_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(
        adapter_dir,
        save_function=torch.save,
        max_shard_size="5GB"
    )
    
    print(f"Training completed. Model and LoRA adapter saved to {final_model_dir}")
    print("Done!")
