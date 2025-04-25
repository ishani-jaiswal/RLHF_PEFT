import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Tuple
import wandb
from tqdm.auto import tqdm
from training_utils import ATRLHFTrainingManager, calculate_code_metrics
from main import CodeRewardModelWithConfidence

class CodeDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Clean and validate the dataset
        self.valid_indices = self._get_valid_indices()
        print(f"Found {len(self.valid_indices)} valid examples out of {len(dataset)}")

    def _get_valid_indices(self) -> List[int]:
        valid_indices = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            # Check various possible code field names
            code = None
            for field in ['code', 'source', 'content', 'function', 'solution']:
                if field in item and isinstance(item[field], str) and len(item[field].strip()) > 0:
                    code = item[field]
                    break
            if code is not None:
                valid_indices.append(i)
        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Get the actual dataset index
        dataset_idx = self.valid_indices[idx]
        item = self.dataset[dataset_idx]
        
        # Find the code field
        code = None
        for field in ['code', 'source', 'content', 'function', 'solution']:
            if field in item and isinstance(item[field], str) and len(item[field].strip()) > 0:
                code = item[field].strip()
                break
        
        if code is None:
            raise ValueError(f"No valid code found in item at index {dataset_idx}")
        
        # Calculate quality metrics
        try:
            metrics = calculate_code_metrics(code)
            quality_score = sum(metrics.values()) / len(metrics)
        except Exception as e:
            print(f"Warning: Error calculating metrics for index {dataset_idx}: {e}")
            quality_score = 0.5  # Default score for failed metrics
        
        # Tokenize code with proper special tokens
        try:
            # Add special tokens
            code_with_special = f"{self.tokenizer.bos_token}{code}{self.tokenizer.eos_token}"
            
            # Tokenize with proper padding
            inputs = self.tokenizer(
                text=code_with_special,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True,
                add_special_tokens=False  # We already added them manually
            )
            
            # Ensure we have valid tensors
            input_ids = inputs['input_ids'].squeeze()
            attention_mask = inputs['attention_mask'].squeeze()
            
            # Verify tensor shapes
            if input_ids.dim() == 0 or attention_mask.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'quality_score': torch.tensor(quality_score, dtype=torch.float)
            }
        except Exception as e:
            print(f"Warning: Error tokenizing code at index {dataset_idx}: {e}")
            # Return a dummy tensor with the right shape in case of tokenization error
            dummy_input = torch.zeros(self.max_length, dtype=torch.long)
            dummy_mask = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_input,
                'attention_mask': dummy_mask,
                'quality_score': torch.tensor(0.0, dtype=torch.float)
            }

def prepare_datasets(tokenizer) -> Tuple[Dataset, Dataset]:
    # Load high-quality code datasets
    print("Loading datasets...")
    datasets = {
        'code_alpaca': load_dataset("sahil2801/CodeAlpaca-20k"),
        'codeparrot': load_dataset("codeparrot/codeparrot-clean-train", split='train[:20000]'),  # Using more samples
        'code_search_net': load_dataset("code_search_net", "python", split='train[:10000]'),  # Python subset
        'gsm8k': load_dataset("gsm8k", "main", split='train')  # For mathematical reasoning
    }
    
    print("Processing datasets...")
    # Combine datasets for training
    train_datasets = []
    
    # Process Code Alpaca
    if 'code_alpaca' in datasets:
        train_datasets.append(datasets['code_alpaca']['train'])
    
    # Process CodeParrot
    if 'codeparrot' in datasets:
        train_datasets.append(datasets['codeparrot'])
    
    # Process Code Search Net
    if 'code_search_net' in datasets:
        train_datasets.append(datasets['code_search_net'])
    
    # Process GSM8K (use problem statements as prompts)
    if 'gsm8k' in datasets:
        gsm8k_data = datasets['gsm8k'].map(lambda x: {'code': x['question'] + '\n' + x['answer']})
        train_datasets.append(gsm8k_data)
    
    # Combine all datasets
    train_data = concatenate_datasets(train_datasets)
    
    # Create validation split
    train_val = train_data.train_test_split(test_size=0.1)
    train_data = train_val['train']
    eval_data = train_val['test']
    
    # Convert to custom dataset format
    train_dataset = CodeDataset(train_data, tokenizer)
    eval_dataset = CodeDataset(eval_data, tokenizer)
    
    return train_dataset, eval_dataset

def main_training_loop(
    model_name: str = "facebook/opt-125m",  # Using a much smaller model for testing
    batch_size: int = 2,  # Reduced batch size
    num_epochs: int = 5,  # Reduced epochs
    learning_rate: float = 1e-5,
    max_grad_norm: float = 0.5,
    project_name: str = "code-llm-rlhf-atrlhf"
):
    # Initialize wandb
    wandb.init(
        project=project_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm
        }
    )
    
    print("\n=== Initializing Models ===")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    print("Loading policy model...")
    print("This may take a few minutes...")
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model with memory-efficient settings
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="offload_folder",  # Offload weights to disk if needed
        max_memory={0: "4GiB"}  # Limit GPU memory usage
    )
    
    print("✓ Model loaded successfully")
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nConfiguring tokenizer...")
    # Ensure tokenizer has necessary special tokens
    special_tokens = {
        'pad_token': '[PAD]',
        'eos_token': '</s>',
        'bos_token': '<s>',
        'unk_token': '[UNK]'
    }
    
    # Add special tokens if they don't exist
    num_added = 0
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            setattr(tokenizer, token_type, token)
            num_added += 1
    
    # Resize token embeddings if needed
    if num_added > 0:
        print(f"Added {num_added} special tokens, resizing embeddings...")
        policy_model.resize_token_embeddings(len(tokenizer))
    print("✓ Tokenizer configured")
    print("Enabling gradient checkpointing...")
    policy_model.gradient_checkpointing_enable()
    print("✓ Policy model ready")
    
    print("\nInitializing reward model...")
    reward_model = CodeRewardModelWithConfidence(policy_model, tokenizer)
    print("✓ Reward model ready")
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(tokenizer)
    
    # Convert datasets to lists of prompts
    def extract_code(item):
        # Check various possible code field names
        for field in ['code', 'source', 'content', 'function', 'solution']:
            if field in item and isinstance(item[field], str) and len(item[field].strip()) > 0:
                return item[field].strip()
        return None
    
    train_prompts = []
    eval_prompts = []
    
    # Process training dataset
    print("\n=== Processing Datasets ===")
    train_pbar = tqdm(range(len(train_dataset)), desc="Processing training data")
    for i in train_pbar:
        code = extract_code(train_dataset[i])
        if code is not None:
            train_prompts.append(code)
        train_pbar.set_postfix({"valid": len(train_prompts)})
    
    # Process evaluation dataset
    eval_pbar = tqdm(range(len(eval_dataset)), desc="Processing validation data")
    for i in eval_pbar:
        code = extract_code(eval_dataset[i])
        if code is not None:
            eval_prompts.append(code)
        eval_pbar.set_postfix({"valid": len(eval_prompts)})
    
    print(f"Extracted {len(train_prompts)} training prompts and {len(eval_prompts)} evaluation prompts")
    
    # Initialize training manager
    training_manager = ATRLHFTrainingManager(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        ppo_epochs=4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        vf_coef=0.1
    )
    
    # Train the model
    best_model_state = training_manager.train(train_prompts, eval_prompts)
    
    # Save the best model
    torch.save({
        'policy_model': best_model_state,
        'reward_model': reward_model.state_dict()
    }, "best_atrlhf_model.pt")
    wandb.save("best_atrlhf_model.pt")
    
    wandb.finish()

if __name__ == "__main__":
    main_training_loop()