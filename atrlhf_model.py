import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from huggingface_hub import login
from datasets import load_dataset

# === Model Setup ===
login(token="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")

# Enhanced reward model for code generation
class CodeRewardModelWithConfidence(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        
        # Multiple heads for different aspects of code quality
        self.syntax_head = nn.Linear(hidden_size, 1)
        self.efficiency_head = nn.Linear(hidden_size, 1)
        self.readability_head = nn.Linear(hidden_size, 1)
        self.functionality_head = nn.Linear(hidden_size, 1)
        self.confidence_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, -1, :]
        
        # Calculate different aspects of code quality
        syntax_score = torch.tanh(self.syntax_head(hidden_states))
        efficiency_score = torch.tanh(self.efficiency_head(hidden_states))
        readability_score = torch.tanh(self.readability_head(hidden_states))
        functionality_score = torch.tanh(self.functionality_head(hidden_states))
        confidence = torch.sigmoid(self.confidence_head(hidden_states))
        
        # Combine scores with different weights
        reward = (
            0.3 * syntax_score +
            0.3 * functionality_score +
            0.2 * efficiency_score +
            0.2 * readability_score
        ).squeeze(-1)
        
        return reward, confidence.squeeze(-1)

# Replace the reward model initialization
reward_model = CodeRewardModelWithConfidence(base_model)

# Using DeepSeek instruct model with compatible settings
policy_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_flash_attention_2=False
)
policy_model.gradient_checkpointing_enable()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True
)

# Using OpenHermes for reward model with adjusted settings
base_model = AutoModelForCausalLM.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_flash_attention_2=False
)
base_model.gradient_checkpointing_enable()
reward_model = RewardModelWithConfidence(base_model)

# Set up tokenizer and model configs
tokenizer.pad_token = tokenizer.eos_token
policy_model.config.pad_token_id = tokenizer.eos_token_id
policy_model.train()

# Adjust PPO config with correct parameter names
ppo_config = PPOConfig( 
    learning_rate=1e-5, 
    batch_size=4, 
    mini_batch_size=1, 
    gradient_accumulation_steps=2, 
    num_ppo_epochs=4, 
    max_grad_norm=0.5, 
    vf_coef=0.1, 
    seed=42, )


# Initialize PPO trainer
trainer = PPOTrainer(
    ppo_config,
    policy_model,
    ref_model=None,
    tokenizer=tokenizer,
    device_map="auto",
    compute_dtype=torch.float16,
    reward_model=reward_model,
)

# Update PPO Hyperparameters to match config
clip_epsilon = 0.2  # This will be used in atrlhf_ppo_step
gamma = 0.99       # This will be used in compute_advantages
lam = 0.95        # This will be used in compute_advantages

optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

# === Training Loop (Single PPO Step with ATRLHF) ===
def compute_advantages(rewards, values, confidence, gamma=0.99, lam=0.95):
    deltas = confidence * (rewards - values)
    advantages = torch.zeros_like(rewards)
    advantage = 0.0
    for t in reversed(range(len(rewards))):
        advantage = deltas[t] + gamma * lam * advantage
        advantages[t] = advantage
    return advantages

def atrlhf_ppo_step(prompts, old_logprobs, responses, returns, advantages):
    for _ in range(ppo_epochs):
        inputs = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).to(policy_model.device)
        outputs = policy_model(**inputs, labels=inputs["input_ids"])
        logprobs = -F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                    inputs["input_ids"].view(-1), reduction='none')
        logprobs = logprobs.view(inputs["input_ids"].shape)

        ratio = torch.exp(logprobs - old_logprobs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        loss = -torch.min(surrogate1, surrogate2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === Example PPO Batch ===
def atrlhf_training_step(prompts, policy_model, reward_model, N=4):
    all_advantages, all_returns, all_responses, all_old_logprobs = [], [], [], []

    for prompt in prompts:
        responses = generate_n_responses(policy_model, prompt, N)
        tokenized = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).to(policy_model.device)

        with torch.no_grad():
            rewards, confidences = reward_model(tokenized["input_ids"], tokenized["attention_mask"])
            values = torch.zeros_like(rewards)  # replace with value_model if you have one

        # Filter by confidence threshold or sample proportionally
        confidence_threshold = 0.5
        keep_mask = confidences > confidence_threshold
        filtered_responses = [responses[i] for i in range(len(responses)) if keep_mask[i]]

        rewards, confidences, values = rewards[keep_mask], confidences[keep_mask], values[keep_mask]

        if len(rewards) == 0:
            continue  # skip this prompt if no confident samples

        advantages = compute_advantages(rewards, values, confidences)
        returns = rewards + values  # can be improved using GAE

        old_logprobs = torch.zeros_like(rewards)  # placeholder, get from policy if storing logprobs

        all_advantages.append(advantages)
        all_returns.append(returns)
        all_responses.extend(filtered_responses)
        all_old_logprobs.append(old_logprobs)

    # Flatten and train
    if all_responses:
        atrlhf_ppo_step(prompts, torch.cat(all_old_logprobs), all_responses, 
                        torch.cat(all_returns), torch.cat(all_advantages))


def generate_n_responses(policy_model, prompt, N=4, max_length=100):
    responses = []
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(policy_model.device)
    
    for _ in range(N):
        with torch.no_grad():
            outputs = policy_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    
    return responses


# Load primary training dataset
code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k")

# Load evaluation dataset
human_eval = load_dataset("openai/human-eval")
mbpp = load_dataset("mbpp")