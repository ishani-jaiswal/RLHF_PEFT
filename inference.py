import torch
from transformers import AutoTokenizer
from main import CodeRewardModelWithConfidence
import json

def load_trained_model(checkpoint_path: str = "best_model.pt"):
    """Load the trained model and tokenizer"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
    
    # Initialize model with checkpoint
    model = CodeRewardModelWithConfidence.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def generate_code(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate code based on the prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get quality assessment
    reward, confidence = model(outputs, return_conf=True)
    
    return {
        "generated_code": generated_code,
        "quality_score": float(reward.mean().item()),
        "confidence": float(confidence.mean().item())
    }

def main():
    # Load model
    print("Loading model...")
    model, tokenizer = load_trained_model()
    
    while True:
        # Get prompt from user
        prompt = input("\nEnter your code generation prompt (or 'quit' to exit):\n")
        if prompt.lower() == 'quit':
            break
        
        # Generate code
        print("\nGenerating code...")
        result = generate_code(model, tokenizer, prompt)
        
        # Print results
        print("\nGenerated Code:")
        print("=" * 80)
        print(result["generated_code"])
        print("=" * 80)
        print(f"\nQuality Score: {result['quality_score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()
