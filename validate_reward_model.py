import subprocess
import tempfile
import ast
from radon.complexity import cc_visit
from pylint import epylint as lint

def validate_reward_model(reward_model, test_cases):
    metrics = {
        'functional_correctness': 0,
        'syntax_validity': 0,
        'complexity_score': 0,
        'lint_score': 0
    }
    total_cases = 0
    
    for test_case in test_cases:
        code = test_case['code']
        expected_output = test_case['expected_output']
        
        # Get model's prediction
        inputs = tokenizer(code, return_tensors="pt").to(reward_model.device)
        reward, confidence = reward_model(inputs.input_ids, inputs.attention_mask)
        
        # Validate multiple aspects
        try:
            # Syntax check
            ast.parse(code)
            metrics['syntax_validity'] += 1
            
            # Complexity check
            complexity = cc_visit(code)
            metrics['complexity_score'] += 1.0 / (1.0 + max([c.complexity for c in complexity]))
            
            # Lint check
            (pylint_stdout, _) = lint.py_run(code, return_std=True)
            lint_score = float(pylint_stdout.getvalue().split('Your code has been rated at ')[1].split('/')[0]) / 10
            metrics['lint_score'] += lint_score
            
            # Functional correctness
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
                f.write(code)
                f.flush()
                result = subprocess.run(['python', f.name], capture_output=True, text=True, timeout=5)
                if result.stdout.strip() == expected_output:
                    metrics['functional_correctness'] += 1
                
            total_cases += 1
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            continue
    
    # Calculate average scores
    return {k: v/total_cases if total_cases > 0 else 0 for k, v in metrics.items()}