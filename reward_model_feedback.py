def fine_tune_with_human_feedback(reward_model, feedback_dataset):
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=5e-6)
    
    for batch in feedback_dataset:
        preferred_code = batch['preferred']
        rejected_code = batch['rejected']
        
        # Get rewards for both versions
        preferred_inputs = tokenizer(preferred_code, return_tensors="pt", truncation=True).to(reward_model.device)
        rejected_inputs = tokenizer(rejected_code, return_tensors="pt", truncation=True).to(reward_model.device)
        
        preferred_reward, _ = reward_model(preferred_inputs.input_ids, preferred_inputs.attention_mask)
        rejected_reward, _ = reward_model(rejected_inputs.input_ids, rejected_inputs.attention_mask)
        
        # Compute preference loss
        loss = -torch.log(torch.sigmoid(preferred_reward - rejected_reward))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()