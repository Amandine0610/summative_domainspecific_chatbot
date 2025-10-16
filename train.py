#!/usr/bin/env python3
"""
Ultra-Simple Training Script - No Complex Dependencies
This avoids Trainer, accelerate, and other complex packages
"""

import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob

def find_dataset():
    """Find dataset file automatically."""
    print("🔍 Looking for your dataset...")
    
    patterns = ["*medical*.csv", "*medical*.json", "*chatbot*.csv", "*chatbot*.json"]
    search_dirs = [".", "./data"]
    
    if os.name == 'nt':  # Windows
        search_dirs.extend(["C:/Users/hp/Downloads", "C:/Users/hp/Documents"])
    
    found_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for pattern in patterns:
                files = glob.glob(os.path.join(search_dir, pattern))
                for file in files:
                    if os.path.isfile(file) and os.path.getsize(file) > 1000:
                        found_files.append(file)
    
    unique_files = list(set(found_files))
    unique_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    return unique_files[0] if unique_files else None

def load_data(dataset_path, max_samples=50):
    """Load and prepare data."""
    print(f"📊 Loading data from: {os.path.basename(dataset_path)}")
    
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        print(f"✅ CSV columns: {list(df.columns)}")
        
        # Auto-detect columns
        question_col = None
        answer_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['question', 'query', 'input', 'description', 'patient']):
                question_col = col
            elif any(word in col_lower for word in ['answer', 'response', 'output', 'doctor']):
                answer_col = col
        
        if not question_col or not answer_col:
            cols = list(df.columns)
            question_col = cols[0]
            answer_col = cols[1] if len(cols) > 1 else cols[0]
        
        print(f"🎯 Using: {question_col} -> {answer_col}")
        
        # Convert to conversations
        conversations = []
        for _, row in df.iterrows():
            if pd.notna(row[question_col]) and pd.notna(row[answer_col]):
                q = str(row[question_col]).strip()
                a = str(row[answer_col]).strip()
                if q and a:
                    conversations.append(f"Patient: {q}\nDoctor: {a}")
                    
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        for item in data:
            if 'question' in item and 'answer' in item:
                conversations.append(f"Patient: {item['question']}\nDoctor: {item['answer']}")
    
    # Limit samples
    if len(conversations) > max_samples:
        conversations = conversations[:max_samples]
    
    print(f"✅ Prepared {len(conversations)} conversations")
    return conversations

def simple_train(conversations, model_name="distilgpt2", output_dir="./models/simple_chatbot"):
    """Simple training without Trainer."""
    print(f"🤖 Loading {model_name}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    print("✅ Model loaded")
    
    # Prepare data
    print("🔤 Tokenizing data...")
    
    # Simple tokenization
    inputs = []
    for conv in conversations:
        tokens = tokenizer(
            conv,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs.append(tokens)
    
    print(f"✅ Tokenized {len(inputs)} samples")
    
    # Simple training loop
    print("🎓 Training (simple approach)...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    total_loss = 0
    num_steps = 0
    
    for epoch in range(10):  # Just 1 epoch for quick test
        print(f"Epoch {epoch + 1}/{epoch + 1}")
        
        for i, batch in enumerate(inputs):
            # Forward pass
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_steps += 1
            
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / num_steps
                print(f"  Step {i + 1}/{len(inputs)}, Loss: {avg_loss:.4f}")
    
    print("✅ Training completed!")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"💾 Model saved to: {output_dir}")
    
    # Quick test
    print("🧪 Quick test...")
    model.eval()
    
    test_input = "Patient: What are the symptoms of diabetes?"
    inputs = tokenizer.encode(test_input, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
      outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=inputs.shape[1] + 100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.95,
        top_k=50
    )
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Test response: {response}")
    
    return output_dir

def main():
    """Main function."""
    print("🏥 ULTRA-SIMPLE HEALTHCARE CHATBOT TRAINER")
    print("=" * 50)
    print("(No complex dependencies - just basic PyTorch)")
    print()
    
    # Find dataset
    dataset_path = find_dataset()
    
    if not dataset_path:
        # Use sample data
        dataset_path = "./data/healthcare_qa_dataset.json"
        print("💡 No dataset found, using sample data")
    
    if not os.path.exists(dataset_path):
        print("❌ No dataset available!")
        print("Please place your ai-medical-chatbot.csv in this directory")
        return
    
    try:
        # Load data
        conversations = load_data(dataset_path, max_samples=50)
        
        if not conversations:
            print("❌ No valid conversations found!")
            return
        
        # Train model
        model_path = simple_train(conversations)
        
        print("\n🎉 SUCCESS!")
        print("=" * 30)
        print(f"✅ Model saved: {model_path}")
        print(f"📊 Trained on: {len(conversations)} conversations")
        
        print("\n🚀 NEXT STEPS:")
        print(f"1. Test with: python -c \"from transformers import pipeline; chatbot = pipeline('text-generation', model='{model_path}'); print(chatbot('Patient: What is diabetes?', max_length=100))\"")
        print("2. Or create a simple chat interface")
        
        # Create simple chat script
        chat_script = f'''#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "{model_path}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("🏥 Healthcare Chatbot Ready!")
print("Type 'quit' to exit")

while True:
    user_input = input("\\nPatient: ")
    if user_input.lower() == 'quit':
        break
    
    prompt = f"Patient: {{user_input}}\\nDoctor:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    doctor_response = response.split("Doctor:")[-1].strip()
    print(f"Doctor: {{doctor_response}}")
'''
        
        with open("chat_with_bot.py", "w", encoding="utf-8") as f:
            f.write(chat_script)
        
        print(f"3. Chat interface created: python chat_with_bot.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()