import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./models/simple_chatbot"  # or your actual model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def clean_response(text):
    return text.replace('-->', '').replace('>', '').replace('<', '').replace('-', '').replace('  ', ' ').strip()

def is_in_domain(user_input):
    medical_keywords = [
        "symptom", "treatment", "doctor", "medicine", "pain", "fever", "diabetes", "heart", "blood", "health",
        "disease", "infection", "cough", "headache", "nurse", "hospital", "clinic", "prescription", "dose","virus",
        "bacteria", "allergy", "asthma", "cancer", "surgery", "therapy", "mental health","flu", "cold", "injury", 
        "emergency", "vaccination", "immunization", "nutrition", "exercise","eye", "ear", "nose", "throat", "skin", 
        "bone", "joint", "muscle", "pregnancy", "childbirth","infant", "child", "adolescent", "adult", "elderly", "geriatrics",
        "pediatrics","cardiology", "neurology", "orthopedics", "dermatology", "psychiatry","radiology", "pathology", "anatomy", 
        "physiology","pharmacology"
    ]
    return any(word in user_input.lower() for word in medical_keywords)

def chat_with_bot(user_input):
    if not is_in_domain(user_input):
        return "Sorry, I can only answer healthcare-related questions."
    
    prompt = f"Patient: {user_input}\nDoctor:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
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
    doctor_response = clean_response(doctor_response)
    return doctor_response

iface = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(lines=2, label="Your question"),
    outputs=gr.Textbox(label="Our Response"),
    title="🏥 Healthcare Chatbot",
    description="Ask a medical question and get a response from the chatbot."
)

if __name__ == "__main__":
    # create the public link
    iface.launch(share=True)