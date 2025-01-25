import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# App title
st.title("Fine-Tuned phi-1_5 Text Generator")

# User input text box
prompt_text = st.text_area("Enter your text prompt:", height=150)

# Generate button
generate_button = st.button("Generate Text") # Add a button

# Model and tokenizer names - Use your fine-tuned model's identifier from Hugging Face Hub
model_name = "Sol1986/phi-1_5-T-dataset-finetuned"

# Load tokenizer and model
@st.cache_resource  # Caching to load model and tokenizer only once
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Determine device: GPU if available, CPU otherwise
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device) # Move model to the determined device

    return tokenizer, model, device # Return device

tokenizer, model = load_model_and_tokenizer(model_name)

# Model inference function
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
 
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Generate and display text only when the button is clicked AND there is a prompt
if generate_button and prompt_text: # Check if button is clicked AND prompt is not empty
    st.write("Generating text...")
    generated_output = generate_text(prompt_text)
    st.success("Generated Text:")
    st.write(generated_output)
elif generate_button and not prompt_text: # If button is clicked but no prompt
    st.warning("Please enter a text prompt before generating.")
