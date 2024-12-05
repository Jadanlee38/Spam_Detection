import requests
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gdown

# Download and cache the model from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1PajLfeMd3_vhjEBxShA0CMj2oTez6wnS"
MODEL_PATH = "fine_tuned_model.pt"

@st.cache_resource
def download_and_load_model():
    # Check if the model is already downloaded
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

# Load the model once and cache it
tokenizer, model = download_and_load_model()

# Define the prediction function
def predict(subject, body):
    text = subject + " " + body
    encodings = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit UI
st.title("Email Spam Classifier")
st.write("Paste the email subject and body below to classify it as Spam or Not Spam.")

# Input fields
subject = st.text_input("Email Subject")
body = st.text_area("Email Body")

if st.button("Classify"):
    if subject.strip() == "" and body.strip() == "":
        st.error("Please provide both subject and body for classification.")
    else:
        result = predict(subject, body)
        st.success(f"The email is classified as: **{result}**")
