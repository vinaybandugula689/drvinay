import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


@st.cache_resource()
def load_model():
    
    model = torch.load('best_model.pth', map_location=torch.device('cpu'))    
    model.eval()  
    return model
 
# Define the label mapping
label2id = {
    0: 'Healthy',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}

# Preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  
    return image

# Make predictions
def predict(image, model):
    with torch.no_grad():
        output = model(image)
        logits = output.logits if hasattr(output, 'logits') else output
        _, prediction = torch.max(logits, 1)
        return prediction.item(), logits

# Streamlit app
st.title('Diabetic Retinopathy Classification')

# File uploader
uploaded_file = st.file_uploader("Upload an image of an eye", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Load the model
    model = load_model()

    # Predict the class of the uploaded image
    if st.button("Classify"):
        label_id, logits = predict(processed_image, model)
        label = label2id[label_id]

        st.write(f"Prediction: **{label}**")

        # Display confidence scores for each class
        confidence_scores = torch.softmax(logits, dim=1).numpy()
        st.write("Confidence Scores for each class:")
        for idx, score in enumerate(confidence_scores[0]):
            st.write(f"{label2id[idx]}: {score * 100:.2f}%")

