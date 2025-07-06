import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

# --- UI CONFIGURATION AND CUSTOM CSS ---
st.set_page_config(page_title="Chest Xray Classification", layout="wide")

st.markdown("""
    <style>
    body, .stApp, .main {
        background-color: #bb2000;
    }
    .top-left-text {
        color: white !important;
        font-size: 16px;
        font-weight: bold;
        position: fixed;
        top: 20px;
        left: 30px;
        z-index: 100;
        line-height: 1.3;
        user-select: none;
    }
    .title-text {
        color: white;
        font-size: 56px;
        font-weight: bold;
        text-align: center;
        letter-spacing: 2px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .caption-text {
        color: white;
        font-size: 18px;
        text-align: center;
        margin-bottom: 16px;
        margin-top: 20px;
    }
    .result-label {
        color: white;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 0px;
    }
    .result-pred {
        color: white;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 40px;
        letter-spacing: 2px;
    }
    .stButton > button {
        color: #bb2000;
        background-color: white;
        border-radius: 30px;
        font-size: 20px;
        font-weight: bold;
        width: 220px;
        height: 48px;
        margin: 0 auto;
        display: block;
    }
    .stFileUploader {
        text-align: center;
    }
    .stSuccess {
        color: white !important;
        font-weight: bold !important;
    }
    .stSuccess > div {
        color: white !important;
     font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- TOP LEFT SIMPLE TEXT ---
st.markdown("""
<div class="top-left-text">
    Developed by: Saakar Agrawal<br>
    College of Biomedical Engineering and Applied Sciences
</div>
""", unsafe_allow_html=True)

# --- CENTERED TITLE ---
st.markdown('<div class="title-text">CHEST XRAY CLASSIFICATION</div>', unsafe_allow_html=True)

# --- CENTERED EXAMPLE IMAGE ---
try:
    example_img = Image.open("example_xrays.jpg")
    st.image(example_img, use_column_width=False, width=400)
except Exception:
    st.warning("Example image 'example_xrays.jpg' not found in app directory.")

# --- CAPTION ---
st.markdown('<div class="caption-text">ATTACH YOUR XRAY IMAGE AND ANALYZE IT!</div>', unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

# --- MODEL ARCHITECTURE DEFINITION (EXACT MATCH TO TRAINING) ---
def create_model():
    """
    Create the exact same model architecture as used during training
    """
    # Load a pretrained EfficientNetB0 model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify the first convolutional layer to accept 1 input channel (grayscale)
    # Get the first conv layer
    first_conv_layer = model.features[0][0]
    
    # Create a new conv layer with 1 input channel - EXACT MATCH TO TRAINING
    new_first_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=False if first_conv_layer.bias is None else True  # EXACT MATCH
    )
    
    # Initialize the new conv weights by averaging the original RGB weights
    new_first_conv.weight.data = first_conv_layer.weight.data.mean(dim=1, keepdim=True)
    
    # Replace the first conv layer
    model.features[0][0] = new_first_conv
    
    # Add dropout before the classifier for regularization - EXACT MATCH
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),  # Add dropout with 0.5 probability
        nn.Linear(model.classifier[1].in_features, 4)  # 4 output classes
    )
    
    return model

# --- MODEL LOADING FUNCTION ---
@st.cache_resource
def load_model(path):
    """
    Load the trained model with the exact architecture used during training
    """
    try:
        # Create model with exact same architecture
        model = create_model()
        
        # Load the saved state dictionary
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model once
try:
    model = load_model("chest_model.pth")
    if model is not None:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load model. Please check the model file.")
except FileNotFoundError:
    st.error("Model file 'chest_model.pth' not found. Please place it in the app directory.")
    model = None
except Exception as e:
    st.error(f"Unexpected error loading model: {str(e)}")
    model = None

# Class names matching the training
class_names = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis']

# --- IMAGE PREPROCESSING FUNCTION (EXACT MATCH TO TRAINING) ---
def preprocess_image(image: Image.Image):
    """
    Preprocess the image using the exact same transforms as used during training
    """
    # Use the same normalization as training: mean=[0.5], std=[0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # IMAGE_SIZE = 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # EXACT MATCH TO TRAINING
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- DISPLAY UPLOADED IMAGE AND PREDICT ---
if uploaded_file is not None:
    try:
        # Convert to grayscale (L mode)
        user_img = Image.open(uploaded_file).convert('L')
        st.image(user_img, caption="Your uploaded X-ray", use_column_width=False, width=300)
        
        if st.button("Show result"):
            if model is not None:
                with st.spinner("Analyzing X-ray..."):
                    try:
                        # Preprocess the image
                        input_tensor = preprocess_image(user_img)
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            _, predicted = torch.max(outputs, 1)
                            prediction = class_names[predicted.item()]
                        
                        # Display result
                        st.markdown('<div class="result-label">RESULT:</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-pred">{prediction.upper()} !!</div>', unsafe_allow_html=True)
                        
                        # Optional: Show prediction confidence
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence = probabilities[0][predicted.item()].item()
                        st.markdown(f'<div style="color: white; text-align: center; font-size: 16px;">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            else:
                st.error("Model is not loaded. Cannot perform prediction.")
                
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")
else:
    st.markdown('<div class="result-label">RESULT:</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-pred">#class should be displayed here#</div>', unsafe_allow_html=True)