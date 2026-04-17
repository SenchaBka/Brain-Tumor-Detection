import streamlit as st
import requests
from PIL import Image
import io

# 1. Page Configuration
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

st.title("🧠 NeuroScan AI")
st.subheader("MRI Brain Tumor Detection Portal")
st.write("Upload an axial MRI scan to run an AI-powered diagnostic analysis.")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    # 3. Predict Button
    if st.button('Run Diagnostic Analysis'):
        with st.spinner('Analyzing scan...'):
            try:
                # Convert the uploaded file to bytes for the API
                files = {"file": uploaded_file.getvalue()}
                
                # Send request to your Flask API
                response = requests.post("http://127.0.0.1:5000/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 4. Display Results
                    st.divider()
                    label = result['label']
                    confidence = result['confidence'] * 100
                    
                    if label == "Tumor":
                        st.error(f"**Result: {label} Detected**")
                    else:
                        st.success(f"**Result: {label} (Healthy)**")
                        
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                    
                else:
                    st.error("Error: Could not get a response from the API.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

st.sidebar.info("This tool uses a transfer-learned EfficientNetB0 model to identify brain tumors from MRI scans.")