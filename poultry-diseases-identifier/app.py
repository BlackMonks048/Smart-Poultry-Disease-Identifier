import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from typing import Tuple, Dict

# Constants
MODEL_PATH = "./model/mobilenetV2/mobilenetv2.h5"
CLASSES = {0: 'Coccidiosis', 1: 'Healthy', 2: 'New Castle Disease', 3: 'Salmonella'}
IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 70.0

# Disease Information Dictionary
DISEASE_INFO = {
    'Coccidiosis': {
        'Description': "A parasitic disease of the intestinal tract of animals caused by coccidian protozoa.",
        'Symptoms': "Bloody droppings, weight loss, ruffled feathers, low energy.",
        'Treatment': "Anticoccidial drugs (e.g., Amprolium, Toltrazuril) prescribed by a vet.",
        'Prevention': "Keep litter dry, good sanitation, avoid overcrowding."
    },
    'New Castle Disease': {
        'Description': "A contagious bird disease affecting many domestic and wild avian species.",
        'Symptoms': "Respiratory distress, nervous signs (twisted neck), drop in egg production.",
        'Treatment': "No specific treatment exists. Supportive care to prevent secondary infections.",
        'Prevention': "Vaccination is the most effective method."
    },
    'Salmonella': {
        'Description': "A bacterial infection that can affect both poultry and humans.",
        'Symptoms': "Diarrhea, loss of appetite, dejection, ruffled feathers.",
        'Treatment': "Antibiotics (after sensitivity test) and supportive therapy.",
        'Prevention': "Biosecurity measures, rodent control, clean water."
    },
    'Healthy': {
        'Description': "The bird appears to be in good health.",
        'Symptoms': "N/A",
        'Treatment': "Continue providing good care and nutrition.",
        'Prevention': "Maintain hygiene and vaccination schedule."
    }
}

# Page Configuration
st.set_page_config(
    page_title="Poultry Disease Identifier",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Loads the pre-trained model with custom objects."""
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(**kwargs)
            
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    ) 
    return model

def process_image(image: Image.Image) -> np.ndarray:
    """Preprocesses the image for the model."""
    image = ImageOps.fit(image, IMAGE_SIZE, Image.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

def predict(image: Image.Image, model) -> Tuple[str, float, np.ndarray]:
    """Runs prediction on the image."""
    img_array = process_image(image)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    class_index = np.argmax(score)
    pred_class = CLASSES[class_index]
    confidence = 100 * np.max(score)
    
    return pred_class, confidence, score.numpy()

def get_report_text(pred_class: str, confidence: float, disease_info: Dict) -> str:
    """Generates a text report for the diagnosis."""
    report = f"Poultry Disease Analysis Report\n"
    report += f"===============================\n\n"
    report += f"Diagnosis: {pred_class}\n"
    report += f"Confidence Score: {confidence:.2f}%\n\n"
    
    if pred_class in disease_info:
        info = disease_info[pred_class]
        report += f"Description: {info['Description']}\n"
        report += f"Symptoms: {info['Symptoms']}\n"
        report += f"Treatment: {info['Treatment']}\n"
        report += f"Prevention: {info['Prevention']}\n"
        
    report += "\nIMPORTANT: This is an AI-generated result. Please consult a veterinarian for a definitive diagnosis and treatment plan."
    return report

# Sidebar
with st.sidebar:
    st.title("Project Info")
    st.info(
        """
        **Poultry Disease Identifier**
        
        This app uses a Machine Learning model (MobileNetV2) to detect poultry diseases and identify healthy birds from images of chicken feces.
        
        **Classes:**
        - Coccidiosis
        - Healthy
        - New Castle Disease
        - Salmonella
        """
    )
    st.markdown("---")
    st.markdown("Developed by: **Karthick S**")

# Main Content
st.title("🐔 Poultry Disease Identifier")
st.header("MobileNetV2: Detect disease and healthy poultry feces")

# Load Model
with st.spinner('Loading Model...'):
    model = load_model()

# File Uploader
file = st.file_uploader("Upload an image of chicken feces", type=["jpg", "png", "jpeg"])

if file is None:
    st.warning("Please upload an image file to proceed.")
else:
    try:
        image = Image.open(file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True, caption="Analysis Source")
            
        with col2:
            st.subheader("Analysis Results")
            with st.spinner("Analyzing..."):
                pred_class, confidence, all_scores = predict(image, model)
            
            # Confidence Threshold Logic
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ Low Confidence Diagnosis ({confidence:.2f}%). The model is unsure. Please verify with a professional.")
                st.error(f"**Best Guess:** {pred_class}")
            else:
                # Display Result
                if pred_class == 'Healthy':
                    st.success(f"**Diagnosis:** {pred_class}")
                else:
                    st.error(f"**Diagnosis:** {pred_class}")
                st.metric("Confidence Score", f"{confidence:.2f}%")
            
            # Detailed Disease Info (always show for best guess)
            if pred_class in DISEASE_INFO:
                info = DISEASE_INFO[pred_class]
                with st.expander("ℹ️ Disease Information", expanded=True):
                    st.markdown(f"**Description:** {info['Description']}")
                    if info['Symptoms'] != "N/A":
                        st.markdown(f"**Symptoms:** {info['Symptoms']}")
                    
                    # Use "Recommendation" instead of "Treatment" for Healthy class
                    treatment_label = "Recommendation" if pred_class == 'Healthy' else "Treatment"
                    
                    st.markdown(f"**{treatment_label}:** {info['Treatment']}")
                    st.markdown(f"**Prevention:** {info['Prevention']}")

            st.markdown("---")
            
            # Download Report
            report_text = get_report_text(pred_class, confidence, DISEASE_INFO)
            st.download_button(
                label="📄 Download Analysis Report",
                data=report_text,
                file_name="poultry_disease_report.txt",
                mime="text/plain"
            )
            
            with st.expander("📊 View Probability Distribution"):
                for idx, class_name in CLASSES.items():
                    prob = all_scores[idx]
                    st.write(f"**{class_name}**")
                    st.progress(float(prob))
                
    except Exception as e:
        st.error(f"Error processing image: {e}")