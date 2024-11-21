import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import cv2
from io import BytesIO

# Setting the page configuration
st.set_page_config(page_title="Height Segmentation Model", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation Pane ⤵️")
st.sidebar.markdown('<style>div.row {display: flex;}</style>', unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: white;'>Go to:</h3>", unsafe_allow_html=True)

# Sidebar options
page = st.sidebar.selectbox("Select Page", ["Home", "Predictions", "Images and Masks", "Insights", "Meet the team"])

# Load models for height and segmentation prediction
@st.cache_resource
def load_height_model():
    height_model = tf.keras.models.load_model('model.h5')
    return height_model

@st.cache_resource
def load_segmentation_model():
    segmentation_model = tf.keras.models.load_model('model.h5')
    return segmentation_model
    
# Custom CSS for consistent button styling and footer positioning
st.markdown(
    """
    <style>
    /* Add hover effect on images with border */
    .hover-effect:hover {
        transform: scale(1.05); transition: transform 0.3s ease;
    }
    
    /* Adding border around the sample images */
    .sample-image {
        border: 3px solid #3e8e41; padding: 10px; border-radius: 8px;
    }
    
    /* Footer positioning at the bottom of each page */
    .footer {
        position: fixed; bottom: 0; width: 70%; background-color: #0E1117; color: white; text-align: center; padding: 10px; font-size: 15px;
    }

    /* Cursor style for clickable images */
    .clickable:hover {
        cursor: pointer;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for the generated mask
if "generated_mask" not in st.session_state:
    st.session_state["generated_mask"] = None

# Preprocess the input image for the model
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    if len(image_array.shape) == 4:  # Grayscale image
        image_array = np.stack([image_array] * 3, axis=-1)  # Convert to RGB
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Postprocess the mask for visualization
def postprocess_mask(mask):
    """
    Converts the model's prediction to a visualizable mask.
    The model outputs a 4D tensor with shape (1, height, width, channels).
    """
    if len(mask.shape) == 4:  # (batch_size, height, width, channels)
        mask = mask[0, ..., 0]  # Remove batch and channel dimensions
    elif len(mask.shape) == 3:  # (height, width, channels)
        mask = mask[..., 0]  # Remove channel dimension
    elif len(mask.shape) == 2:  # Already a single-channel 2D array
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold and scale to [0, 255]
    return Image.fromarray(mask)

# Load both models
height_model = load_height_model()
segmentation_model = load_segmentation_model()
st.title("Welcome to Height Segmentation Model 📊🛰️")

# Home Page
if page == "Home":
    st.markdown("""This application allows users to upload satellite images and receive height and segmentation predictions based on a trained model.""")
    st.image("sample.png")

    # Add label above download button
    st.markdown("""Welcome, you can access and download our project documentation here.""")
    st.download_button(
        label="Download as PDF",
        data="Height_Segmentation_Presentation.pdf",
        mime="application/pdf"
    )

# Predictions Page
if page == "Predictions":
    st.markdown("### Make Predictions🚀")
    
    # Model selection
    model_type = st.selectbox("Select Model Type", ["Height Model", "Segmentation Model"])

    # Sample images for testing
    st.subheader("Select Samples from Test Set")
    
    sample_images = {
        "Sample Image 1": "test_image1.jpg", 
        "Sample Image 2": "test_image2.jpg",
        "Sample Image 3": "test_image3.jpg"
    }
    
    # Create columns to display sample images
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    selected_image = None
    
    for idx, (label, image_path) in enumerate(sample_images.items()):
        img = Image.open(image_path)
        with cols[idx]:
            # Add hover effect with class 'hover-effect' and 'clickable' for clickable images
            st.image(img, use_column_width=False, width=250)
    
    # Create a single centered column for the buttons below the images
    center_col = st.columns(1)[0]
    
    for idx, (label, image_path) in enumerate(sample_images.items()):
        img = Image.open(image_path)
        with cols[idx]:
            if st.button(label):
                selected_image = Image.open(image_path)
    
    # Upload image section
    uploaded_file = st.file_uploader("Or upload Your Satellite Image", type=["jpg", "png"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)
    
    if selected_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(selected_image, caption="Original Image", use_column_width=True)

        # Generate Mask Button
        if st.button("Generate Mask"):
            preprocessed_image = preprocess_image(selected_image)
            mask = segmentation_model.predict(preprocessed_image)
            mask_image = postprocess_mask(mask)
            st.session_state["generated_mask"] = mask_image  # Save mask to session state
        
        # Display the persistent mask
        if st.session_state["generated_mask"] is not None:
            with col2:
                st.image(st.session_state["generated_mask"], caption="Generated Mask", use_column_width=True)

    if selected_image is not None:
        # Model prediction section
        if st.button("Run Model"):
            # Preprocess the image for model input
            img_array = np.array(selected_image.resize((256, 256))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Run the selected model based on the model type
            if model_type == "Height Model":
                height_prediction = height_model.predict(img_array)
                st.subheader("Height Model Prediction")
                st.image(height_prediction[0], caption='Height Output', use_column_width=True)
                st.write("Performance metrics: []")

            elif model_type == "Segmentation Model":
                segmentation_prediction = segmentation_model.predict(img_array)
                st.subheader("Segmentation Model Prediction")
                # Fixed to use index [0] for the output
                st.image(segmentation_prediction[0], caption='Segmentation Output', use_column_width=True)
                st.write("Test Mean IoU: 0.74")

# Insights Page
elif page == "Insights":
    st.markdown("### Key Insights")
    st.write("Model performance metrics and comparative analysis are shown below.")

    # Create a DataFrame with labeled rows
    data = {
        "Metric": ["Accuracy", "IoU"],
        "Height Model": [0.9, 0.85],
        "Segmentation Model": [0.94, 0.74],
    }
    df = pd.DataFrame(data).set_index("Metric")

    # Render a bar chart using the DataFrame
    st.bar_chart(df)
    
# View Images and Masks Page
elif page == "Images and Masks":
    st.markdown("### View Images and Masks")

    # Define image and mask pairs
    image_mask_pairs = {
        "austin17-1": ("austin17-8.jpg", "austin17-8-mask.png"),
        "austin17-2": ("austin17-9.jpg", "austin17-9-mask.png"),
        "austin17-3": ("austin17-10.jpg", "austin17-10-mask.png"),
        "austin17-4": ("austin17-11.jpg", "austin17-11-mask.png")
    }

    # Dropdown to select image and mask pair
    selected_pair = st.selectbox("Select Image and Mask Pair", list(image_mask_pairs.keys()))

    # Display selected image and mask
    if selected_pair:
        image_path, mask_path = image_mask_pairs[selected_pair]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Display the selected image and mask side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f'{selected_pair} - Original Image')
        
        with col2:
            st.image(mask, caption=f'{selected_pair} - Mask')

# Meet the Team Page
elif page == "Meet the team":
    st.markdown("### Meet The Team 👩🏾‍💻👩🏾‍💻")

    # Team members' information
    team_members = [
        {
            "name": "EMMANUEL NKHUBALALE - Data Science Intern | Team Lead",
            "photo": "Nkadimeng.jpg",
            "email": "physimanuel@gmail.com",
            "linkedin": "https://linkedin.com/in/emmanuel"
        },
        {
            "name": "NOLWAZI MNDEBELE - Data Science Intern | Project Manager",
            "photo": "Nolwazi.jpg",
            "email": "mndebelenf@gmail.com",
            "linkedin": "https://linkedin.com/in/nolwazi"
        },
        {
            "name": "CARROLL TSHABANE - Data Science Intern",
            "photo": "Carroll.jpg",
            "email": "ctshabane@gmail.com",
            "linkedin": "https://linkedin.com/in/carroll"
        },
        {
            "name": "JAN MOTENE - Data Science Intern",
            "photo": "Jan.jpg",
            "email": "motenejo@gmail.com",
            "linkedin": "https://linkedin.com/in/jan"
        },
        {
            "name": "ZAMANCWABE MAKHATHINI - Data Science Intern/ Engineer",
            "photo": "Zama.jpg",
            "email": "zamancwabemakhathini@gmail.com",
            "linkedin": "https://linkedin.com/in/zamancwabe"
        },
        {
            "name": "MUWANWA TSHIKOVHI - Data Science Intern",
            "photo": "test_image1.jpg",
            "email": "tshikovhimuwanwa@gmail.com",
            "linkedin": "https://linkedin.com/in/muwanwa"
        },
        {
            "name": "SIBUKISO NHLENGETHWA - Data Science Intern",
            "photo": "Sibukiso.jpg",
            "email": "sibukisot@gmail.com",
            "linkedin": "https://linkedin.com/in/sibukiso"
        }
    ]

    # Display the team members
    col1, col2 = st.columns(2)
    for idx, member in enumerate(team_members):
        with (col1 if idx % 2 == 0 else col2 ):
            st.image(member['photo'], width=120)
            st.markdown(f"**Name**: {member['name']}")
            st.markdown(f"**📧 Email**: {member['email']}")
            st.markdown(f"**🟦LinkedIn**: {member['linkedin']}")

# Footer section
st.markdown(
    """
    <div class="footer">
        This application aims to assist in optimizing telecommunications network design by providing accurate height and segmentation outputs from satellite imagery.<br>
        Height Segmentation Model Team B &copy; 2024
    </div>
    """,
    unsafe_allow_html=True
)