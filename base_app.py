import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Setting the page configuration
st.set_page_config(page_title="Segmentation Model", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation Pane ⤵️")
st.sidebar.markdown('<style>div.row {display: flex;}</style>', unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: white;'>Go to:</h3>", unsafe_allow_html=True)

# Sidebar options
page = st.sidebar.selectbox("Select Page", ["Home", "Predictions", "Insights",'Feedback', "Meet the team"])

# Define custom loss function
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2 * intersection + smooth) / (union + smooth)

# Load segmentation model
@st.cache_resource
def load_segmentation_model():
    try:
        return load_model(
            'final_model_resunet_finetuned_lr_reduced_5.h5',
            custom_objects={'dice_loss': dice_loss, 'MeanIoU': MeanIoU},
            compile=False
        )
    except Exception as e:
        st.error(f"Error loading Segmentation Model: {e}")
        return None

# Define image preprocessing class
class CustomImageDataGenerator:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def preprocess_image(self, image):
        # Resize and normalize the input image
        image_resized = cv2.resize(image, self.target_size) / 255.0
        return image_resized

    def generate(self, image):
        # Preprocess the input image
        processed_image = self.preprocess_image(image)
        # Expand dimensions to fit model's input shape (add batch dimension)
        return np.expand_dims(processed_image, axis=0)

# Prediction function
def predict_segmentation(uploaded_file):
    data_generator = CustomImageDataGenerator(target_size=(256, 256))

    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Decode to BGR format

    if image is None:
        st.error("Image not found or unable to load.")
        return None, None

    processed_image = data_generator.generate(image)

    # Make prediction
    prediction = segmentation_model.predict(processed_image)

    # Binarize prediction
    prediction_bin = (prediction > 0.5).astype(np.float32).squeeze()

    return image, prediction_bin

# Load segmentation model
segmentation_model = load_segmentation_model()

# Home Page
if page == "Home":
    st.title("Welcome to the Segmentation Model 📊🛰️")
    st.markdown("""This application allows users to upload satellite images and receive segmentation predictions based on a trained model.""")
    st.image("New_image.png")

    # Add Instructions Section
    st.subheader("How to Use the Application:")
    st.markdown("""
    1. **Go to the 'Predictions' page** from the sidebar.
    2. **Upload your satellite image** (JPG, PNG, JPEG) on the 'Predictions' page.
    3. **Click on 'Submit'** to run the model and generate the segmentation mask for your image.
    4. **View Results**: The processed image and its segmentation mask will be displayed side by side.
    5. **Explore other features**: You can also view insights and learn about the team in the other pages.
    """)

    # Add download section for project documentation
    st.markdown("Welcome, you can access and download our project documentation here.")
    st.download_button(
        label="Download as PDF",
        data="Segmentation_Presentation.pdf",
        mime="application/pdf"
    )

# Predictions Page
if page == "Predictions":
    st.title("Generate Predictions 🚀")
    st.markdown("Upload a satellite image to generate its segmentation mask or explore the provided sample images.")

    # Display sample images
    sample_images = {
        "Sample Image 1": "test_image1.jpg",
        "Sample Image 2": "test_image2.jpg",
        "Sample Image 3": "test_image3.jpg"
    }

    st.subheader("Sample Images")
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    # Display sample images in columns
    for idx, (label, img_path) in enumerate(sample_images.items()):
        with cols[idx]:
            st.image(img_path, caption=label, use_column_width=True)
            if st.button(f"Select {label}"):
                selected_image = img_path

    # If a sample image is selected, load and display it with its segmentation mask
    if 'selected_image' in locals():
        st.markdown(f"**You selected:** {selected_image}")
        original_image = cv2.imread(selected_image)
        processed_image = CustomImageDataGenerator(target_size=(256, 256)).generate(original_image)

        # Simulate prediction for sample images (replace this with your actual model prediction if needed)
        predicted_mask = segmentation_model.predict(processed_image).squeeze()

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        with col2:
            st.image(predicted_mask, caption="Predicted Segmentation Mask", use_column_width=True, clamp=True, channels="GRAY")

    # Allow users to upload their own images
    st.markdown("### Or Upload Your Own Image")
    uploaded_file = st.file_uploader("Upload Your Satellite Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        original_image, predicted_mask = predict_segmentation(uploaded_file)
        if original_image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            with col2:
                st.image(predicted_mask, caption="Predicted Segmentation Mask", use_column_width=True, clamp=True, channels="GRAY")


# Insights Page
elif page == "Insights":
    st.markdown("### Key Insights")
    st.write("Model performance metrics and comparative analysis are shown below.")

    # Create a DataFrame with labeled rows
    data = {
        "Metric": ["Accuracy", "IoU", "Dice Loss"],
        "Segmentation Model": [0.94, 0.78, 0.20],
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

if page == "Feedback":
    st.title("We Value Your Feedback 💬")
    st.markdown("Please share your thoughts about the application. Your feedback helps us improve!")

    # Text area for user feedback
    user_feedback = st.text_area("Enter your feedback here:", placeholder="Type your valuable feedback...")

    # Submit button
    if st.button("Submit Feedback"):
        if user_feedback.strip():  # Check if the feedback is not empty
            st.success("Thank you for your feedback! 🙌")
            # Optional: Save feedback to a file
            with open("user_feedback.txt", "a") as f:
                f.write(f"{user_feedback}\n")
        else:
            st.warning("Feedback cannot be empty. Please enter your feedback.")

# Meet the Team Page
elif page == "Meet the team":
    st.markdown("### Meet The Team 👩🏾‍💻👩🏾‍💻")

    # Mission Statement
    st.markdown("""
    **Our Mission**: We are a dynamic and passionate group of data science interns driven by the vision to revolutionize telecommunications network design using advanced deep learning and satellite imagery. Our goal is to build cutting-edge solutions that not only optimize current systems but also empower industries to thrive in the future.
    
    We are not just limited to data science and machine learning. As a team, we have cultivated a strong foundation in **data engineering** and **software engineering** with a special focus on **backend development**. We are continuously expanding our knowledge and skills in these areas to build end-to-end solutions that integrate seamlessly with modern infrastructure. We embrace the challenge of learning new technologies, adapting to evolving tools, and leveraging our expertise to deliver robust and scalable systems. Our work is not just about technical innovation, but also about creating lasting impact and contributing to Africa's growth in technology and beyond.
    """)

    # Team members' information
    team_members = [
        {
            "name": "EMMANUEL NKHUBALALE - Data Science Intern | Team Lead",
            "photo": "Nkadimeng.jpg",
            "email": "physimanuel@gmail.com",
            "linkedin": "https://www.linkedin.com/in/nkhubalale-emmanuel-nkadimeng/"
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
            "linkedin": "https://www.linkedin.com/in/carroll-tshabane-bb816475/"
        },
        {
            "name": "JAN MOTENE - Data Science Intern",
            "photo": "Jan.jpg",
            "email": "motenejo@gmail.com",
            "linkedin": "https://www.linkedin.com/in/jan-motene-4864771b4/"
        },
        {
            "name": "ZAMANCWABE MAKHATHINI - Data Science Intern/ Engineer",
            "photo": "Zama.jpg",
            "email": "zamancwabemakhathini@gmail.com",
            "linkedin": "https://linkedin.com/in/zamancwabe"
        },
        {
            "name": "MUWANWA TSHIKOVHI - Data Science Intern",
            "photo": "Muwanwa.jpg",
            "email": "tshikovhimuwanwa@gmail.com",
            "linkedin": "https://www.linkedin.com/in/muwanwa-tshikovhi-64a5b6196/"
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
        with (col1 if idx % 2 == 0 else col2):
            st.image(member['photo'], width=120)
            st.markdown(f"**Name**: {member['name']}")
            st.markdown(f"**📧 Email**: {member['email']}")
            st.markdown(f"**🟦LinkedIn**: {member['linkedin']}")

# Footer section
st.markdown(
    """
    <div class="footer" style="font-size: 12px;">
        This application aims to assist in optimizing telecommunications network design by providing accurate segmentation outputs from satellite imagery.<br>
        Future updates will include height prediction capabilities to enhance design accuracy.<br>
        Height Segmentation Model Team B &copy; 2024
    </div>
    """,
    unsafe_allow_html=True
)


# Custom CSS for consistent styling and footer positioning
st.markdown(
    """
    <style>
   
    /* Footer positioning at the bottom of each page */
    .footer {
        position: fixed; bottom: 0; width: 70%; background-color: #0E1117; color: white; text-align: center; padding: 10px; font-size: 15px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)