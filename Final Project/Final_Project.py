import streamlit as st
from google.cloud import vision
import pytesseract
from PIL import Image
import pyttsx3
import torch
import tempfile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Set up the OpenAI model (replace with your API key if needed)
llm = OpenAI(model="text-davinci-003", temperature=0.7)

# Application Title
st.title("AI-Powered Assistance for Visually Impaired Individuals")
st.sidebar.title("Features")
st.sidebar.markdown("Select a feature to explore:")

# Sidebar Options
menu_options = ["Home", "Real-Time Scene Understanding", "Text-to-Speech Conversion", "Object Detection",
                "Personalized Assistance"]
selected_feature = st.sidebar.radio("Choose a Feature", menu_options)


# Function: Real-Time Scene Understanding
def describe_scene(image_path):
    """Generate scene descriptions using Google Vision API."""
    try:
        client = vision.ImageAnnotatorClient()
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        labels = [label.description for label in response.label_annotations]
        return "This image contains: " + ", ".join(labels)
    except Exception as e:
        return f"Error in scene understanding: {e}"


# Function: Text-to-Speech Conversion
def text_to_speech(image):
    """Extract text from image using OCR and convert to audio."""
    try:
        text = pytesseract.image_to_string(Image.open(image))
        st.write("Extracted Text:")
        st.write(text)
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return text
    except Exception as e:
        return f"Error in text-to-speech conversion: {e}"


# Function: Object and Obstacle Detection
def detect_objects(image_path):
    """Detect objects in the image using YOLOv5."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        results = model(image_path)
        detected_image = results.render()[0]
        return Image.fromarray(detected_image)
    except Exception as e:
        return f"Error in object detection: {e}"


# Function: Personalized Assistance
def personalized_assistance(image_path):
    """Provide task-specific guidance using LangChain."""
    try:
        # Use Google Vision to extract scene description
        description = describe_scene(image_path)

        # Generate context-specific guidance using LangChain
        template = PromptTemplate(input_variables=["description"],
                                  template="Provide personalized assistance for an image described as: {description}")
        chain = LLMChain(llm=llm, prompt=template)
        guidance = chain.run(description=description)

        return guidance
    except Exception as e:
        return f"Error in personalized assistance: {e}"


# Home Page
if selected_feature == "Home":
    st.subheader("🏠 Home")
    st.markdown("""
    Welcome to the AI Assistance application! Use the features in the sidebar to explore:
    - Real-Time Scene Descriptions
    - Text Extraction and Audio Reading
    - Object Detection for Navigation
    - Personalized Guidance for Daily Tasks
    """)
    # Display a static image with fallback
    try:
        st.image("https://source.unsplash.com/800x400/?AI,Technology", use_column_width=True,
                 caption="AI and Technology")
    except Exception:
        st.warning("Unable to load the online image. Showing a local placeholder instead.")
        st.image("local_placeholder.jpg", use_column_width=True,
                 caption="AI Assistance")  # Ensure the local image exists

# Real-Time Scene Understanding Feature
elif selected_feature == "Real-Time Scene Understanding":
    st.subheader("📸 Real-Time Scene Understanding")
    uploaded_image = st.file_uploader("Upload an Image for Scene Understanding", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getvalue())
            image_path = temp_file.name
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        description = describe_scene(image_path)
        st.write(description)

# Text-to-Speech Conversion Feature
elif selected_feature == "Text-to-Speech Conversion":
    st.subheader("🗣️ Text-to-Speech Conversion")
    uploaded_image = st.file_uploader("Upload an Image for Text Extraction and Audio", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getvalue())
            image_path = temp_file.name
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        speech_output = text_to_speech(image_path)
        st.write(speech_output)

# Object Detection Feature
elif selected_feature == "Object Detection":
    st.subheader("🔍 Object and Obstacle Detection")
    uploaded_image = st.file_uploader("Upload an Image for Object Detection", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getvalue())
            image_path = temp_file.name
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        detected_img = detect_objects(image_path)
        if isinstance(detected_img, str):
            st.write(detected_img)  # Show error if any
        else:
            st.image(detected_img, caption="Detected Objects", use_column_width=True)

# Personalized Assistance Feature
elif selected_feature == "Personalized Assistance":
    st.subheader("🧠 Personalized Assistance for Daily Tasks")
    uploaded_image = st.file_uploader("Upload an Image for Personalized Guidance", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getvalue())
            image_path = temp_file.name
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        guidance = personalized_assistance(image_path)
        st.write("Personalized Guidance:")
        st.write(guidance)

# Footer Section
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, LangChain, Google Vision API, and YOLOv5.")
