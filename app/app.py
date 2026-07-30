import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

from torchvision.models import resnet18, ResNet18_Weights

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 Cats vs Dogs Classifier")

st.write(
    "Upload an image and let the AI predict whether it is a Cat or a Dog."
)

# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 2)

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "model" / "cats_vs_dogs_resnet18.pth"

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model, device, weights


model, device, weights = load_model()

# ==========================================
# Upload Image
# ==========================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ==========================================
    # Preprocessing
    # ==========================================

    transform = weights.transforms()

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ==========================================
    # Prediction
    # ==========================================

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    classes = ["Cat", "Dog"]

    predicted_class = classes[predicted.item()]

    confidence = confidence.item() * 100

    # ==========================================
    # Display Result
    # ==========================================

    st.divider()

    st.subheader("Prediction Result")

    if predicted_class == "Cat":
        st.success("🐱 Cat")
    else:
        st.success("🐶 Dog")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.divider()

    st.subheader("Prediction Probabilities")

    cat_prob = probabilities[0][0].item()
    dog_prob = probabilities[0][1].item()

    st.write("🐱 Cat")

    st.progress(cat_prob)

    st.write(f"{cat_prob*100:.2f}%")

    st.write("🐶 Dog")

    st.progress(dog_prob)

    st.write(f"{dog_prob*100:.2f}%")