import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

def download_model(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# ตัวอย่างลิงก์ Google Drive
black_model_url = "https://drive.google.com/file/d/12bJwAWoaSpMDkYhx2te2C5HF5ko-W6Yt/view?usp=sharing"
white_model_url = "https://drive.google.com/file/d/12FSQHYw1ldDGg9NDVdaVP_AgRU_YX_c_/view?usp=sharing"

download_model(black_model_url, "generator_model_black_100.h5")
download_model(white_model_url, "generator_model_white_100.h5")
# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# ฟังก์ชันประมวลผลและสร้างภาพ
def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))  # Resize เป็น 256x256
    image = image / 127.5 - 1  # Normalize [-1, 1]
    return tf.expand_dims(image, axis=0)  # เพิ่มมิติ batch

def generate_image(generator, input_image):
    input_image = preprocess_image(input_image)
    generated_image = generator(input_image, training=True)
    generated_image = (generated_image + 1) * 127.5  # Unnormalize [0, 255]
    return tf.squeeze(generated_image).numpy().astype("uint8")

# โหลดโมเดลทั้งสอง
generator_black = load_model("generator_model_black_100.h5")
generator_white = load_model("generator_model_white_100.h5")

# UI บน Streamlit
st.title("Pix2Pix: Edge to Artistic Image")
st.subheader("Choose a version to generate your artistic image.")

# เลือกโมเดลที่ต้องการใช้งาน
option = st.selectbox("Select a version:", ["Black Version", "White Version"])

if option == "Black Version":
    st.subheader("Upload an edge image for the Black version")
    uploaded_file = st.file_uploader("Upload an edge image", type=["jpg", "png"], key="black_version")
    generator = generator_black
elif option == "White Version":
    st.subheader("Upload an edge image for the White version")
    uploaded_file = st.file_uploader("Upload an edge image", type=["jpg", "png"], key="white_version")
    generator = generator_white

# ประมวลผลเมื่อมีการอัปโหลดภาพ
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image_np = np.array(input_image)

    st.image(input_image, caption="Input Image", use_container_width=True)

    with st.spinner("Generating image..."):
        output_image = generate_image(generator, input_image_np)
        st.image(output_image, caption="Generated Image", use_container_width=True)