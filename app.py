import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models

from PIL import Image
import streamlit as st

import asyncio


device = 'cuda'
MODEL_PATH = 'models/CatDog_ResNet50.pth'
INPUT_SIZE = 224
CLASSES = ['Cat', 'Dog']
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

@st.cache_resource

def run_async_task(async_func, *args):
 
    loop = None

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_func(*args))
    except:
        if loop is not None:
            loop.close()

        loop = asyncio.new_event_loop()

        loop.run_until_complete(async_func(*args))
    finally:
        if loop is not None:
            loop.close()



transform_input = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


def load_model():
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

async def main():
    st.title("Cat And Dog Classification")

    uploaded_file = st.file_uploader("Input a image: ", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption = 'Selected Image', use_container_width=True)

        img_tensor = transform_input(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred = torch.argmax(probs).item()

        st.markdown(f"Model Predict: {CLASSES[pred]}")
        st.markdown(f"Confidence: ")
        for i, label in enumerate(CLASSES):
            st.write(f"{label} : {probs[i]*100:.2f} %")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    


