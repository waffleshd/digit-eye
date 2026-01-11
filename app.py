import gradio as gr
import main
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision


def predict(model, data):
    # Make prediction on the first image
    with torch.no_grad():
        logits = model(data.unsqueeze(0))  # Add batch dimension
        predicted_digit = logits.argmax(1).item()  # Get the digit with highest probability

        #print(f"Predicted: {predicted_digit}, Actual: {label[0].item()}")

    return predicted_digit

def setup_model():
    model = main.NeuralNet()
    model.load_state_dict(torch.load('model.pth', weights_only=False))
    model.eval()  # Set to evaluation mode

    return model

def setup_img(main_class):
    images, labels = main_class.grab_images()
    # MNIST tensors come in CHW; squeeze channel and move to HxW for Pillow/Gradio
    np_imdata = images[0].squeeze(0).mul(255).clamp(0, 255).byte().numpy()

with gr.Blocks() as demo:
    main_class = main.Main()

    img, txt = setup_img(main_class)
    
    gr.Image(value=img, image_mode='L', height=280, width=280)
    gr.Markdown(value=txt)



if __name__ == "__main__":
    demo.launch()