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
    image, label = main_class.grab_images()
    # MNIST tensors come in BCHW; squeeze batch and channel to get HxW for Pillow/Gradio
    np_imdata = image.squeeze().mul(255).clamp(0, 255).byte().numpy()

    return np_imdata, str(label), image.squeeze(0)

def new_img_and_predict(model, main_class):
    img, txt, orig_img = setup_img(main_class)
    predicted = predict(model, orig_img)

    return img, str(txt), str(predicted)

with gr.Blocks() as demo:
    main_class = main.Main()
    model = setup_model()

    with gr.Tab("MNIST Image Predictor"):
        img_display = gr.Image(image_mode='L', height=280, width=280)
        txt_display = gr.Markdown(value="")
        predicted_display = gr.Markdown(value="")

        new_image = gr.Button('New Image')
        new_image.click(fn=lambda: new_img_and_predict(model, main_class), 
                        outputs=[img_display, txt_display, predicted_display])
    
    with gr.Tab("Custom Handwriting Detector"):
        gr.Markdown("WIP")



if __name__ == "__main__":
    demo.launch()