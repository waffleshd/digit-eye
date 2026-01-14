import gradio as gr
import main
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch


def predict(model, data):
    # Make prediction on the first image
    with torch.no_grad():
        logits = model(data.unsqueeze(0))  # Add batch dimension
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_digit = probabilities.max(1)
        predicted_digit = predicted_digit.item()
        confidence = confidence.item()

        #print(f"Predicted: {predicted_digit}, Actual: {label[0].item()}")
    return predicted_digit, confidence

def setup_model():
    model = main.NeuralNet()
    model.load_state_dict(torch.load('model.pth', weights_only=False))
    model.eval()  # Set to evaluation mode

    return model

def setup_img(main_class):
    image, label = main_class.grab_images()
    # MNIST tensors come in BCHW; squeeze batch and channel to get HxW for Pillow/Gradio
    np_imdata = image.squeeze().mul(255).clamp(0, 255).byte().numpy()
    
    # Scale image to 280x280 to fill the display area
    pil_img = Image.fromarray(np_imdata, mode='L')
    pil_img = pil_img.resize((280, 280), Image.NEAREST)
    np_imdata = np.array(pil_img)

    return np_imdata, str(label), image.squeeze(0)

def new_img_and_predict(model, main_class):
    img, txt, orig_img = setup_img(main_class)
    predicted, confidence = predict(model, orig_img)

    txt_html = f'<div style="font-size: 24px;"><b>MNIST Label:</b> {txt}</div>'

    if txt == str(predicted):
        predicted_html = f'<div style="font-size: 24px; color: green;"><b>Predicted:</b> {predicted} ({confidence*100:.1f}%)</div>'
    else:
        predicted_html = f'<div style="font-size: 24px; color: red;"><b>Predicted:</b> {predicted} ({confidence*100:.1f}%)</div>'

    return img, txt_html, predicted_html

def predict_custom_digit(canvas):
    if canvas is None:
        return "Please draw something"
    
    # Extract the numpy array from the canvas
    # ImageEditor returns a dict with 'composite' key containing the numpy array
    if isinstance(canvas, dict):
        img_array = canvas.get('composite')
    else:
        img_array = canvas
    
    if img_array is None:
        return "Please draw something"
    
    print(f"Raw canvas - shape: {img_array.shape}, dtype: {img_array.dtype}, min/max: {img_array.min()}/{img_array.max()}")
    
    # Handle the image data type
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # Invert if needed (white drawing on black background)
    img_array = 255 - img_array
    
    # Apply Gaussian blur to create gradients/anti-aliasing
    pil_img = Image.fromarray(img_array, mode='L')
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Resize from 280x280 to 28x28 (compress by factor of 10)
    pil_img = pil_img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(pil_img)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    
    print(f"Processed tensor shape: {img_tensor.shape}")
    print(f"Processed tensor min/max: {img_tensor.min()}/{img_tensor.max()}")
    
    # Make prediction
    predicted, confidence = predict(model, img_tensor)
    
    print(f"Predicted: {predicted}, Confidence: {confidence*100:.1f}%")
    
    return f"{predicted} ({confidence*100:.1f}%)"


with gr.Blocks() as demo:
    main_class = main.Main()
    model = setup_model()

    with gr.Tab("MNIST Image Predictor"):
        with gr.Row():
            img_display = gr.Image(image_mode='L', height=280, width=280)
            with gr.Column():
                txt_display = gr.HTML(value="")
                predicted_display = gr.HTML(value="")

        new_image = gr.Button('New Image')
        new_image.click(fn=lambda: new_img_and_predict(model, main_class), 
                        outputs=[img_display, txt_display, predicted_display])
    
    with gr.Tab("Custom Handwriting Detector"):
        with gr.Row():
            with gr.Column(min_width=300):
                canvas = gr.ImageEditor(
                    height=280,
                    width=280,
                    type="numpy",
                    image_mode="L",
                    canvas_size=(280,280),
                    brush=gr.Brush(
                        colors=["#000000", "#404040", "#808080", "#C0C0C0", "#FFFFFF"],
                        default_color="#000000",
                        default_size=10
                    ),
                )

        predict_btn = gr.Button("Predict")
        prediction_label = gr.Label(value="",label="Prediction")

        predict_btn.click(fn=predict_custom_digit, inputs=[canvas], outputs=prediction_label)



if __name__ == "__main__":
    demo.launch()