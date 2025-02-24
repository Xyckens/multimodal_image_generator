import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

# Load pre-trained models
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "thibaud/controlnet-sd15-canny"

# Load ControlNet (Canny Edge Detection)
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)

# Load Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
)

# Use efficient scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Move to GPU
pipe.to("cuda")

# Load input image
input_image_path = "input_road.png"  # Replace with your image path
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Apply Canny Edge Detection
canny = cv2.Canny(image, 100, 200)
canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

# Convert to PIL image format
from PIL import Image
canny_pil = Image.fromarray(canny)

# Display Canny Edge Image
plt.figure(figsize=(5, 5))
plt.imshow(canny, cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show()

# Prompt variations for generating different lane styles
options = [
    "a double bend",
    "an upwards slope",
    "a downwards slope",
    "a wide curve right",
    "a wide curve left",
    "a crosswalk",
]
# Generate variations
generated_images = []
for text in options:
    image = pipe(
        prompt=f"A minimalistic empty road with an orange lane with {text}, white background",
        image=canny_pil,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    generated_images.append(image)

# Show all generated images
fig, axes = plt.subplots(1, len(generated_images), figsize=(15, 5))
for i, img in enumerate(generated_images):
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Variation {i+1}")
plt.show()

# Save generated images
for i, img in enumerate(generated_images):
    img.save(f"generated_road_{i+1}.png")
