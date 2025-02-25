import torch
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/control_v11p_sd15_canny"

# Load ControlNet (Canny Edge Detection)
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

# Load Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
pipe.to("cuda", device_map="auto")

input_image_path = "untitled.png"
image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
image = cv2.resize(image, (840, 840))

canny = cv2.Canny(image, 90, 210)
#canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

# Convert to PIL image format
canny_pil = Image.fromarray(canny)

options = [
    "a double bend",
    "an upwards slope",
    "a downwards slope",
    "a wide curve right",
    "a wide curve left",
    "a tight curve right",
    "a tight curve left",
    "a crosswalk",
    "a zebra-crossing",
]
generated_images = []
for text in options:
    image = pipe(
        prompt=f"imagine the continuation of this road with {text}",
        image=canny_pil,
        num_inference_steps=30,
        guidance_scale=8.5
    ).images[0]
    
    generated_images.append(image)

for i, img in enumerate(generated_images):
    img.save(f"outputs/generated_{i+1}_{input_image_path}")
