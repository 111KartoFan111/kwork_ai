from diffusers import StableDiffusionInpaintPipeline
import torch

# Определяем, есть ли CUDA (GPU) или используем CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
).to(device)

print("Модель успешно загружена!")
