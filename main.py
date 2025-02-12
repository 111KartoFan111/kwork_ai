#!python
import argparse
import requests
from io import BytesIO
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Функция загрузки изображения
def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.RequestException as e:
        print(f"Ошибка загрузки изображения: {e}")
        exit(1)

# Функция обработки изображения и поиска зелёных зон
def process_image(image):
    image_np = np.array(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    image_hsv = cv2.GaussianBlur(image_hsv, (5, 5), 0)

    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 230, 230])
    mask = cv2.inRange(image_hsv, lower_green, upper_green)

    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return Image.fromarray(filled_mask)

# Функция генерации изображения
def generate_image(image, mask, prompt, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(device)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=7.5,
        num_inference_steps=50,
        strength=0.8
    ).images[0]

    result.save(output_path)
    print(f"Готово! Результат сохранён в {output_path}")

# Основная логика CLI
def main():
    parser = argparse.ArgumentParser(description="Автоматическая генерация благоустройства по изображению.")
    parser.add_argument("image_url", type=str, help="URL изображения")
    parser.add_argument("building_type", type=int, choices=range(1, 10), help="Тип здания (1-9)")
    parser.add_argument("--output", type=str, default="output.jpg", help="Путь сохранения результата")

    args = parser.parse_args()

    # Список промптов
    prompts = {
        1: "A cozy cottage with illuminated pathways, lush greenery, two parking spaces, a decorative fence, and several trees.",
        2: "A modern residential complex with driveways, pedestrian walkways, well-organized parking spaces, and abundant landscaping.",
        3: "A contemporary office building with a spacious entrance, eco-friendly design, and well-maintained green areas.",
        4: "A modern school with playgrounds, pedestrian-friendly walkways, well-placed lighting, and vibrant green landscaping.",
        5: "A colorful kindergarten with multiple playgrounds, pedestrian-friendly walkways, safe driveways, and lush greenery.",
        6: "A cozy café or restaurant with a well-lit entrance, parking spaces, pedestrian pathways, and decorative landscaping.",
        7: "A commercial store with convenient parking, pedestrian walkways, a well-lit entrance, and surrounding greenery.",
        8: "A luxury hotel with a grand entrance, well-organized parking, pedestrian-friendly walkways, and lush landscaping.",
        9: "A spacious warehouse facility with driveways, pedestrian pathways, designated parking, and functional lighting."
    }

    image = load_image(args.image_url)
    mask = process_image(image)
    generate_image(image, mask, prompts[args.building_type], args.output)

if __name__ == "__main__":
    main()
