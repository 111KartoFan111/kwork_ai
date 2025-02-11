from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
import torch

# Запрос ссылки у пользователя
image_url = input("Введите URL изображения: ")
try:
    response = requests.get(image_url)
    response.raise_for_status()
except requests.RequestException as e:
    print(f"Ошибка загрузки изображения: {e}")
    input("Нажмите Enter для выхода...")
    exit()

image = Image.open(BytesIO(response.content)).convert("RGB")

# Обработка изображения (поиск зелёных зон)
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
mask_pil = Image.fromarray(filled_mask)

# Меню выбора промптов
prompts = {
    1: { 'name': 'Коттедж', 'prompt': "A cozy cottage with illuminated pathways, lush greenery, two parking spaces, a decorative fence, and several trees." },
    2: { 'name': 'Жилой дом', 'prompt': "A modern residential complex with driveways, pedestrian walkways, well-organized parking spaces, children's and sports playgrounds, an adult relaxation area, street lamps, and abundant landscaping." },
    3: { 'name': 'Офисное здание', 'prompt': "A contemporary office building with a spacious entrance, eco-friendly design, pedestrian zones, bicycle parking, and well-maintained green areas." },
    4: { 'name': 'Школа', 'prompt': "A modern school with safe driveways, playgrounds, pedestrian-friendly walkways, well-placed lighting, and vibrant green landscaping." },
    5: { 'name': 'Детский сад', 'prompt': "A colorful kindergarten with multiple playgrounds, pedestrian-friendly walkways, safe driveways, and lush greenery." },
    6: { 'name': 'Кафе/Ресторан', 'prompt': "A cozy café or restaurant with a well-lit entrance, parking spaces, pedestrian pathways, and decorative landscaping with street lamps." },
    7: { 'name': 'Магазин', 'prompt': "A commercial store with convenient parking, pedestrian walkways, a well-lit entrance, and surrounding greenery." },
    8: { 'name': 'Гостиница', 'prompt': "A luxury hotel with a grand entrance, well-organized parking, pedestrian-friendly walkways, decorative street lighting, and lush landscaping." },
    9: { 'name': 'Склад', 'prompt': "A spacious warehouse facility with driveways, pedestrian pathways, designated parking, and surrounding greenery with functional lighting." }
}

print("Выберите тип здания:")
for key, value in prompts.items():
    print(f"{key}. {value['name']}")

try:
    choice = int(input("Введите номер (1-9): "))
    if choice not in prompts:
        raise ValueError
except ValueError:
    print("Неверный выбор! Завершение программы.")
    input("Нажмите Enter для выхода...")
    exit()

selected_prompt = prompts[choice]['prompt']

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(device)

# Генерация изображения
result = pipe(
    prompt=selected_prompt,
    image=image,
    mask_image=mask_pil,
    guidance_scale=7.5,
    num_inference_steps=50,
    strength=0.8
).images[0]

result.save("output.png")
print("Готово! Результат сохранён в output.png")
input("Нажмите Enter для выхода...")
