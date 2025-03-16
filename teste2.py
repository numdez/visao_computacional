import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def preprocess_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    image[mask > 0] = [0, 255, 0]
    
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def calculate_ndvi(image):
    image = image.astype(np.float32)
    red = image[:, :, 2]
    nir = image[:, :, 0]
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = np.clip(ndvi, -1, 1)
    
    # Inverter a lógica da máscara para focar no não-verde
    mask_non_green = (image[:, :, 1] == 0) & (image[:, :, 0] == 0) & (image[:, :, 2] == 255)
    ndvi[mask_non_green] = 1  # Agora o NDVI é alterado nas áreas não-verde
    
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    return ndvi_normalized

def select_best_threshold_method(ndvi):
    _, otsu_threshold = cv2.threshold(ndvi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Focar nas áreas não verdes
    return otsu_threshold

def invert_image(image):
    """Inverte a imagem para que áreas destacadas se tornem não destacadas e vice-versa"""
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

curPath = Path(__file__).parent
imgsPath = f'{curPath}/imgs' 
image_files = [f for f in os.listdir(imgsPath) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"Erro: Nenhuma imagem encontrada na pasta '{imgsPath}'.")
    exit()

for image_file in image_files:
    image_path = os.path.join(imgsPath, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem '{image_file}'.")
        continue
    
    image_original = image.copy()
    image = preprocess_green(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ndvi = calculate_ndvi(image)
    
    best_threshold = select_best_threshold_method(ndvi)
    
    dilation_kernel = np.ones((7, 7), np.uint8)
    dilated_ndvi = cv2.dilate(best_threshold, dilation_kernel, iterations=2)
    
    # Inverter o resultado da limiarização
    inverted_result = invert_image(dilated_ndvi)
    
    # Aplicar a máscara invertida de destaque na imagem original
    highlighted_image = cv2.bitwise_and(image_original, image_original, mask=inverted_result)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title(f'Imagem Original: {image_file}')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ndvi, cmap='gray')
    plt.title('NDVI Normalizado')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(highlighted_image)
    plt.title('Imagem Original com Máscara Invertida de Destaque')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    deforested_area = np.sum(inverted_result == 255)
    total_pixels = inverted_result.size
    deforested_percentage = (deforested_area / total_pixels) * 100
    
    print(f"Imagem: {image_file}")
    print(f"Área Desmatada: {deforested_area} pixels ({deforested_percentage:.2f}%)")
    print("-" * 40)
