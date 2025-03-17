import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def preprocess_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definindo múltiplos intervalos de verde
    lower_green_1 = np.array([25, 40, 40])
    upper_green_1 = np.array([90, 255, 255])
    
    lower_green_2 = np.array([40, 50, 40])
    upper_green_2 = np.array([100, 255, 255])
    
    lower_green_3 = np.array([60, 40, 40])
    upper_green_3 = np.array([120, 255, 255])

    # Criando as máscaras para cada intervalo de verde
    mask1 = cv2.inRange(hsv, lower_green_1, upper_green_1)
    mask2 = cv2.inRange(hsv, lower_green_2, upper_green_2)
    mask3 = cv2.inRange(hsv, lower_green_3, upper_green_3)
    
    # Combinando as máscaras para cobrir mais tons de verde
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    
    # Usando a máscara invertida para remover os verdes
    image_no_green = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    return image_no_green, mask

def calculate_ndvi(image):
    # Para o NDVI, consideramos a imagem sem as áreas verdes
    image = image.astype(np.float32)
    red = image[:, :, 2]  # Banda Vermelha
    nir = image[:, :, 0]  # Banda Infravermelha
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = np.clip(ndvi, -1, 1)
    
    # Normalizar o NDVI para um valor entre 0 e 255
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    return ndvi_normalized

def select_best_threshold_method(ndvi):
    _, otsu_threshold = cv2.threshold(ndvi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Focar nas áreas não verdes
    return otsu_threshold

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
    
    # Remover as árvores aplicando múltiplas camadas de filtro de verde
    image_no_green, green_mask = preprocess_green(image)
    
    # Calcular o NDVI usando a imagem final sem as áreas verdes
    ndvi = calculate_ndvi(image_no_green)
    
    # Seleção do melhor limiar de threshold
    best_threshold = select_best_threshold_method(ndvi)
    
    # Dilation para destacar as áreas de interesse
    dilation_kernel = np.ones((7, 7), np.uint8)
    dilated_ndvi = cv2.dilate(best_threshold, dilation_kernel, iterations=2)
    
    plt.figure(figsize=(15, 5))
    
    # Exibir a imagem original
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Imagem Original: {image_file}')
    plt.axis('off')
    
    # Exibir a imagem sem os verdes
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(image_no_green, cv2.COLOR_BGR2RGB))
    plt.title('Imagem sem as Árvores')
    plt.axis('off')
    
    # Exibir o NDVI
    plt.subplot(1, 4, 3)
    plt.imshow(ndvi, cmap='gray')
    plt.title('NDVI Normalizado')
    plt.axis('off')
    
    # Exibir a imagem dilatada com o threshold
    plt.subplot(1, 4, 4)
    plt.imshow(dilated_ndvi, cmap='gray')
    plt.title('Threshold com Dilation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Cálculo da área desmatada
    deforested_area = np.sum(dilated_ndvi == 255)
    total_pixels = dilated_ndvi.size
    deforested_percentage = (deforested_area / total_pixels) * 100
    
    print(f"Imagem: {image_file}")
    print(f"Área Desmatada: {deforested_area} pixels ({deforested_percentage:.2f}%)")
    print("-" * 40)
