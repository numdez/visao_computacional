import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Função para calcular o NDVI
def calculate_ndvi(image):
    # Converter a imagem para float32
    image = image.astype(np.float32)
    
    # Extrair as bandas do espectro visível (vermelho e infravermelho)
    red = image[:, :, 2]  # Banda do vermelho
    nir = image[:, :, 0]  # Banda do infravermelho (ajuste conforme necessário)
    
    # Calcular o NDVI
    ndvi = (nir - red) / (nir + red + 1e-8)  # Evitar divisão por zero
    ndvi = np.clip(ndvi, -1, 1)  # Limitar valores entre -1 e 1
    
    # Normalizar o NDVI para o intervalo [0, 255]
    ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return ndvi_normalized

# Caminho da pasta com as imagens
curPath = Path(__file__).parent

imgsPath = f'{curPath}/imgs' 

# Listar todas as imagens na pasta
image_files = [f for f in os.listdir(imgsPath) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Verificar se há imagens na pasta
if not image_files:
    print(f"Erro: Nenhuma imagem encontrada na pasta '{imgsPath}'.")
    exit()

# Processar cada imagem
for image_file in image_files:
    # Caminho completo da imagem
    image_path = os.path.join(imgsPath, image_file)
    
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem '{image_file}'.")
        continue
    
    # Converter a imagem para RGB (para exibição)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Segmentação por NDVI
    ndvi = calculate_ndvi(image)
    
    # Aplicar um threshold no NDVI para identificar áreas desmatadas
    _, binary_ndvi = cv2.threshold(ndvi, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Operações morfológicas para remover ruídos
    kernel = np.ones((5, 5), np.uint8)
    morph_ndvi = cv2.morphologyEx(binary_ndvi, cv2.MORPH_OPEN, kernel)
    morph_ndvi = cv2.morphologyEx(morph_ndvi, cv2.MORPH_CLOSE, kernel)
    
    # 2. Segmentação por Clustering (K-means)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    k = 2  # Número de clusters (vegetação e desmatamento)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)
    
    # Identificar a área desmatada no clustering
    # O cluster com menor intensidade média é considerado desmatamento
    if np.mean(centers[0]) < np.mean(centers[1]):
        desmatamento_label = 0
    else:
        desmatamento_label = 1
    
    # Criar uma máscara para a área desmatada no clustering
    desmatamento_mask = (labels == desmatamento_label).reshape(image.shape[:2])
    
    # Verificar visualmente se a máscara está correta
    # Se a máscara estiver destacando áreas não desmatadas, invertemos a lógica
    if np.sum(desmatamento_mask) > 0.5 * desmatamento_mask.size:  # Se mais de 50% da imagem for marcada como desmatamento
        desmatamento_label = 1 - desmatamento_label  # Inverte o rótulo
        desmatamento_mask = (labels == desmatamento_label).reshape(image.shape[:2])
    
    # Exibir resultados
    plt.figure(figsize=(15, 10))
    
    # Imagem original
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title(f'Imagem Original: {image_file}')
    plt.axis('off')
    
    # NDVI
    plt.subplot(2, 3, 2)
    plt.imshow(ndvi, cmap='gray')
    plt.title('NDVI')
    plt.axis('off')
    
    # Sobreposição do NDVI na imagem original
    overlay_ndvi = image_rgb.copy()
    overlay_ndvi[morph_ndvi == 255] = [255, 0, 0]  # Destacar áreas em vermelho
    plt.subplot(2, 3, 3)
    plt.imshow(overlay_ndvi)
    plt.title('NDVI Destacado')
    plt.axis('off')
    
    # Clustering (K-means)
    plt.subplot(2, 3, 4)
    plt.imshow(segmented_image)
    plt.title('Clustering (K-means)')
    plt.axis('off')
    
    # Sobreposição do Clustering na imagem original
    overlay_clustering = image_rgb.copy()
    overlay_clustering[desmatamento_mask] = [255, 0, 0]  # Destacar áreas em vermelho
    plt.subplot(2, 3, 5)
    plt.imshow(overlay_clustering)
    plt.title('Clustering Destacado')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Quantificação das áreas desmatadas
    deforested_area_ndvi = np.sum(morph_ndvi == 255)
    deforested_area_clustering = np.sum(desmatamento_mask)
    total_pixels = morph_ndvi.size
    
    deforested_percentage_ndvi = (deforested_area_ndvi / total_pixels) * 100
    deforested_percentage_clustering = (deforested_area_clustering / total_pixels) * 100
    
    print(f"Imagem: {image_file}")
    print(f"Área Desmatada (NDVI): {deforested_area_ndvi} pixels ({deforested_percentage_ndvi:.2f}%)")
    print(f"Área Desmatada (Clustering): {deforested_area_clustering} pixels ({deforested_percentage_clustering:.2f}%)")
    print("-" * 40)