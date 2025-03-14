import cv2
import numpy as np
from pathlib import Path

curPath = Path(__file__).parent

# Carrega a imagem (substitua 'amazon_forest.jpg' pelo caminho da sua imagem)
img = cv2.imread(f'{curPath}/imgs/area_desmatada.jpg')
if img is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
    exit()

img = cv2.resize(img, (800, 600))

# --- 1. Pré-processamento ---
# Aplica um filtro Gaussiano para suavizar a imagem
img_blur = cv2.GaussianBlur(img, (7, 7), 0)

# Aplica um filtro bilateral mais agressivo para melhor remoção de ruídos
img_bilateral = cv2.bilateralFilter(img_blur, 15, 100, 100)

# Converte para tons de cinza
gray = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)

# Aplica um filtro de mediana para remover ruídos pontuais
gray = cv2.medianBlur(gray, 5)

# Aplica Threshold de Otsu
ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Valor de threshold (Otsu):", ret)

# --- 2. Segmentação por Região (FloodFill) ---
img_flood = img.copy()
mask = np.zeros((img_flood.shape[0] + 2, img_flood.shape[1] + 2), np.uint8)

# Seleção automática do ponto semente
contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    seed_point = tuple(contours[0][0][0])  # Ponto inicial automático
else:
    seed_point = (100, 100)  # Caso não haja contornos detectados

num, im_flood, mask_flood, rect = cv2.floodFill(
    img_flood, mask, seed_point, (0, 0, 255), (10, 10, 10), (10, 10, 10), 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
)
print("Número de pixels preenchidos pelo floodFill:", num)

# --- 3. Clusterização com K-means ---
img_filtered = cv2.medianBlur(img, 7)  # Reduz ruídos antes do K-means
Z = img_filtered.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4  # Número de clusters ajustável
ret_kmeans, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
img_kmeans = res.reshape((img.shape))
img_kmeans = cv2.medianBlur(img_kmeans, 5)  # Suaviza ruídos após o K-means
print("K-means convergiu com centroide:", center)

# --- 4. Segmentação com Watershed ---
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fecha pequenas falhas
sure_bg = cv2.dilate(closing, kernel, iterations=5)
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
ret_dt, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img_watershed = img.copy()
img_watershed[markers == -1] = [255, 0, 0]

# --- 5. Filtragem Automática de Cor HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define automaticamente os intervalos para a cor verde com base no histograma
hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
peak_hue = np.argmax(hist_hue)
lower_green = np.array([max(peak_hue - 10, 0), 40, 40])
upper_green = np.array([min(peak_hue + 10, 179), 255, 255])

# Aplica a máscara automaticamente
mask_green = cv2.inRange(hsv, lower_green, upper_green)
img_color_filtered = cv2.bitwise_and(img, img, mask=mask_green)

'''
res = cv2.bitwise_and(img, img, mask=mask_green)
kernel = np.ones((5, 5), np.uint8)
img_color_filtered = cv2.morphologyEx(mask_green, cv2.MORPH_GRADIENT, kernel)
'''




# Exibe os resultados
cv2.imshow("Imagem Original", img)
cv2.imshow("Thresholding Otsu", thresh_otsu)
cv2.imshow("Region Growing (FloodFill)", img_flood)
cv2.imshow("K-means Clustering", img_kmeans)
cv2.imshow("Watershed", img_watershed)
cv2.imshow("Color Filtering (Green)", img_color_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()