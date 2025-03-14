# %%
# Tudo
import cv2
import numpy as np
from pathlib import Path

curPath = Path(__file__).parent

# Carrega a imagem (substitua 'amazon_forest.jpg' pelo caminho da sua imagem)
img = cv2.imread(f'{curPath}/imgs/area_desmatada.jpg')
if img is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
    exit()

# Opcional: redimensiona para facilitar a visualização
img = cv2.resize(img, (800, 600))

# 1. Limiarização com Otsu
# Converte para escala de cinza e aplica thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray_blurred = cv2.GaussianBlur(gray, (5,5), 0)
ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("Valor de threshold (Otsu):", ret)

# 2. Segmentação baseada em região (Region Growing) utilizando floodFill
# Cria uma cópia da imagem para a operação e uma máscara (necessária para floodFill)
img_flood = img.copy()
mask = np.zeros((img_flood.shape[0] + 2, img_flood.shape[1] + 2), np.uint8)
# Define um ponto semente manualmente (ajuste conforme a imagem)
seed_point = (100, 100)
# floodFill: preenche a região a partir do seed_point com a cor vermelha
# Os valores de (loDiff, upDiff) definem a tolerância para a similaridade de cor
flood_flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
num, im_flood, mask_flood, rect = cv2.floodFill(img_flood, mask, seed_point, (0, 0, 255), (20, 20, 20), (20, 20, 20), flood_flags)
print("Número de pixels preenchidos pelo floodFill:", num)

# 3. Clusterização com K-means
# Prepara os dados (cada pixel é uma amostra com 3 características: B, G, R)
Z = img.reshape((-1, 3))
Z = np.float32(Z)
# Define os critérios e número de clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4  # número de clusters
ret_kmeans, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
img_kmeans = res.reshape((img.shape))
print("K-means convergiu com centroide:", center)

# 4. Segmentação com Watershed
# Para aplicar o watershed, é necessário obter uma imagem binária
ret_thresh, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Remove ruídos utilizando abertura morfológica
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# Determina a área de fundo com dilatação
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Calcula a transformada de distância e obtém a área de primeiro plano
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret_dt, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
# Determina a área desconhecida subtraindo o primeiro plano do fundo
unknown = cv2.subtract(sure_bg, sure_fg)
# Marca os rótulos dos componentes conectados
ret_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # para garantir que o fundo seja diferente de 0
markers[unknown == 255] = 0
# Aplica o watershed; os contornos detectados serão marcados com -1
markers = cv2.watershed(img, markers)
img_watershed = img.copy()
img_watershed[markers == -1] = [255, 0, 0]  # marca os contornos em azul

# 5. Color Filtering em HSV
# Converte a imagem para o espaço de cor HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Define o intervalo para a cor verde (aproximado para vegetação)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
# Aplica a máscara para extrair apenas as áreas verdes
img_color_filtered = cv2.bitwise_and(img, img, mask=mask_green)

# Exibe os resultados em janelas separadas
cv2.imshow("Imagem Original", img)
cv2.imshow("Thresholding Otsu", thresh_otsu)
cv2.imshow("Region Growing (FloodFill)", img_flood)
cv2.imshow("K-means Clustering", img_kmeans)
cv2.imshow("Watershed", img_watershed)
cv2.imshow("Color Filtering (Green)", img_color_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
