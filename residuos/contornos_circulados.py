import numpy as np
import cv2
from matplotlib import pyplot as plt

# Carregar a imagem
img = cv2.imread('imagens/imagem_urbana.png')

# Aplicar o desfoque gaussiano para reduzir ruídos
img_blur = cv2.GaussianBlur(img, (5, 5), 1.25)

# Definir os limites para detecção de bordas
lower = np.percentile(img_blur, 10)
upper = np.percentile(img_blur, 90)

# Aplicar o operador Canny para detectar as bordas
bordas = cv2.Canny(img_blur, lower, upper)

# Encontrar os contornos na imagem
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Criar uma cópia da imagem original para desenhar os contornos (opcional)
img_contornos = img.copy()

# Desenhar os contornos na imagem (opcional)
cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)

# Verificar se há contornos
if contornos:
    # Encontrar os pontos de limite da bounding box englobando todos os contornos
    # A função cv2.boundingRect calcula a caixa delimitadora mínima que envolve todos os contornos
    x, y, w, h = cv2.boundingRect(np.concatenate(contornos))  # Junta todos os contornos em um único array
    
    # Desenhar a bounding box global na imagem original
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Caixa azul
    
# Exibir as imagens
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Bounding Box'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title('Edge Image with Contours'), plt.xticks([]), plt.yticks([])

plt.show()
