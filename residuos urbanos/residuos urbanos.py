import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminhos das imagens
caminhos = {
    "Saco de Lixo": r"C:\Users\Eduardo Carvalho\Downloads\visao_computacional-main\visao_computacional-main\residuos urbanos\saco de lixo.jpg",
    "Pneu Velho": r"C:\Users\Eduardo Carvalho\Downloads\visao_computacional-main\visao_computacional-main\residuos urbanos\pneu velho.jpg",
    "Lata Amassada": r"C:\Users\Eduardo Carvalho\Downloads\visao_computacional-main\visao_computacional-main\residuos urbanos\lata amassada 2.jpg"
}

def processar_imagem(titulo, caminho):
    # Carregar imagem
    imagem = cv2.imread(caminho)
    imagem_original = imagem.copy()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Redução de ruído
    imagem_blur = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

    # Detecção de bordas com Canny
    bordas = cv2.Canny(imagem_blur, 50, 150)

    # Operações morfológicas (fechamento para juntar contornos quebrados)
    kernel = np.ones((5, 5), np.uint8)
    bordas_fechadas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel)

    # Segmentação com Watershed
    # Transformar para binário
    ret, binaria = cv2.threshold(imagem_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remover ruídos
    abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)

    # Obter região de fundo
    fundo = cv2.dilate(abertura, kernel, iterations=3)

    # Obter região desconhecida
    dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)
    ret, certeza_objeto = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Marcar os rótulos
    certeza_objeto = np.uint8(certeza_objeto)
    desconhecido = cv2.subtract(fundo, certeza_objeto)

    ret, marcadores = cv2.connectedComponents(certeza_objeto)
    marcadores = marcadores + 1
    marcadores[desconhecido == 255] = 0

    # Aplicar Watershed
    marcadores = cv2.watershed(imagem, marcadores)
    imagem[marcadores == -1] = [0, 0, 255]  # Contorno em vermelho

    # Mostrar resultado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Canny + Morfologia")
    plt.imshow(bordas_fechadas, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Contorno (Watershed)")
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()

# Processar cada imagem
for titulo, caminho in caminhos.items():
    processar_imagem(titulo, caminho)
