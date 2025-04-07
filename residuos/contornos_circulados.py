import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

'''
img = cv2.imread('imagens/.png')

img_blur = cv2.GaussianBlur(img, (5, 5), 1.25)

lower = np.percentile(img_blur, 10)
upper = np.percentile(img_blur, 90)

bordas = cv2.Canny(img_blur, lower, upper)

contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contornos = img.copy()

cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)

if contornos:
    x, y, w, h = cv2.boundingRect(np.concatenate(contornos))
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('imagem original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title('objetos na imagem'), plt.xticks([]), plt.yticks([])

plt.show()'''

def processar_imagem(img):

    img_blur = cv2.GaussianBlur(img, (5, 5), 1.25)

    lower = np.percentile(img_blur, 10)
    upper = np.percentile(img_blur, 90)

    bordas = cv2.Canny(img_blur, lower, upper)

    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contornos = img.copy()

    img_objetos = img.copy()

    cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)

    if contornos:
        x, y, w, h = cv2.boundingRect(np.concatenate(contornos))
        
        cv2.rectangle(img_objetos, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return img, img_contornos, img_objetos


def mostrar_resultados(imagem_original, imagem_contorno, imagem_objetos):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
    plt.title('imagem original')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(imagem_contorno, cv2.COLOR_BGR2RGB))
    plt.title('contornos na imagem')
    plt.axis("off")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(imagem_objetos, cv2.COLOR_BGR2RGB))
    plt.title('objetos na imagem')
    plt.axis("off")
    plt.tight_layout()

    plt.show()


caminho_pasta = "imagens"
arquivos = [f for f in os.listdir(caminho_pasta) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for nome_arquivo in arquivos:
    caminho_imagem = os.path.join(caminho_pasta, nome_arquivo)
    imagem = cv2.imread(caminho_imagem)

    if imagem.shape[1] > 1280:
        escala = 1280 / imagem.shape[1]
        imagem = cv2.resize(imagem, (0, 0), fx=escala, fy=escala)

    imagem_original, imagem_contorno, imagem_objetos = processar_imagem(imagem)
    mostrar_resultados(imagem_original, imagem_contorno, imagem_objetos)
