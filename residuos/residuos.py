import cv2
import numpy as np
import matplotlib.pyplot as plt

imPath = "imagem.jpg"

imagem = cv2.imread(imPath)

if imagem.shape[1] > 1280:
    escala = 1280 / imagem.shape[1]
    imagem = cv2.resize(imagem, (0, 0), fx=escala, fy=escala)

def blur_adaptativo(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contraste = np.std(gray)
    sigma = max(0.3, min(1.5, 1.5 - (contraste / 100)))
    kernel = int(2 * round(sigma * 2) + 1)
    return cv2.GaussianBlur(img_bgr, (kernel, kernel), sigmaX=sigma)

imagem_blur = blur_adaptativo(imagem)

cinza = cv2.cvtColor(imagem_blur, cv2.COLOR_BGR2GRAY)
borda = cv2.Canny(cinza, 100, 200)

kernel = np.ones((5, 5), np.uint8)
borda_dilatada = cv2.dilate(borda, kernel, iterations=1)

_, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_inv = cv2.bitwise_not(thresh)

kernel_small = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel_small, iterations=2)

sure_bg = cv2.dilate(opening, kernel_small, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1 
markers[unknown == 255] = 0

imagem_watershed = imagem.copy()
cv2.watershed(imagem_watershed, markers)
imagem_watershed[markers == -1] = [0, 0, 255]

contornos, _ = cv2.findContours(borda_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > 700]
bboxes = [cv2.boundingRect(c) for c in contornos_filtrados]

def contem_todos(bbox_grande, bboxes_pequenas):
    xg, yg, wg, hg = bbox_grande
    for xb, yb, wb, hb in bboxes_pequenas:
        if not (xg <= xb and yg <= yb and xg + wg >= xb + wb and yg + hg >= yb + hb):
            return False
    return True

indice_englobador = None
for i, bbox in enumerate(bboxes):
    outras_bboxes = bboxes[:i] + bboxes[i+1:]
    if contem_todos(bbox, outras_bboxes):
        indice_englobador = i
        break

imagem_saida = imagem.copy()
if indice_englobador is not None:
    c = contornos_filtrados[indice_englobador]
    x, y, w, h = bboxes[indice_englobador]
    cv2.drawContours(imagem_saida, [c], -1, (0, 255, 0), 3)
    cv2.putText(imagem_saida, "Objeto principal", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    for idx, c in enumerate(contornos_filtrados, 1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(imagem_saida, [c], -1, (255, 0, 0), 2)
        cv2.putText(imagem_saida, f"Objeto {idx}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagem_saida, cv2.COLOR_BGR2RGB))
plt.title("Resultado final com contornos")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imagem_watershed, cv2.COLOR_BGR2RGB))
plt.title("Imagem segmentada (Watershed)")
plt.axis("off")
plt.tight_layout()
plt.show()
