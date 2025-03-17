from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

def preprocess_green(image):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        image_no_green = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        return image_no_green, mask
    except Exception as e:
        raise RuntimeError(f"Erro no pré-processamento: {str(e)}")

def calculate_vari(image):
    try:
        image = image.astype(np.float32)
        red = image[:, :, 2]
        green = image[:, :, 1]
        blue = image[:, :, 0]
        
        vari = (green - red) / (green + red - blue + 1e-8)
        vari_normalized = ((vari + 1) / 2 * 255).astype(np.uint8)
        return vari_normalized
    except Exception as e:
        raise RuntimeError(f"Erro no cálculo do VARI: {str(e)}")

def apply_threshold_and_dilate(vari):
    try:
        _, otsu_threshold = cv2.threshold(vari, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(otsu_threshold, kernel, iterations=1)
        return dilated
    except Exception as e:
        raise RuntimeError(f"Erro na limiarização: {str(e)}")

def apply_mask_to_original(image, mask):
    try:
        # Cria imagem com áreas removidas destacadas em vermelho
        negative_mask = cv2.bitwise_not(mask)
        highlighted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        highlighted[negative_mask == 255] = [255, 0, 0]  # Destaca em vermelho
        
        masked_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        return masked_image, highlighted
    except Exception as e:
        raise RuntimeError(f"Erro na aplicação da máscara: {str(e)}")

def encode_image(image):
    try:
        if image is None:
            return None
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Erro na codificação da imagem: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return render_template('index.html', error="Nenhum arquivo enviado")
            
            file = request.files['image']
            if file.filename == '':
                return render_template('index.html', error="Nenhum arquivo selecionado")
            
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return render_template('index.html', error="Formato não suportado. Use PNG, JPG ou JPEG")

            img_bytes = file.read()
            np_array = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return render_template('index.html', error="Formato de imagem inválido")

            image_no_green, green_mask = preprocess_green(image)
            vari_image = calculate_vari(image_no_green)
            threshold_image = apply_threshold_and_dilate(vari_image)
            final_image, negative_mask = apply_mask_to_original(image, threshold_image)

            return render_template('index.html',
                original_image=encode_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                image_no_green=encode_image(image_no_green),
                ndvi_image=encode_image(vari_image),
                threshold_image=encode_image(threshold_image),
                final_image_with_mask=encode_image(final_image),
                negative_mask_image=encode_image(negative_mask))
            
        except Exception as e:
            error_msg = f"Erro no processamento: {str(e)}"
            print(error_msg)
            return render_template('index.html', error=error_msg)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)