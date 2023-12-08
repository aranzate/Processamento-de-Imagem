# opencv: biblioteca para reconhecer textos
# pip install opencv-python
from re import T
import cv2
import numpy as np

# tesseract: inteligencia de reconhecimento de caracteres
# pip install pytesseract
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

images = [
    "text.PNG",
    "cascao2.PNG",
    "milcairao.jpg",
    "placa.jpg",
    "raiox.jpg",
    "noisy.png"
]

def getGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def removeNoise(image):
    return cv2.medianBlur(image,5)

def showImage(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)

def getBoxes(image):
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Loop through the detected words
    for i in range(len(boxes['text'])):
        # Check if the word is not empty and has a confidence level
        if boxes['text'][i] != '' and int(boxes['conf'][i]) > 0:
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
            conf = boxes['conf'][i]
            
            # Draw a rectangle around the word
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(image, boxes['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image

def getBoxesByCharacter(image):
    boxes = pytesseract.pytesseract.image_to_boxes(image)
    if len(image.shape) == 2:  # Grayscale image (height, width)
        imH, imW = image.shape
    elif len(image.shape) == 3:  # Color image (height, width, channels)
        imH, imW, _ = image.shape

    #splitlines breaks in lines
    for b in boxes.splitlines():
        b = b.split(' ')
        caractere,x,y,w,h = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(image, (x, imH-y), (w, imH-h), (0,0,255), 1)


def getImageToString(image):
    return pytesseract.image_to_string(image)

def getImageToBoxes(image):
    return pytesseract.image_to_boxes(image)

def getGaussian(image, varGa1, varGa2):
    return cv2.GaussianBlur(image, (varGa1, varGa2), 0)

def getNormalization(image):
    norm_img = np.zeros((image.shape[0],image.shape[1]))
    return cv2.normalize(image, norm_img, 15, 255, cv2.NORM_MINMAX)

def getThresholding(image):
    return cv2.threshold(image, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def filterCombo(image):
    resultado = getImageToString(getGray(image))
    print("String encontrada na imagem:\n" + resultado)
    getBoxes(getGray(image))
    showImage(getGray(image))
    
    gauss_image = getGaussian(getGray(image), 3, 3)
    threshold_image = getThresholding(gauss_image)
    norm_image = getNormalization(threshold_image)

    resultado = getImageToString(norm_image)
    print("String encontrada na imagem:\n" + resultado)
    getBoxesByCharacter(norm_image)
    showImage(norm_image)

def xRay(image):
    getBoxes(image)
    showImage(image)
    gauss_image = getGaussian(getGray(image), 3, 3) # Aplica o desfoque Gaussiano na imagem
    norm_image = getNormalization(gauss_image)

    resultado = getImageToString(norm_image)
    print("String encontrada na imagem:\n" + resultado)

    getBoxes(norm_image)
    showImage(norm_image)

def noisyImage(image):
    resultado = getImageToString(getGray(image))
    print("String encontrada na imagem:\n" + resultado)
    getBoxes(getGray(image))
    showImage(getGray(image))
    
    reduced_noises_image = getGaussian(getGray(image), 3, 3)
    threshold_image = getThresholding(reduced_noises_image)
    norm_image = getNormalization(threshold_image)

    resultado = getImageToString(norm_image)
    print("String encontrada na imagem:\n" + resultado)
    getBoxesByCharacter(norm_image)
    showImage(norm_image)


for imageName in images:

    # ler imagem
    image = cv2.imread(imageName) 
    print(imageName)

    #tres combinações de filtros utilizadas nas imagens, retire o comentário em uma das funções abaixo para utilizar a combinação de filtros

    #filterCombo(image)
    #noisyImage(image)
    #xRay(image)

    # String encontrada na imagem
    resultado = getImageToString(image)
    print("String encontrada na imagem:\n" + resultado)

    # Localização das palavras encontradas na imagem
    resultado2 =  getImageToBoxes(image)
    print("Localização das palavras encontradas na imagem:\n" +resultado2)

    # Converte a imagem para escala de cinza
    gray_image = getGray(image) 
    showImage(gray_image)

    # Remove ruído da imagem
    no_noise_image = removeNoise(image)
    showImage(no_noise_image)

    # Obtém as palavras da imagem
    boxedImage = getBoxes(image) # Obtém as palavras da imagem
    showImage(image)

    # Obtém caracteres da imagem
    image2 = cv2.imread(imageName) 
    boxedImage = getBoxesByCharacter(image2) 
    showImage(image2)
    
    