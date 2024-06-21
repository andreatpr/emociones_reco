import cv2
from deepface import DeepFace
import numpy as np

# puede ser mas optimo con una mask circular
# Funciones para Fourier

# distancia euclidiana
def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# gaussiano paso bajo.
def gaussianLP(shape, D0=50):
    rows, cols = shape
    base = np.zeros((rows, cols))
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return cv2.merge([base, base])

# Función para aplicar filtro de Fourier con filtro Gaussiano de paso bajo
def fourier_filter_gaussian(frame, D0=50):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (15, 15), 0)
    dft = cv2.dft(np.float32(imgray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = imgray.shape
    # hasattr: Comprueba si el objeto tiene el atributo gaussian_mask(para no recalcularlo cada vez que se llama a la función)
    if not hasattr(fourier_filter_gaussian, "gaussian_mask") or fourier_filter_gaussian.gaussian_mask.shape[:2] != (rows, cols):
        fourier_filter_gaussian.gaussian_mask = gaussianLP((rows, cols), D0)
    
    # Aplicar filtro
    fshift = dft_shift * fourier_filter_gaussian.gaussian_mask
    f_ishift = np.fft.ifftshift(fshift)
    imgorig = cv2.idft(f_ishift)
    imgorig = cv2.magnitude(imgorig[:, :, 0], imgorig[:, :, 1])
    imgorig = cv2.normalize(imgorig, None, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(imgorig)

# Función para aplicar KMeans segmentation
def segmentation_kmeans(frame, k=2):
    pix_val = frame.reshape((-1, 3))
    pix_val = np.float32(pix_val)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)  # Reducir a 5 iteraciones
    compact, label, (centers) = cv2.kmeans(pix_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    segmented_image = res.reshape((frame.shape))
    return segmented_image

# captura webcam
cap = cv2.VideoCapture(1)
if(cap):
    print("Webcam connected")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Captura frame
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # aplica filtros
    frame_filtered = fourier_filter_gaussian(frame)
    frame_segmented = segmentation_kmeans(frame, k=10)

    # Iterar sobre cada cara detectada
    for i, (x, y, w, h) in enumerate(faces):
        # Dibujar rectángulo alrededor de la cara en cada fotograma
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame_filtered, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame_segmented, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    # convierte grises a BGR 
    if len(frame_filtered.shape) == 2: 
        frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_GRAY2BGR)
        
    if len(frame_segmented.shape) == 2:
        frame_segmented = cv2.cvtColor(frame_segmented, cv2.COLOR_GRAY2BGR)


    # convierte a rgb para deepface
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_filtered_rgb = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2RGB)
    frame_segmented_rgb = cv2.cvtColor(frame_segmented, cv2.COLOR_BGR2RGB)

    # analisa con deepface
    try:
        result_filtered = DeepFace.analyze(frame_filtered_rgb, actions=['emotion'], enforce_detection=False)
        dominant_emotion_filtered = result_filtered[0]['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing filtered frame: {e}")
        dominant_emotion_filtered = "No face detected"

    try:
        result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing original frame: {e}")
        dominant_emotion = "No face detected"

    try:
        result_segmented = DeepFace.analyze(frame_segmented_rgb, actions=['emotion'], enforce_detection=False)
        dominant_emotion_segmented = result_segmented[0]['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing segmented frame: {e}")
        dominant_emotion_segmented = "No face detected"

    # muestra emociones
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_filtered, dominant_emotion_filtered, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_segmented, dominant_emotion_segmented, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # combino en un frame
    combined_frame = np.hstack((frame_filtered, frame,  frame_segmented))
    cv2.imshow('Emotion Detection', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
