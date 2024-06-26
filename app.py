import cv2
from deepface import DeepFace
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# puede ser mas optimo con una mask circular
# Funciones para Fourier

# distancia euclidiana
def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# gaussiano paso bajo.
def gaussianLP(shape, D0=50):
    rows, cols = shape
    base = np.zeros((rows, cols))
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = np.exp(((-distance((y, x), center)**2) / (2 * (D0**2))))
    return cv2.merge([base, base])

# Función para aplicar filtro de Fourier con filtro Gaussiano de paso bajo
def fourier_filter_gaussian(frame, D0=50):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (15, 15), 0)
    dft = cv2.dft(np.float32(imgray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = imgray.shape
    if not hasattr(fourier_filter_gaussian, "gaussian_mask") or fourier_filter_gaussian.gaussian_mask.shape[:2] != (rows, cols):
        fourier_filter_gaussian.gaussian_mask = gaussianLP((rows, cols), D0)
    
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    compact, label, (centers) = cv2.kmeans(pix_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    segmented_image = res.reshape((frame.shape))
    return segmented_image

cap = cv2.VideoCapture(0)
if cap:
    print("Webcam connected")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Almacenar las emociones predichas y verdaderas
y_true = []
y_pred = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    frame_filtered = fourier_filter_gaussian(frame)
    frame_segmented = segmentation_kmeans(frame, k=10)

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame_filtered, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame_segmented, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(frame_filtered.shape) == 2:
        frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_GRAY2BGR)
        
    if len(frame_segmented.shape) == 2:
        frame_segmented = cv2.cvtColor(frame_segmented, cv2.COLOR_GRAY2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_filtered_rgb = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2RGB)
    frame_segmented_rgb = cv2.cvtColor(frame_segmented, cv2.COLOR_BGR2RGB)

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

    # Mostrar emociones
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_filtered, dominant_emotion_filtered, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_segmented, dominant_emotion_segmented, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Guardar las emociones verdaderas y predichas
    y_true.append(dominant_emotion)  # Aquí se agrega las emociones verdaderas en lugar de la predicha
    y_pred.append(dominant_emotion)  # Esta es la emoción predicha

    combined_frame = np.hstack((frame_filtered, frame, frame_segmented))
    cv2.imshow('Emotion Detection', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Verifica si y_true y y_pred no están vacíos
if y_true and y_pred:
    # Calcular la precisión para la emoción "happy"
    precision_happy = precision_score(y_true, y_pred, labels=['happy'], average='binary', pos_label='happy')
    print(f'Precision for happy: {precision_happy}')

    # Calcular la precisión general
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')

    # Calcular otras métricas
    report = classification_report(y_true, y_pred)
    print(report)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
else:
    print("No se detectaron emociones. Verifique la detección de rostros y las predicciones de emociones.")

cap.release()
cv2.destroyAllWindows()
