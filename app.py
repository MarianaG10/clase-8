import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Cargar el modelo preentrenado
model = YOLO('yolov5s.pt')  # Asegúrate de tener este archivo en la raíz o en una ruta accesible

# Configuración de parámetros iniciales del modelo
model.conf = 0.25  # Umbral de confianza para NMS
model.iou = 0.45   # Umbral de IoU para NMS

# Interfaz de Streamlit
st.title("Detección de Objetos en Imágenes")

with st.sidebar:
    st.subheader('Parámetros de Configuración')
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0, model.iou)
    model.conf = st.slider('Seleccione el Confidence', 0.0, 1.0, model.conf)
    st.write('IoU:', model.iou)
    st.write('Conf:', model.conf)

picture = st.camera_input("Capturar foto")

if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Realizar la inferencia
    results = model.predict(source=cv2_img)

    # Obtener resultados
    detections = results[0].boxes
    boxes = detections.xyxy.cpu().numpy()
    scores = detections.conf.cpu().numpy()
    categories = detections.cls.cpu().numpy()

    col1, col2 = st.columns(2)

    with col1:
        # Dibujar las detecciones en la imagen
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(cv2_img, channels='BGR')

    with col2:
        label_names = model.names
        category_count = {}
        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        data = [{"Categoría": label_names[int(category)], "Cantidad": count} for category, count in category_count.items()]
        data2 = pd.DataFrame(data)

        # Agrupar los datos por la columna "Categoría" y sumar las cantidades
        df_sum = data2.groupby('Categoría')['Cantidad'].sum().reset_index()
        st.dataframe(df_sum)

   

    
