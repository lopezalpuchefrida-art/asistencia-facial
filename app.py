import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Control de Asistencia Facial", layout="centered")

CARPETA_ROSTROS = 'rostros_registrados'
os.makedirs(CARPETA_ROSTROS, exist_ok=True)

@st.cache_resource
def cargar_rostros():
    known_encodings = []
    known_names = []
    archivos = [f for f in os.listdir(CARPETA_ROSTROS) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fn in archivos:
        img = face_recognition.load_image_file(os.path.join(CARPETA_ROSTROS, fn))
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(os.path.splitext(fn)[0].capitalize())
    return known_encodings, known_names

known_encodings, known_names = cargar_rostros()

if 'asistencia' not in st.session_state:
    st.session_state.asistencia = pd.DataFrame(columns=['Nombre', 'Fecha', 'Hora'])

st.title("📷 Control de Asistencia Facial")

opcion = st.sidebar.selectbox("Menú Principal", ["Pasar Asistencia", "Registrar Nueva Persona", "Ver Lista de Asistencia"])

if opcion == "Pasar Asistencia":
    st.header("Pase de Lista")
    img_file_buffer = st.camera_input("Captura tu rostro para registrar asistencia")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, locations)

        nombre_detectado = "Desconocido"
        for loc, enc in zip(locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, enc)
            dists = face_recognition.face_distance(known_encodings, enc)
            if len(dists) > 0:
                best_idx = np.argmin(dists)
                if matches[best_idx]:
                    nombre_detectado = known_names[best_idx]

        if nombre_detectado != "Desconocido":
            if nombre_detectado not in st.session_state.asistencia['Nombre'].values:
                ahora = datetime.now()
                nueva_fila = pd.DataFrame([{
                    'Nombre': nombre_detectado,
                    'Fecha': ahora.strftime('%Y-%m-%d'),
                    'Hora': ahora.strftime('%H:%M:%S')
                }])
                st.session_state.asistencia = pd.concat([st.session_state.asistencia, nueva_fila], ignore_index=True)
                st.success(f"✅ ¡Bienvenido(a), {nombre_detectado}! Asistencia registrada.")
            else:
                st.warning(f"⚠️ {nombre_detectado} ya había registrado asistencia.")
        else:
            st.error("❌ Rostro no reconocido. Regístrate primero en el menú lateral.")

elif opcion == "Registrar Nueva Persona":
    st.header("Añadir Usuario")
    nuevo_nombre = st.text_input("Nombre completo:")
    foto_nueva = st.file_uploader("Sube una foto clara del rostro", type=['jpg', 'jpeg', 'png'])

    if st.button("Guardar Registro") and nuevo_nombre and foto_nueva:
        path = os.path.join(CARPETA_ROSTROS, f"{nuevo_nombre.strip().lower()}.jpg")
        with open(path, "wb") as f:
            f.write(foto_nueva.getbuffer())
        st.cache_resource.clear()
        st.success(f"🎉 {nuevo_nombre} registrado(a) con éxito.")

elif opcion == "Ver Lista de Asistencia":
    st.header("📋 Reporte de Asistencia")
    st.dataframe(st.session_state.asistencia, use_container_width=True)
    if not st.session_state.asistencia.empty:
        csv = st.session_state.asistencia.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Reporte Excel/CSV", csv, "asistencia.csv", "text/csv")
