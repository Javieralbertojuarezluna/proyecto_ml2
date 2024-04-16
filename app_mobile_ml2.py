import numpy as np
import pandas as pd
import streamlit as st
import zipfile
import io
import joblib

# Función para cargar el modelo desde el archivo .zip
def load_model_from_zip():
    # Ruta del archivo zip
    zip_path = "rfc.zip"

    # Nombre del archivo .pkl dentro del zip
    pkl_filename = "rfc.pkl"

    # Cargar el archivo .pkl desde el zip
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        with zip_file.open(pkl_filename) as pkl_file:
            model = joblib.load(pkl_file)

    return model

# Cargar el modelo
model = load_model_from_zip()

# Columnas del conjunto de datos
cols = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi']

def main(): 
    st.title("Clasificador de Precios de celulares")
    html_temp = """
    <div style="background-color: #800020; padding: 10px">
    <h2 style="color: gold; text-align: center;">Universidad Panamericaca MCD</h2>
    <h3 style="color: gold; text-align: center; font-size: 16px;">Equipo:</h3>
    <ul style="color: gold; text-align: center; font-size: 14px; list-style-type: none;">
    <li>Alberto Jorge Julián</li>
    <li>Javier Alberto Juarez Luna</li>
    <li>Roberto Capitaine Venegas</li>
    </ul>
    </div>
    <br>
    
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Mapear valores numéricos a categorías para variables binarias
    binario_map = {0: "No", 1: "Sí"}

    battery_power = st.text_input("Total de batería que puede almacenar en mAh", "0") 
    blue = st.radio("¿Tiene Bluetooth?", options=[0, 1], format_func=lambda x: binario_map[x])
    clock_speed = st.text_input("Velocidad del microprocesador (segundos)", "0.0")
    dual_sim = st.radio("¿Doble SIM?", options=[0, 1], format_func=lambda x: binario_map[x])
    fc = st.text_input("MP de cámara frontal", "0")
    four_g = st.radio("¿Es 4G?", options=[0, 1], format_func=lambda x: binario_map[x])
    int_memory = st.text_input("Memoria interna en GB", "0")
    m_dep = st.text_input("Grosor del celular en cm", "0.0")
    mobile_wt = st.text_input("Peso del celular", "0")
    n_cores = st.text_input("Número de cores del procesador", "0")
    pc = st.text_input("MP de cámara principal", "0")
    px_height = st.text_input("Resolución de pixeles en altura", "0")
    px_width = st.text_input("Resolución de pixeles en anchura", "0")
    ram = st.text_input("Memoria RAM en MB", "0")
    sc_h = st.text_input("Altura de pantalla en cm", "0")
    sc_w = st.text_input("Anchura de pantalla en cm", "0")
    talk_time = st.text_input("Duración de batería en horas en una sola carga", "0")
    three_g = st.radio("¿Cuenta con 3G?", options=[0, 1], format_func=lambda x: binario_map[x])
    touch_screen = st.radio("¿Es touch screen?", options=[0, 1], format_func=lambda x: binario_map[x])
    wifi = st.radio("¿Tiene WIFI?", options=[0, 1], format_func=lambda x: binario_map[x])
    
    
    if st.button("Predict"): 
        features = [[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores,pc,px_height,px_width,ram,sc_h,sc_w,
        talk_time,three_g,touch_screen,wifi]]
        data = {'battery_power': int(battery_power), 'blue': int(blue), 'clock_speed': float(clock_speed), 'dual_sim': int(dual_sim), 'fc': int(fc),
                'four_g': int(four_g), 'int_memory': int(int_memory), 'm_dep': float(m_dep), 'mobile_wt':int(mobile_wt), 'n_cores':int(n_cores),
                'pc':int(pc), 'px_height':int(px_height), 'px_width':int(px_width), 'ram':int(ram), 'sc_h':int(sc_h), 'sc_w':int(sc_w),
                'talk_time':int(talk_time), 'three_g':int(three_g), 'touch_screen':int(touch_screen), 'wifi':int(wifi)}
        print(data)
        df = pd.DataFrame([list(data.values())], columns=cols)
        features_list = df.values.tolist() 
        prediction = model.predict(features_list)

        output = int(prediction[0])
        # Convertir la predicción numérica a una etiqueta de clase
        if prediction[0] == 0:
            text = "Costo Bajo"
        if prediction[0] == 1:
            text = "Costo Mediano"
        if prediction[0] == 2:
            text = "Costo Alto"
        if prediction[0] == 3:
            text = "Costo muy alto"
        else:
            text = "Error"

        st.success('Los datos indican que el celular tiene un {}'.format(text))

if __name__ == '__main__': 
    main()
