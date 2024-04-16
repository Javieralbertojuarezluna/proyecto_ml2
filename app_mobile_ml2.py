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
    st.title("Predictor de Precios de celulares")
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
    battery_power = st.text_input("Battery power", "0") 
    blue = st.text_input("Blue", "0")
    clock_speed = st.text_input("Clock speed", "0.0")
    dual_sim = st.text_input("Dual sim", "0")
    fc = st.text_input("fc", "0")
    four_g = st.text_input("four_g", "0")
    int_memory = st.text_input("int_memory", "0")
    m_dep = st.text_input("m_dep", "0.0")
    mobile_wt = st.text_input("mobile_wt", "0")
    n_cores = st.text_input("n_cores", "0")
    pc = st.text_input("pc", "0")
    px_height = st.text_input("px_height", "0")
    px_width = st.text_input("px_width", "0")
    ram = st.text_input("ram", "0")
    sc_h = st.text_input("sc_h", "0")
    sc_w = st.text_input("sc_w", "0")
    talk_time = st.text_input("talk_time", "0")
    three_g = st.text_input("three_g", "0")
    touch_screen = st.text_input("touch_screen", "0")
    wifi = st.text_input("wifi", "0")
    
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
