import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib

model = joblib.load('rfc_vf.pkl')
cols=['battery_power', 'mobile_wt','px_height', 'px_width', 'ram']    

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
    battery_power = st.text_input("Total de batería que puede almacenar en mAh", "0")
    mobile_wt = st.text_input("Peso del celular", "0")
    px_height = st.text_input("Resolución de pixeles en altura", "0")
    px_width = st.text_input("Resolución de pixeles en anchura", "0")
    ram = st.text_input("Memoria RAM en MB", "0")
    
    if st.button("Predict"): 
        features = [[battery_power, mobile_wt,px_height, px_width, ram]]
        data = {'battery_power': int(battery_power),'mobile_wt':int(mobile_wt),'px_height':int(px_height), 'px_width':int(px_width), 'ram':int(ram) }
        print(data)
        df = pd.DataFrame([list(data.values())], columns=cols)
        features_list = df.values.tolist() 
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 0:
            text = "Costo Bajo"
        elif output == 1:
            text = "Costo Mediano"
        elif output == 2:
            text = "Costo Alto"
        elif output == 3:
            text = "Costo muy alto"
        else:
            text = "Error"

        st.success('Los datos indican que el celular tiene un {}'.format(text))

if __name__ == '__main__': 
    main()


