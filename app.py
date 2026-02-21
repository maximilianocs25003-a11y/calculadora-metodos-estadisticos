import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Configuración de Estilo (Blanco y Negro con Verde Bosque)
st.set_page_config(page_title="Proyecto Probabilidad y Estadística", layout="wide")

st.markdown("""
    <style>
    /* Forzar fondo blanco en toda la app */
    .stApp { background-color: #ffffff; }
    
    /* Forzar color negro en TODOS los textos y cabeceras */
    h1, h2, h3, h4, h5, h6, p, label, li, span, div { 
        color: #000000 !important; 
        font-family: 'Arial', sans-serif;
    }
    
    /* Estilo para el Expander (Procedimiento) */
    .st-expander {
        border: 1px solid #2e7d32 !important;
        background-color: #f9f9f9 !important;
    }
    
    /* Inputs y Botones */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #2e7d32;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Proyecto de Probabilidad y Estadística")

# 2. Entrada de Datos
entrada = st.text_input("Ingresa los datos separados por espacios:", placeholder="Ej: 80 95 110 110 135")

if entrada:
    try:
        datos = [float(x) for x in entrada.replace(',', ' ').split()]
        n = len(datos)
        
        if n < 2:
            st.warning("⚠️ Se necesitan al menos 2 datos para el análisis.")
        else:
            # --- CÁLCULOS BASE ---
            datos_ordenados = sorted(datos)
            suma_total = sum(datos)
            media = suma_total / n
            
            # Moda
            frecuencias = {}
            for d in datos: frecuencias[d] = frecuencias.get(d, 0) + 1
            max_f = max(frecuencias.values())
            if max_f > 1:
                modas = [k for k, v in frecuencias.items() if v == max_f]
                moda_res = ", ".join(map(str, modas))
            else:
                moda_res = "No existe (Amodal)"
            
            # Mediana
            if n % 2 != 0:
                mediana = datos_ordenados[n // 2]
            else:
                m1, m2 = datos_ordenados[n // 2 - 1], datos_ordenados[n // 2]
                mediana = (m1 + m2) / 2
            
            # Varianza y Desviación
            suma_desviaciones_cuad = sum([(x - media)**2 for x in datos])
            varianza = suma_desviaciones_cuad / (n - 1)
            desviacion = np.sqrt(varianza)

            # --- VISUALIZACIÓN ---
            col_graf, col_tab = st.columns([2, 1])
            with col_graf:
                st.subheader("Gráfica de Tendencia")
                x_idx = np.arange(n)
                m_reg, b_reg = np.polyfit(x_idx, datos, 1)
                linea = m_reg * x_idx + b_reg
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(x_idx, datos, color='#1b5e20', s=100, label='Datos', edgecolors='black')
                ax.plot(x_idx, linea, color='red', linewidth=2, label='Línea de Tendencia')
                ax.set_facecolor('#ffffff')
                ax.grid(True, linestyle=':', alpha=0.5)
                st.pyplot(fig)

            with col_tab:
                st.subheader("Tabla de Datos")
                st.dataframe(pd.DataFrame({"Posición": range(1, n+1), "Valor": datos}), hide_index=True)

            # --- TABLA ORDENADA ---
            st.divider()
            st.subheader("Datos Ordenados y Mediana")
            df_ord = pd.DataFrame({"Posición": range(1, n+1), "Valor": datos_ordenados})
            
            def highlight_mediana(row):
                if n % 2 != 0:
                    target = (n + 1) // 2
                    return ['background-color: #c8e6c9' if row['Posición'] == target else '' for _ in row]
                else:
                    t1, t2 = n // 2, (n // 2) + 1
                    return ['background-color: #c8e6c9' if row['Posición'] in [t1, t2] else '' for _ in row]
            
            st.table(df_ord.style.apply(highlight_mediana, axis=1))

            # --- RESULTADOS FINALES ---
            st.divider()
            st.subheader("Resumen de Resultados")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("MEDIA", f"{media:.2f}")
            r2.metric("MODA", moda_res)
            r3.metric("MEDIANA", f"{mediana:.2f}")
            r4.metric("VARIANZA", f"{varianza:.2f}")
            r5.metric("DESV. EST.", f"{desviacion:.2f}")

            # --- PROCEDIMIENTO DETALLADO ---
            with st.expander("📝 CLIC AQUÍ PARA VER EL PROCEDIMIENTO PASO A PASO"):
                st.markdown("### 1. MEDIA ARITMÉTICA ($\mu$)")
                st.latex(r"\text{Fórmula: } \bar{x} = \frac{\sum x_i}{n}")
                st.write(f"**Sustitución:** $\\bar{{x}} = \\frac{{{suma_total}}}{{{n}}}$")
                st.write(f"**Resolución:** Se dividen todos los valores sumados entre el total de elementos.")
                st.write(f"**Resultado:** {media:.4f}")

                st.markdown("---")
                st.markdown("### 2. MEDIANA ($\tilde{x}$)")
                st.write(f"**Paso 1:** Ordenar datos de menor a mayor: `{datos_ordenados}`")
                if n % 2 != 0:
                    st.latex(r"\text{Fórmula (Impar): } \text{Posición} = \frac{n+1}{2}")
                    st.write(f"**Sustitución:** Posición = $\\frac{{{n}+1}}{{2}} = {(n+1)/2}$")
                else:
                    st.latex(r"\text{Fórmula (Par): } \frac{X_{n/2} + X_{(n/2)+1}}{2}")
                    st.write(f"**Sustitución:** $\\frac{{{datos_ordenados[n//2-1]} + {datos_ordenados[n//2]}}}{{2}}$")
                st.write(f"**Resultado:** {mediana}")

                st.markdown("---")
                st.markdown("### 3. VARIANZA MUESTRAL ($s^2$)")
                st.latex(r"\text{Fórmula: } s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1}")
                st.write(f"**Sustitución (Suma de Cuadrados):** $s^2 = \\frac{{{suma_desviaciones_cuad:.4f}}}{{{n}-1}}$")
                st.write(f"**Resolución:** Se resta la media a cada dato, se eleva al cuadrado, se suma todo y se divide entre n-1.")
                st.write(f"**Resultado:** {varianza:.4f}")

                st.markdown("---")
                st.markdown("### 4. DESVIACIÓN ESTÁNDAR ($s$)")
                st.latex(r"\text{Fórmula: } s = \sqrt{s^2}")
                st.write(f"**Sustitución:** $s = \\sqrt{{{varianza:.4f}}}$")
                st.write(f"**Resolución:** Se extrae la raíz cuadrada del resultado de la varianza.")
                st.write(f"**Resultado:** {desviacion:.4f}")

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
