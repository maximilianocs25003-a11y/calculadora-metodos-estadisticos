import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# 1. Configuración de Estilo Académico
st.set_page_config(page_title="Proyecto Probabilidad y Estadística", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3, p, label, .stMetric { color: #000000 !important; font-family: sans-serif; }
    .stTextInput>div>div>input { background-color: #f0f2f6; color: #000000; border: 1px solid #2e7d32; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Calculadora Estadística: Proyecto Personal")

# 2. Entrada de Datos
entrada = st.text_input("Ingresa los datos (separados por espacios)", placeholder="Ej: 15 20 18 22 15 30")

if entrada:
    try:
        # Procesamiento de datos
        datos_originales = [float(x) for x in entrada.replace(',', ' ').split()]
        n = len(datos_originales)
        
        if n < 2:
            st.warning("⚠️ Se requieren al menos 2 datos.")
        else:
            # Cálculos
            y = np.array(datos_originales)
            x_idx = np.arange(n)
            media = np.mean(y)
            mediana = np.median(y)
            
            # Moda manual para evitar errores de scipy
            frecuencias = {}
            for d in datos_originales: frecuencias[d] = frecuencias.get(d, 0) + 1
            max_f = max(frecuencias.values())
            if max_f > 1:
                modas = [k for k, v in frecuencias.items() if v == max_f]
                moda_texto = ", ".join(map(str, modas))
            else:
                moda_texto = "No hay moda"
            
            varianza = np.var(y, ddof=1)
            desviacion = np.std(y, ddof=1)
            
            # Regresión
            m, b_reg = np.polyfit(x_idx, y, 1)
            linea = m * x_idx + b_reg
            r_cuadrado = 1 - (np.sum((y - linea)**2) / np.sum((y - media)**2))

            # --- Visualización ---
            col_graf, col_tab = st.columns([2, 1])
            with col_graf:
                st.subheader("Tendencia de los Datos")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(x_idx, y, color='#2e7d32', s=100, label='Datos', edgecolors='black')
                ax.plot(x_idx, linea, color='red', linewidth=2, label='Tendencia')
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                st.pyplot(fig)
            with col_tab:
                st.subheader("Datos Ingresados")
                st.dataframe(pd.DataFrame({"Índice": x_idx, "Valor": y}), hide_index=True)

            # --- Ordenamiento y Mediana ---
            st.divider()
            st.subheader("Tratamiento de Datos: Orden Ascendente")
            datos_ordenados = sorted(datos_originales)
            df_ordenado = pd.DataFrame({"Posición": range(1, n+1), "Valor": datos_ordenados})
            
            def resaltar_mediana(row):
                if n % 2 != 0:
                    target = (n + 1) // 2
                    return ['background-color: #c8e6c9' if row['Posición'] == target else '' for _ in row]
                else:
                    target1, target2 = n // 2, (n // 2) + 1
                    return ['background-color: #c8e6c9' if row['Posición'] in [target1, target2] else '' for _ in row]

            st.table(df_ordenado.style.apply(resaltar_mediana, axis=1))

            # --- Resultados ---
            st.divider()
            res = st.columns(5)
            res[0].metric("MEDIA", f"{media:.2f}")
            res[1].metric("MODA", moda_texto)
            res[2].metric("MEDIANA", f"{mediana:.2f}")
            res[3].metric("VARIANZA", f"{varianza:.2f}")
            res[4].metric("DESV. EST.", f"{desviacion:.2f}")

            # --- Procedimiento Paso a Paso (Corregido) ---
            with st.expander("📝 VER PROCEDIMIENTO DETALLADO"):
                # Media
                st.markdown("#### 1. Cálculo de la Media")
                st.latex(r"\bar{x} = \frac{\sum x_i}{n}")
                suma_total = sum(datos_originales)
                st.write(f"Sustitución: $\\bar{{x}} = \\frac{{{suma_total}}}{{{n}}}$")
                st.write(f"**Resultado:** {media:.4f}")

                # Mediana
                st.markdown("#### 2. Cálculo de la Mediana")
                st.write(f"Datos ordenados: {datos_ordenados}")
                if n % 2 != 0:
                    st.write(f"Posición central: {(n+1)//2}")
                else:
                    st.write(f"Promedio de posiciones: {n//2} y {(n//2)+1}")
                st.write(f"**Resultado:** {mediana}")

                # Varianza
                st.markdown("#### 3. Cálculo de la Varianza (s²)")
                st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}")
                suma_cuadrados = sum([(val - media)**2 for val in datos_originales])
                st.write(f"Suma de cuadrados de desviaciones: {suma_cuadrados:.4f}")
                st.write(f"Operación: ${suma_cuadrados:.4f} / ({n} - 1)$")
                st.write(f"**Resultado:** {varianza:.4f}")

                # Desviación
                st.markdown("#### 4. Cálculo de la Desviación Estándar (s)")
                st.latex(r"s = \sqrt{s^2}")
                st.write(f"Operación: $\\sqrt{{{varianza:.4f}}}$")
                st.write(f"**Resultado:** {desviacion:.4f}")

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
