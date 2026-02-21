import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# 1. Configuración de Estilo Académico (Blanco y Negro)
st.set_page_config(page_title="Proyecto Probabilidad y Estadística", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    h1, h2, h3, p, label, .stMetric {
        color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
        color: #000000;
        border: 1px solid #2e7d32;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .highlight-mediana {
        background-color: #c8e6c9 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Calculadora Estadística: Proyecto Personal")
st.write("Introduce los datos de tu muestra para generar el análisis completo y el reporte de procedimientos.")

# 2. Entrada de Datos
entrada = st.text_input("Ingresa los datos (separados por espacios)", placeholder="Ej: 15 20 18 22 15 30")

if entrada:
    try:
        # Procesamiento inicial
        datos_originales = [float(x) for x in entrada.replace(',', ' ').split()]
        n = len(datos_originales)
        
        if n < 2:
            st.warning("⚠️ Se requieren al menos 2 datos para realizar los cálculos.")
        else:
            # Cálculos Estadísticos
            y = np.array(datos_originales)
            x_idx = np.arange(n)
            
            media = np.mean(y)
            mediana = np.median(y)
            # Moda
            moda_res = stats.mode(y, keepdims=True)
            moda = moda_res.mode[0] if moda_res.count[0] > 1 else "No hay moda única"
            
            varianza = np.var(y, ddof=1)
            desviacion = np.std(y, ddof=1)
            
            # Regresión Lineal
            m, b_reg = np.polyfit(x_idx, y, 1)
            linea = m * x_idx + b_reg
            r_cuadrado = 1 - (np.sum((y - linea)**2) / np.sum((y - media)**2))

            # --- SECCIÓN DE GRÁFICA Y TABLA DE DATOS ---
            st.divider()
            col_graf, col_tab = st.columns([2, 1])

            with col_graf:
                st.subheader("Tendencia de los Datos")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(x_idx, y, color='#2e7d32', s=100, label='Datos (Muestra)', edgecolors='black')
                ax.plot(x_idx, linea, color='red', linewidth=2, label=f'Tendencia Lineal')
                ax.set_facecolor('#ffffff')
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                st.pyplot(fig)

            with col_tab:
                st.subheader("Datos Ingresados")
                df_original = pd.DataFrame({"Índice": x_idx, "Valor": y})
                st.dataframe(df_original, use_container_width=True, hide_index=True)

            # --- SECCIÓN DE ORDENAMIENTO Y MEDIANA ---
            st.divider()
            st.subheader("Tratamiento de Datos: Orden Ascendente")
            datos_ordenados = sorted(datos_originales)
            
            # Crear DataFrame para resaltar la mediana
            df_ordenado = pd.DataFrame({"Posición": range(1, n+1), "Valor": datos_ordenados})
            
            def resaltar_mediana(row):
                if n % 2 != 0: # Impar
                    idx_mediana = (n + 1) // 2
                    return ['background-color: #c8e6c9' if row['Posición'] == idx_mediana else '' for _ in row]
                else: # Par
                    idx1, idx2 = n // 2, (n // 2) + 1
                    return ['background-color: #c8e6c9' if row['Posición'] in [idx1, idx2] else '' for _ in row]

            st.write("En la tabla inferior se resaltan el o los valores centrales utilizados para la mediana:")
            st.table(df_ordenado.style.apply(resaltar_mediana, axis=1))

            # --- RESULTADOS FINALES ---
            st.divider()
            st.subheader("Reporte de Resultados")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MEDIA", f"{media:.2f}")
            c2.metric("MODA", f"{moda if isinstance(moda, str) else round(moda, 2)}")
            c3.metric("MEDIANA", f"{mediana:.2f}")
            c4.metric("VARIANZA", f"{varianza:.2f}")
            c5.metric("DESV. EST.", f"{desviacion:.2f}")

            # --- PROCEDIMIENTO PASO A PASO ---
            with st.expander("📝 VER PROCEDIMIENTO DETALLADO (Paso a Paso)"):
                st.markdown("### 1. Cálculo de la Media ($\mu$ o $\\bar{x}$)")
                st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")
                st.write(f"Sustitución: $\\bar{x} = \\frac{{{' + '.join(map(str, datos_originales))}}}{{{n}}}$")
                st.write(f"**Resultado:** {media:.4f}")

                st.markdown("### 2. Cálculo de la Mediana")
                st.write(f"Datos ordenados: {datos_ordenados}")
                if n % 2 != 0:
                    st.write(f"Como n ({n}) es impar, la mediana es el valor central en la posición $\\frac{{{n}+1}}{{2}} = {(n+1)//2}$")
                else:
                    st.write(f"Como n ({n}) es par, la mediana es el promedio de las posiciones $\\frac{{{n}}}{{2}} = {n//2}$ y $\\frac{{{n}}}{{2}}+1 = {(n//2)+1}$")
                st.write(f"**Resultado:** {mediana}")

                st.markdown("### 3. Cálculo de la Varianza Muestral ($s^2$)")
                st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}")
                suma_cuadrados = sum([(val - media)**2 for val in datos_originales])
                st.write(f"Suma de desviaciones al cuadrado: {suma_cuadrados:.4f}")
                st.write(f"División: $\\frac{{{suma_cuadrados:.4f}}}{{{n} - 1}}$")
                st.write(f"**Resultado:** {varianza:.4f}")

                st.markdown("### 4. Cálculo de la Desviación Estándar ($s$)")
                st.latex(r"s = \sqrt{s^2}")
                st.write(f"Sustitución: $s = \\sqrt{{{varianza:.4f}}}$")
                st.write(f"**Resultado:** {desviacion:.4f}")

                st.markdown("### 5. Coeficiente de Determinación ($R^2$)")
                st.write(f"Este valor indica qué tan bien la línea roja representa tus datos.")
                st.write(f"**Precisión del modelo:** {r_cuadrado:.4f}")

    except ValueError:
        st.error("❌ Error: Ingresa solo números válidos.")