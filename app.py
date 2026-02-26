import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Configuración de Estilo Académico
st.set_page_config(page_title="Proyecto Probabilidad y Estadística", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6, p, label, span, .stMetric { 
        color: #000000 !important; 
        font-family: 'Arial', sans-serif;
    }
    .streamlit-expanderHeader {
        background-color: #e8f5e9 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
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
st.markdown("---")

# 2. Entrada de Datos
entrada = st.text_input("Ingresa los datos separados por espacios:", placeholder="Ej: 10 25 30 45 50")

if entrada:
    try:
        datos = [float(x) for x in entrada.replace(',', ' ').split()]
        n = len(datos)
        
        if n < 2:
            st.warning("⚠️ Se necesitan al menos 2 datos para el análisis.")
        else:
            # --- CÁLCULOS BASE ---
            datos_ordenados = sorted(datos)
            x_idx = np.arange(n) # El eje X es el índice (0, 1, 2...)
            y_vals = np.array(datos)
            
            sum_x = np.sum(x_idx)
            sum_y = np.sum(y_vals)
            sum_xy = np.sum(x_idx * y_vals)
            sum_x2 = np.sum(x_idx**2)
            
            media = sum_y / n
            
            # Regresión Lineal Manual para el procedimiento
            m_num = (n * sum_xy) - (sum_x * sum_y)
            m_den = (n * sum_x2) - (sum_x**2)
            m_slope = m_num / m_den if m_den != 0 else 0
            b_intercept = (sum_y - (m_slope * sum_x)) / n
            
            # Mediana
            if n % 2 != 0:
                mediana = datos_ordenados[n // 2]
            else:
                m1, m2 = datos_ordenados[n // 2 - 1], datos_ordenados[n // 2]
                mediana = (m1 + m2) / 2
            
            # Varianza y R2
            y_pred = m_slope * x_idx + b_intercept
            ss_res = np.sum((y_vals - y_pred)**2)
            ss_tot = np.sum((y_vals - media)**2)
            r_cuadrado = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            
            varianza = ss_tot / (n - 1)
            desviacion = np.sqrt(varianza)

            # --- VISUALIZACIÓN ---
            col_graf, col_tab = st.columns([2, 1])
            with col_graf:
                st.subheader("Gráfica de Tendencia Lineal")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(x_idx, y_vals, color='#2e7d32', s=100, label='Datos Reales')
                ax.plot(x_idx, y_pred, color='red', linewidth=2, label=f'Recta: y = {m_slope:.2f}x + {b_intercept:.2f}')
                ax.set_facecolor('#ffffff')
                ax.grid(True, alpha=0.3)
                plt.legend()
                st.pyplot(fig)
            with col_tab:
                st.subheader("Tabla de Datos")
                st.dataframe(pd.DataFrame({"X (Índice)": x_idx, "Y (Valor)": y_vals}), hide_index=True)

            # --- RESULTADOS FINALES ---
            st.divider()
            st.subheader("Resumen Estadístico")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("MEDIA (x̄)", f"{media:.2f}")
            r2.metric("MEDIANA (x̃)", f"{mediana:.2f}")
            r3.metric("VARIANZA (s²)", f"{varianza:.2f}")
            r4.metric("RECTA", f"y = {m_slope:.2f}x + {b_intercept:.2f}")
            r5.metric("R²", f"{r_cuadrado:.4f}")

            # --- PROCEDIMIENTO DETALLADO ---
            with st.expander("📝 VER PROCEDIMIENTO MATEMÁTICO COMPLETO"):
                # ... (Pasos anteriores de media y mediana se mantienen) ...
                
                st.markdown("### 1. Ecuación de la Recta ($y = mx + b$)")
                st.latex(r"m = \frac{n\sum(xy) - \sum x \sum y}{n\sum(x^2) - (\sum x)^2}")
                st.write(f"**Sustitución Pendiente (m):**")
                st.latex(fr"m = \frac{{{n}({sum_xy}) - ({sum_x})({sum_y})}}{{{n}({sum_x2}) - ({sum_x})^2}}")
                st.write(f"**Resultado m:** {m_slope:.4f}")
                
                st.latex(r"b = \frac{\sum y - m\sum x}{n}")
                st.write(f"**Sustitución Intersección (b):**")
                st.latex(fr"b = \frac{{{sum_y} - ({m_slope:.2f})({sum_x})}}{{{n}}}")
                st.write(f"**Resultado b:** {b_intercept:.4f}")
                st.success(f"**Ecuación Final:** $y = {m_slope:.4f}x + {b_intercept:.4f}$")

                st.markdown("---")
                st.markdown("### 2. Coeficiente de Determinación ($R^2$)")
                st.write("El $R^2$ mide qué tanto la recta explica la variación de los datos.")
                st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
                st.write(f"**Donde:**")
                st.write(f"Sumatoria residuos al cuadrado ($SS_{{res}}$): {ss_res:.4f}")
                st.write(f"Sumatoria total al cuadrado ($SS_{{tot}}$): {ss_tot:.4f}")
                st.latex(fr"R^2 = 1 - \frac{{{ss_res:.4f}}}{{{ss_tot:.4f}}}")
                st.info(f"**Resultado $R^2$:** {r_cuadrado:.4f} ({(r_cuadrado*100):.1f}% de precisión)")

    except Exception as e:
        st.error(f"Error: {e}")
