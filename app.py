import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Configuración de Estilo Académico (Blanco, Negro y Verde Bosque)
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
        border: 1px solid #2e7d32;
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

st.title("📊 Calculadora Estadística Integral")
st.markdown("### Proyecto de Probabilidad y Estadística")
st.markdown("---")

# 2. Entrada de Datos
entrada = st.text_input("📝 Ingresa los datos separados por espacios:", placeholder="Ej: 10 15 15 20 25")

if entrada:
    try:
        # --- PROCESAMIENTO INICIAL ---
        datos = [float(x) for x in entrada.replace(',', ' ').split()]
        n = len(datos)
        
        if n < 2:
            st.warning("⚠️ Se necesitan al menos 2 datos para realizar el análisis completo.")
        else:
            # Cálculos de apoyo
            datos_ordenados = sorted(datos)
            x_idx = np.arange(n)
            y_vals = np.array(datos)
            suma_y = sum(datos)
            media = suma_y / n
            
            # 1. Moda
            frec = {}
            for d in datos: frec[d] = frec.get(d, 0) + 1
            max_f = max(frec.values())
            modas = [k for k, v in frec.items() if v == max_f]
            moda_res = ", ".join(map(str, modas)) if max_f > 1 else "Amodal"

            # 2. Mediana
            if n % 2 != 0:
                mediana = datos_ordenados[n // 2]
            else:
                m1, m2 = datos_ordenados[n // 2 - 1], datos_ordenados[n // 2]
                mediana = (m1 + m2) / 2
            
            # 3. Varianza y Desviación
            suma_desviaciones_cuad = sum([(val - media)**2 for val in datos])
            varianza = suma_desviaciones_cuad / (n - 1)
            desviacion = np.sqrt(varianza)

            # 4. Regresión (Ecuación y R2)
            sum_x = np.sum(x_idx)
            sum_x2 = np.sum(x_idx**2)
            sum_xy = np.sum(x_idx * y_vals)
            
            m_num = (n * sum_xy) - (sum_x * suma_y)
            m_den = (n * sum_x2) - (sum_x**2)
            m_pendiente = m_num / m_den if m_den != 0 else 0
            b_interseccion = (suma_y - (m_pendiente * sum_x)) / n
            
            y_pred = m_pendiente * x_idx + b_interseccion
            ss_res = np.sum((y_vals - y_pred)**2)
            ss_tot = suma_desviaciones_cuad # Ya calculada arriba
            r_cuadrado = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # --- VISUALIZACIÓN ---
            col_g, col_t = st.columns([2, 1])
            with col_g:
                st.subheader("Gráfica de Tendencia")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(x_idx, y_vals, color='#2e7d32', s=100, label='Datos Reales', edgecolors='black')
                ax.plot(x_idx, y_pred, color='red', linewidth=2, label=f'y = {m_pendiente:.2f}x + {b_interseccion:.2f}')
                ax.set_facecolor('#ffffff')
                ax.grid(True, linestyle=':', alpha=0.5)
                plt.legend()
                st.pyplot(fig)
            with col_t:
                st.subheader("Datos Ingresados")
                st.dataframe(pd.DataFrame({"X": x_idx, "Y": y_vals}), hide_index=True)

            # --- TABLA ORDENADA ---
            st.divider()
            st.subheader("Tratamiento de Datos: Orden Ascendente")
            df_ord = pd.DataFrame({"Posición": range(1, n+1), "Valor": datos_ordenados})
            def resalta_mediana(row):
                target = (n+1)//2 if n%2!=0 else [n//2, n//2+1]
                is_med = row['Posición'] == target if n%2!=0 else row['Posición'] in target
                return ['background-color: #c8e6c9' if is_med else '' for _ in row]
            st.table(df_ord.style.apply(resalta_mediana, axis=1))

            # --- RESULTADOS MÉTRICOS ---
            st.divider()
            st.subheader("Reporte de Resultados")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("MEDIA (x̄)", f"{media:.2f}")
            m2.metric("MODA (Mo)", moda_res)
            m3.metric("MEDIANA (x̃)", f"{mediana:.2f}")
            m4.metric("VARIANZA (s²)", f"{varianza:.2f}")
            m5.metric("DESV. EST. (s)", f"{desviacion:.2f}")
            
            st.write(f"**Ecuación de la Recta:** y = {m_pendiente:.4f}x + {b_interseccion:.4f} | **Precisión (R²):** {r_cuadrado:.4f}")

            # --- PROCEDIMIENTO PASO A PASO ---
            st.divider()
            with st.expander("📝 CLIC PARA VER TODO EL PROCEDIMIENTO DETALLADO"):
                
                # Paso 1: Media
                st.markdown("#### 1. Cálculo de la Media ($\mu$)")
                st.latex(r"\bar{x} = \frac{\sum x_i}{n}")
                st.write(f"**Sustitución:** $\\bar{{x}} = \\frac{{{suma_y}}}{{{n}}}$")
                st.write(f"**Resolución:** Se divide la suma total entre el número de datos.")
                st.success(f"**Resultado:** {media:.4f}")

                # Paso 2: Mediana
                st.markdown("#### 2. Cálculo de la Mediana")
                if n % 2 != 0:
                    st.latex(r"\text{Posición} = \frac{n+1}{2}")
                    st.write(f"**Sustitución:** Posición = $\\frac{{{n}+1}}{{2}} = {(n+1)/2}$")
                else:
                    st.latex(r"\tilde{x} = \frac{x_{n/2} + x_{(n/2)+1}}{2}")
                    st.write(f"**Sustitución:** $\\frac{{{datos_ordenados[n//2-1]} + {datos_ordenados[n//2]}}}{{2}}$")
                st.success(f"**Resultado:** {mediana}")

                # Paso 3: Varianza
                st.markdown("#### 3. Cálculo de la Varianza Muestral ($s^2$)")
                st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}")
                st.write(f"**Sustitución:** $s^2 = \\frac{{{suma_desviaciones_cuad:.4f}}}{{{n}-1}}$")
                st.write(f"**Resolución:** Suma de cuadrados de desviaciones entre n-1.")
                st.success(f"**Resultado:** {varianza:.4f}")

                # Paso 4: Desviación Estándar
                st.markdown("#### 4. Cálculo de la Desviación Estándar ($s$)")
                st.latex(r"s = \sqrt{s^2}")
                st.write(f"**Sustitución:** $s = \\sqrt{{{varianza:.4f}}}$")
                st.success(f"**Resultado:** {desviacion:.4f}")

                # Paso 5: Recta de Regresión
                st.markdown("#### 5. Ecuación de la Recta ($y = mx + b$)")
                st.write("**Pendiente (m):**")
                st.latex(r"m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
                st.write(f"**Sustitución m:** $\\frac{{{n}({sum_xy}) - ({sum_x})({suma_y})}}{{{n}({sum_x2}) - ({sum_x})^2}}$")
                st.write(f"**Resultado m:** {m_pendiente:.4f}")
                
                st.write("**Intersección (b):**")
                st.latex(r"b = \bar{y} - m\bar{x}")
                st.write(f"**Sustitución b:** ${media:.2f} - ({m_pendiente:.2f} \\times {sum_x/n:.2f})$")
                st.success(f"**Ecuación Final:** $y = {m_pendiente:.4f}x + {b_interseccion:.4f}$")

                # Paso 6: R-Cuadrado
                st.markdown("#### 6. Coeficiente de Determinación ($R^2$)")
                st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
                st.write(f"**Sustitución:** $1 - \\frac{{{ss_res:.4f}}}{{{ss_tot:.4f}}}$")
                st.success(f"**Resultado R²:** {r_cuadrado:.4f}")

    except Exception as e:
        st.error(f"Error en el proceso: {e}")

