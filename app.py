import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Proyecto Probabilidad y Estadística", layout="wide")

st.markdown("""
    <style>
    /* Fondo blanco total */
    .stApp { background-color: #ffffff; }
    
    /* Forzar visibilidad de textos y títulos */
    h1, h2, h3, h4, h5, h6, p, label, span, .stMetric { 
        color: #000000 !important; 
        font-family: 'Arial', sans-serif;
    }
    
    /* Estilo para los contenedores de pasos */
    .step-box {
        border-left: 5px solid #2e7d32;
        background-color: #f8f9fa;
        padding: 15px;
        margin: 10px 0px;
        border-radius: 5px;
    }

    /* Estilo para los expanders */
    .streamlit-expanderHeader {
        background-color: #e8f5e9 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Análisis Estadístico y Probabilístico")
st.markdown("---")

entrada = st.text_input("📝 Ingresa tus datos (ej: 10 20 20 30 40):")

if entrada:
    try:
        datos = [float(x) for x in entrada.replace(',', ' ').split()]
        n = len(datos)
        
        if n < 2:
            st.warning("⚠️ Se necesitan al menos 2 datos para el análisis completo.")
        else:
            datos_ordenados = sorted(datos)
            suma_total = sum(datos)
            media = suma_total / n
            
            frec_dict = {}
            for d in datos: frec_dict[d] = frec_dict.get(d, 0) + 1
            max_frec = max(frec_dict.values())
            if max_frec > 1:
                modas = [k for k, v in frec_dict.items() if v == max_frec]
                moda_res = ", ".join(map(str, modas))
            else:
                moda_res = "Amodal (Sin moda)"
            
            if n % 2 != 0:
                mediana = datos_ordenados[n // 2]
            else:
                m1, m2 = datos_ordenados[n // 2 - 1], datos_ordenados[n // 2]
                mediana = (m1 + m2) / 2
            
            suma_desviaciones_cuad = sum([(x - media)**2 for x in datos])
            varianza = suma_desviaciones_cuad / (n - 1)
            desviacion = np.sqrt(varianza)

            c_graf, c_tab = st.columns([2, 1])
            with c_graf:
                st.subheader("Tendencia Lineal")
                x_vals = np.arange(n)
                m_r, b_r = np.polyfit(x_vals, datos, 1)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(x_vals, datos, color='#2e7d32', s=100, label='Muestra')
                ax.plot(x_vals, m_r * x_vals + b_r, color='red', label='Tendencia')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            with c_tab:
                st.subheader("Datos en Crudo")
                st.dataframe(pd.DataFrame({"X": datos}), use_container_width=True)

            st.markdown("### 🔢 Datos Ordenados de Menor a Mayor")
            df_ord = pd.DataFrame({"Posición": range(1, n+1), "Valor": datos_ordenados})
            def highlight(row):
                is_med = (n%2!=0 and row['Posición']==(n+1)//2) or (n%2==0 and row['Posición'] in [n//2, n//2+1])
                return ['background-color: #c8e6c9' if is_med else '' for _ in row]
            st.table(df_ord.style.apply(highlight, axis=1))

            st.markdown("### 📈 Resumen de Resultados")
            met1, met2, met3, met4, met5 = st.columns(5)
            met1.metric("Media (x̄)", f"{media:.2f}")
            met2.metric("Moda (Mo)", moda_res)
            met3.metric("Mediana (x̃)", f"{mediana:.2f}")
            met4.metric("Varianza (s²)", f"{varianza:.2f}")
            met5.metric("Desv. Est. (s)", f"{desviacion:.2f}")

            st.markdown("---")
            with st.expander("📝 CLIC AQUÍ: PROCEDIMIENTO MATEMÁTICO DETALLADO"):
                
                # 1. MEDIA
                st.markdown("#### **1. Cálculo de la Media Aritmética**")
                st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")
                st.markdown(f"**Sustitución:** $\\bar{{x}} = \\frac{{{suma_total}}}{{{n}}}$")
                st.markdown(f"**Operación:** Se suman los {n} elementos y el resultado se divide entre {n}.")
                st.info(f"**Resultado:** {media:.4f}")

                # 2. MEDIANA
                st.markdown("#### **2. Cálculo de la Mediana**")
                st.write(f"Datos ordenados: {datos_ordenados}")
                if n % 2 != 0:
                    st.latex(r"\tilde{x} = x_{\left(\frac{n+1}{2}\right)}")
                    st.markdown(f"**Sustitución:** Posición = $\\frac{{{n}+1}}{{2}} = {(n+1)//2}$")
                else:
                    st.latex(r"\tilde{x} = \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}")
                    st.markdown(f"**Sustitución:** $\\frac{{{m1} + {m2}}}{{2}}$")
                st.info(f"**Resultado:** {mediana}")

                # 3. VARIANZA
                st.markdown("#### **3. Cálculo de la Varianza Muestral**")
                st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}")
                st.markdown(f"**Sustitución:** $s^2 = \\frac{{{suma_desviaciones_cuad:.4f}}}{{{n}-1}}$")
                st.markdown(f"**Operación:** Suma de cuadrados de desviaciones dividido entre los grados de libertad ({n-1}).")
                st.info(f"**Resultado:** {varianza:.4f}")

                # 4. DESVIACIÓN
                st.markdown("#### **4. Cálculo de la Desviación Estándar**")
                st.latex(r"s = \sqrt{s^2}")
                st.markdown(f"**Sustitución:** $s = \\sqrt{{{varianza:.4f}}}$")
                st.markdown(f"**Operación:** Se extrae la raíz cuadrada positiva de la varianza calculada arriba.")
                st.info(f"**Resultado:** {desviacion:.4f}")

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
