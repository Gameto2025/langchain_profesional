# ==========================================
# app.py — Plantilla Profesional Streamlit
# Con soporte para LLM + Tools + archivo .env
# ==========================================

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from herramientas import crear_herramientas

# --------------------------------------------
# CARGAR VARIABLES DE ENTORNO DESDE .env
# --------------------------------------------
load_dotenv()  # Carga archivo .env automáticamente

# --------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------
st.set_page_config(page_title="Asistente de Análisis de Datos con IA", layout="centered")
st.title("🦜 Asistente de Análisis de Datos con IA")

st.info("""
Esta herramienta permite generar reportes, responder preguntas sobre los datos 
y crear gráficos usando un DataFrame cargado desde un archivo CSV.
""")

# --------------------------------------------
# SUBIR ARCHIVO
# --------------------------------------------
st.markdown("### 📁 Cargar archivo CSV")
archivo_cargado = st.file_uploader("Selecciona un archivo CSV", type="csv", label_visibility="collapsed")

if archivo_cargado:
    df = pd.read_csv(archivo_cargado)
    st.success("Archivo cargado exitosamente!")
    st.dataframe(df.head())

    # --------------------------------------------
    # CARGAR LLM DESDE GROQ
    # --------------------------------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # lee clave desde .env

    if GROQ_API_KEY is None:
        st.error("❌ ERROR: No se encontró la variable GROQ_API_KEY en el archivo .env")
        st.stop()

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # --------------------------------------------
    # CARGAR HERRAMIENTAS
    # --------------------------------------------
    tools = crear_herramientas(df)
    tool_dict = {t.name: t for t in tools}

    # --------------------------------------------
    # FUNCIÓN SIMPLE PARA PREGUNTAS AL LLM
    # --------------------------------------------
    def preguntar_llm(mensaje):
        respuesta = llm.invoke(mensaje)
        return respuesta.content

    # --------------------------------------------
    # 1️⃣ INFORME GENERAL
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📄 Informe General del Dataset")

    if st.button("Generar Informe General"):
        with st.spinner("Generando informe general…"):
            tool = tool_dict.get("Informaciones DF")
            if tool:
                informe = tool.run({"pregunta": "Reporte general del DataFrame"})
                st.markdown(informe)

                st.download_button(
                    "📥 Descargar Informe General",
                    informe,
                    "informe_general.md"
                )

    # --------------------------------------------
    # 2️⃣ RESUMEN ESTADÍSTICO
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Informe Estadístico")

    if st.button("Generar Informe Estadístico"):
        with st.spinner("Generando análisis estadístico…"):
            tool = tool_dict.get("Resumen Estadístico")
            if tool:
                informe = tool.run({"pregunta": "Resumen estadístico del DataFrame"})
                st.markdown(informe)

                st.download_button(
                    "📥 Descargar Informe Estadístico",
                    informe,
                    "informe_estadistico.md"
                )

    # --------------------------------------------
    # 3️⃣ GENERAR GRÁFICO
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Crear gráfico")

    pregunta_grafico = st.text_input("Describe el gráfico que deseas generar:")

    if st.button("Generar gráfico"):
        if pregunta_grafico.strip() == "":
            st.warning("Por favor, describe el gráfico que deseas generar.")
        else:
            with st.spinner("Generando gráfico…"):
                tool = tool_dict.get("Generar Gráfico")
                if tool:
                    tool.run({"pregunta": pregunta_grafico})

    # --------------------------------------------
    # 4️⃣ INFORME DE INSIGHTS
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## ✨ Informe de Insights del Dataset")

    if st.button("Generar Informe de Insights"):
        with st.spinner("Generando insights…"):
            tool = tool_dict.get("Informe de Insights")
            if tool:
                insights = tool.run({"pregunta": "Genera un informe con los principales insights"})
                st.markdown(insights)

                st.download_button(
                    "📥 Descargar Informe de Insights",
                    insights,
                    "informe_insights.md"
                )

    # --------------------------------------------
    # 5️⃣ PREGUNTA DIRECTA AL LLM
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 🔎 Preguntas directas sobre los datos")

    pregunta = st.text_input("Escribe tu pregunta:")

    if st.button("Responder pregunta"):
        if pregunta.strip() == "":
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Analizando…"):
                respuesta = preguntar_llm(
                    f"""Eres un analista experto. 
                    Responde la siguiente pregunta usando este DataFrame:
                    Columnas: {list(df.columns)}
                    Pregunta: {pregunta}"""
                )
                st.markdown(respuesta)