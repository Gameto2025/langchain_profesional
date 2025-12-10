# ==========================================
# herramientas.py — Versión Profesional
# Contiene todas las herramientas del asistente
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool


# ---------------------------------------------------------
# 1️⃣ INFORMACIÓN GENERAL DEL DATAFRAME
# ---------------------------------------------------------
def informacion_df(pregunta: str, df: pd.DataFrame) -> str:
    """
    Genera un informe narrativo y técnico del DataFrame:
    - Dimensiones
    - Tipos de datos
    - Valores nulos
    - Filas duplicadas
    - Interpretación narrativa generada por LLM
    """

    n_filas, n_columnas = df.shape
    tipos = df.dtypes.astype(str)

    tabla_tipos = (
        pd.DataFrame({"Columna": tipos.index, "Tipo de dato": tipos.values})
        .to_markdown(index=False)
    )

    nulos = df.isnull().sum()
    pct_nulos = (nulos / len(df) * 100).round(2)

    tabla_nulos = (
        pd.DataFrame({
            "Columna": nulos.index,
            "Nulos": nulos.values,
            "% Nulos": pct_nulos.values
        })
        .to_markdown(index=False)
    )

    duplicados = df.duplicated().sum()

    # Prompt de narrativa más profesional
    plantilla = PromptTemplate(
        template="""
        Eres un analista de datos senior y debes generar una narrativa profesional.

        Datos del dataset:
        - Pregunta del usuario: {pregunta}
        - Filas: {n_filas}
        - Columnas: {n_columnas}

        Escribe un texto que incluya:
        1. Una explicación general del dataset
        2. Qué análisis son relevantes
        3. Recomendaciones de preprocesamiento
        4. Posibles insights

        No repitas tablas ni dimensiones.
        """,
        input_variables=["pregunta", "n_filas", "n_columnas"],
    )

    narrativa = (plantilla | StrOutputParser()).invoke({
        "pregunta": pregunta,
        "n_filas": n_filas,
        "n_columnas": n_columnas
    })

    return f"""
# 📊 Informe General del Dataset

## Resumen General
| Métrica | Valor |
|--------|-------|
| Filas | {n_filas} |
| Columnas | {n_columnas} |
| Duplicados | {duplicados} |

---

## Tipos de Datos
{tabla_tipos}

---

## Valores Nulos
{tabla_nulos}

---

## Narrativa Profesional
{narrativa}
"""


# ---------------------------------------------------------
# 2️⃣ RESUMEN ESTADÍSTICO
# ---------------------------------------------------------
def resumen_estadistico(pregunta: str, df: pd.DataFrame) -> str:
    """
    Genera un informe estadístico interpretado por el LLM:
    - describe()
    - explicación numérica
    - detección de outliers
    - conclusiones
    """
    resumen = df.describe(include="number").transpose().to_string()

    plantilla = PromptTemplate(
        template="""
        Eres un analista experto. Interpreta estas estadísticas:

        Pregunta: {pregunta}

        Estadísticas:
        {resumen}

        Escribe un informe que incluya:
        - Interpretación columna por columna
        - Señales de outliers
        - Variabilidad
        - Sugerencias de análisis posteriores
        """,
        input_variables=["pregunta", "resumen"],
    )

    cadena = plantilla | StrOutputParser()
    return cadena.invoke({"pregunta": pregunta, "resumen": resumen})


# ---------------------------------------------------------
# 3️⃣ GENERACIÓN DE GRÁFICOS
# ---------------------------------------------------------
def generar_grafico(pregunta: str, df: pd.DataFrame):
    """
    El LLM genera el código del gráfico.
    Se ejecuta automáticamente con exec()
    y se muestra en Streamlit desde app.py.
    """

    columnas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    num_filas = len(df)

    plantilla = PromptTemplate(
        template="""
        Eres un experto en visualización. Devuelve SOLO código Python.

        Solicitud del usuario:
        "{pregunta}"

        Dataset:
        Filas: {num_filas}
        Columnas:
        {columnas}

        Reglas:
        - Usar matplotlib y seaborn
        - sns.set_theme()
        - figsize=(10,5)
        - Título y etiquetas
        - plt.show()

        Código:
        """,
        input_variables=["pregunta", "columnas", "num_filas"],
    )

    script = (plantilla | StrOutputParser()).invoke({
        "pregunta": pregunta,
        "columnas": columnas_info,
        "num_filas": num_filas
    })

    # Limpiar ```python
    script = script.replace("```python", "").replace("```", "")

    exec_globals = {"df": df, "plt": plt, "sns": sns}
    exec(script, exec_globals)

    return ""


# ---------------------------------------------------------
# 4️⃣ EJECUCIÓN INTELIGENTE DE PYTHON (Correlaciones + REPL)
# ---------------------------------------------------------
def ejecutar_python_inteligente(pregunta: str, df: pd.DataFrame):
    """
    - Detecta preguntas sobre correlaciones y responde automáticamente.
    - Si no detecta correlación, ejecuta código Python directo (REPL seguro).
    """

    pregunta_lower = pregunta.lower()

    # Modo correlación inteligente
    if "correl" in pregunta_lower or "relación" in pregunta_lower:
        corr = df.corr(numeric_only=True)

        tabla = corr.to_markdown()
        return f"""
# 🔎 Análisis de correlación

{tabla}
"""

    # Si no es correlación, ejecuta código Python directamente
    repl = PythonAstREPLTool(locals={"df": df})
    return repl.run(pregunta)


# ---------------------------------------------------------
# 5️⃣ INFORME DE INSIGHTS
# ---------------------------------------------------------
def generar_insights(pregunta: str, df: pd.DataFrame) -> str:
    """
    Informe profesional con:
    - Patrones
    - Tendencias
    - Correlaciones importantes
    - Outliers
    - Recomendaciones
    """

    columnas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])

    plantilla = PromptTemplate(
        template="""
        Eres un analista senior. Responde a la pregunta: {pregunta}

        Columnas y tipos:
        {columnas}

        Genera un informe con:
        1. Patrones principales
        2. Tendencias
        3. Outliers relevantes
        4. Variables más importantes
        5. Recomendaciones accionables
        """,
        input_variables=["pregunta", "columnas"],
    )

    cadena = plantilla | StrOutputParser()
    return cadena.invoke({
        "pregunta": pregunta,
        "columnas": columnas_info
    })


# ---------------------------------------------------------
# 6️⃣ CREAR TODAS LAS HERRAMIENTAS (para LangChain)
# ---------------------------------------------------------
def crear_herramientas(df: pd.DataFrame):
    """
    Registra todas las herramientas como StructuredTool.
    Estas herramientas son llamadas desde app.py.
    """

    herramientas = [
        StructuredTool.from_function(
            name="Informaciones DF",
            func=lambda pregunta: informacion_df(pregunta, df),
            description="Devuelve un informe general del DataFrame.",
            return_direct=True
        ),
        StructuredTool.from_function(
            name="Resumen Estadístico",
            func=lambda pregunta: resumen_estadistico(pregunta, df),
            description="Genera un análisis estadístico completo.",
            return_direct=True
        ),
        StructuredTool.from_function(
            name="Generar Gráfico",
            func=lambda pregunta: generar_grafico(pregunta, df),
            description="Crea un gráfico basado en la solicitud del usuario.",
            return_direct=True
        ),
        StructuredTool.from_function(
            name="Herramienta Python",
            func=lambda pregunta: ejecutar_python_inteligente(pregunta, df),
            description="Ejecuta análisis Python inteligentes sobre el DataFrame.",
            return_direct=True
        ),
        StructuredTool.from_function(
            name="Informe de Insights",
            func=lambda pregunta: generar_insights(pregunta, df),
            description="Genera un informe narrativo de insights del dataset.",
            return_direct=True
        )
    ]

    return herramientas