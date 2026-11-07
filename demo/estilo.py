"""
Estilo visual de la demo: los colores y la tipografía de la presentación del proyecto.

Solo cambia la apariencia. Los colores base y las tipografías están en .streamlit/config.toml; acá va
lo que el tema de Streamlit no cubre: los degradados, las tarjetas de métricas y las etiquetas de sección.
Los selectores apuntan a contenedores con `key` (clase .st-key-<key>) y a los data-testid de
Streamlit 1.67.0, la versión fijada en demo/requirements.txt.
"""

# Paleta de la presentación
AZUL_MARINO = "#0B2D72"
AZUL = "#1A4FD6"
CELESTE = "#33C6EE"
VERDE_AGUA = "#1AA99B"
TEXTO = "#102449"
TEXTO_SUAVE = "#4B5A78"
CLARO = "#FAFBFD"

CSS = f"""
<style>
/* --- Portada: degradado y barras de la primera diapositiva --- */
.st-key-portada {{
    position: relative; overflow: hidden; container-type: inline-size;   /* el título se mide con este ancho */
    background: linear-gradient(135deg, {AZUL_MARINO} 0%, {AZUL} 70%, #2A6AF0 100%);
    padding: 2.2rem 24% 2.2rem 2.5rem;   /* a la derecha queda lugar para las barras */
}}
.st-key-portada::after {{
    content: ""; position: absolute; right: 4%; bottom: 0; width: 15%; height: 80%;
    pointer-events: none; opacity: 0.85;
    background:
        linear-gradient({CELESTE}, {CELESTE}) left bottom / 24% 55% no-repeat,
        linear-gradient({VERDE_AGUA}, {VERDE_AGUA}) center bottom / 24% 75% no-repeat,
        linear-gradient(rgba(250,251,253,.22), rgba(250,251,253,.22)) right bottom / 24% 100% no-repeat;
}}
.st-key-portada h1 {{
    color: {CLARO}; font-weight: 800; text-transform: uppercase; line-height: 1.08;
    font-size: clamp(1.4rem, 5.5cqi, 3rem); padding: 0;
    word-break: normal; overflow-wrap: normal; hyphens: none;   /* no cortar palabras al medio */
}}
.st-key-portada p {{ color: #D3E1FF; font-size: 1.1rem; }}
.st-key-portada p.volanta {{
    color: {CELESTE}; font-family: Montserrat, sans-serif; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase; font-size: 0.9rem;
}}
.st-key-portada [data-testid="stCaptionContainer"] p {{
    color: {CLARO}; font-family: Montserrat, sans-serif; font-weight: 700; font-size: 1rem;
}}
@media (max-width: 700px) {{
    .st-key-portada {{ padding: 1.6rem 1.4rem; }}
    .st-key-portada::after {{ display: none; }}
}}

/* --- Barra lateral: degradado del panel izquierdo de las diapositivas --- */
[data-testid="stSidebar"] {{ background: linear-gradient(165deg, #1F5BE3 0%, {AZUL_MARINO} 100%); }}
[data-testid="stSidebar"] h2 {{ text-transform: uppercase; font-weight: 800; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: #9FC3FF; }}
[data-testid="stSidebar"] button[kind="primary"] {{
    background: {CLARO}; color: {AZUL_MARINO}; border: none;
    font-family: Montserrat, sans-serif; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;
}}
[data-testid="stSidebar"] button[kind="primary"]:hover {{ background: {CELESTE}; color: {AZUL_MARINO}; }}

/* --- Títulos de sección: la etiqueta azul de las diapositivas --- */
[data-testid="stMain"] h3 {{ color: {AZUL_MARINO}; font-weight: 800; }}
[data-testid="stMain"] h4 {{
    display: inline-block; background: {AZUL}; color: {CLARO};
    font-weight: 800; text-transform: uppercase; letter-spacing: 1px;
    font-size: 0.95rem; padding: 0.4rem 0.9rem; margin-top: 1.2rem;
}}
[data-testid="stMain"] [data-testid="stHeaderActionElements"] {{ display: none; }}   /* sin ícono de enlace en los títulos */

/* --- Tarjetas de métricas, con el borde celeste / verde agua de los acentos --- */
.st-key-metricas [data-testid="stMetric"] {{
    background: white; border-top: 5px solid {CELESTE};
    padding: 0.9rem 1.1rem; box-shadow: 0 1px 4px rgba(16, 36, 73, 0.08);
}}
.st-key-metricas [data-testid="stColumn"]:nth-child(even) [data-testid="stMetric"] {{ border-top-color: {VERDE_AGUA}; }}
.st-key-metricas [data-testid="stMetricLabel"] p {{ color: {TEXTO_SUAVE}; font-weight: 600; }}
.st-key-metricas [data-testid="stMetricValue"] {{
    font-family: Montserrat, sans-serif; font-weight: 800; color: {AZUL_MARINO};
}}

/* --- Pie: la línea celeste → verde agua de las diapositivas --- */
.st-key-pie {{
    margin-top: 2rem; padding-top: 0.8rem;
    border-top: 6px solid; border-image: linear-gradient(90deg, {CELESTE} 0%, {VERDE_AGUA} 100%) 1;
}}
</style>
"""


def configurar_grafico(grafico):
    """Tipografías y colores de texto de la presentación en los gráficos de Altair."""
    return (grafico
            .configure_axis(labelFont="Public Sans", titleFont="Public Sans", labelColor=TEXTO_SUAVE,
                            titleColor=TEXTO, labelFontSize=12, titleFontSize=12, gridColor="#DDE4F0")
            .configure_legend(labelFont="Public Sans", labelColor=TEXTO, labelFontSize=13, labelLimit=0)   # sin recortar
            .configure_view(strokeWidth=0))
