# ARCHIVO: APP_FRONT/utils.py
import streamlit as st

def cambiar_pagina(nueva_pagina: str):
    st.session_state['pagina'] = nueva_pagina

def generar_barra_navegacion():
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    col_home, col_indic, col_gravedad, col_clasif, col_pred = st.columns(5, gap="small")
    pagina_actual = st.session_state.get("pagina", "inicio")
    
    # Botones con keys con prefijo 'nav_'
    with col_home:
        st.button("🏠 INICIO", on_click=cambiar_pagina, args=("inicio",), key="nav_inicio")
    with col_indic:
        st.button("🛣️ Indicadores Generales", on_click=cambiar_pagina, args=("indicadores",), disabled=(pagina_actual == "indicadores"), key="nav_indicadores")
    with col_gravedad:
        st.button("🚑 Gravedad Accidente", on_click=cambiar_pagina, args=("gravedad",), disabled=(pagina_actual == "gravedad"), key="nav_gravedad")
    with col_clasif:
        st.button("🔬 Entrenamiento y Evaluación", on_click=cambiar_pagina, args=("clasificacion",), disabled=(pagina_actual == "clasificacion"), key="nav_clasificacion")
    with col_pred:
        st.button("🔮 Modelo Predictivo", on_click=cambiar_pagina, args=("predictivo",), disabled=(pagina_actual == "predictivo"), key="nav_predictivo")
    
    st.markdown("</div>", unsafe_allow_html=True)