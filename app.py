import os

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, json, os
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Configuração da página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSpace Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS customizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0d2137 0%,#0a3d1f 100%); }
    [data-testid="stSidebar"] * { color: #e8f5e9 !important; }
    .metric-card { background:#1a2744; border-radius:12px; padding:18px;
                   border-left:4px solid #40c463; margin:8px 0; }
    .metric-card h3 { color:#40c463; font-size:1.8em; margin:0; }
    .metric-card p  { color:#aaa; margin:0; font-size:0.85em; }
    .section-title  { color:#40c463; border-bottom:2px solid #40c463;
                      padding-bottom:6px; margin-bottom:16px; }
    .chat-user { background:#1e3a5f; border-radius:12px; padding:12px 16px;
                 margin:8px 0; border-left:3px solid #1f6feb; }
    .chat-bot  { background:#1a2f1a; border-radius:12px; padding:12px 16px;
                 margin:8px 0; border-left:3px solid #40c463; }
    .stButton>button { background:#40c463; color:#000; border:none;
                       border-radius:8px; font-weight:bold; }
    .stButton>button:hover { background:#2ea84d; color:#fff; }
    div[data-testid="metric-container"] { background:#1a2744; border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ── Inicializa RAG (cache para não recarregar) ─────────────────────────────
@st.cache_resource
def init_rag():
    DOCUMENTS = [
        {"titulo": "Sensoriamento Remoto na Agricultura de Precisão", "fonte": "Módulo 1",
         "texto": "O sensoriamento remoto por satélite revolucionou a agricultura de precisão. Satélites como Sentinel-2 (ESA) e Landsat-9 (NASA) capturam imagens multiespectrais que permitem calcular índices vegetativos como o NDVI (Normalized Difference Vegetation Index). O NDVI mede a saúde da vegetação comparando reflectância no infravermelho próximo (NIR) com o vermelho visível: NDVI = (NIR - RED) / (NIR + RED). Valores próximos de 1,0 indicam vegetação densa e saudável; valores abaixo de 0,2 indicam solo exposto ou vegetação estressada."},
        {"titulo": "IoT e Conectividade Satelital no Campo", "fonte": "Módulo 2",
         "texto": "A Internet das Coisas (IoT) agrícola integra sensores de solo, clima e equipamentos conectados a redes satelitais de baixa órbita (LEO). Constelações como Starlink (SpaceX), OneWeb e Telesat Lightspeed oferecem latência abaixo de 50ms e cobertura global. O protocolo LoRaWAN é amplamente adotado para comunicação de longa distância com baixo consumo energético em sensores distribuídos no campo."},
        {"titulo": "Inteligência Artificial e Machine Learning na Agricultura", "fonte": "Módulo 3",
         "texto": "Modelos de Machine Learning aplicados à agricultura espacial incluem: CNN para classificação de culturas, LSTM para previsão de séries temporais, Random Forest e XGBoost para estimativa de produtividade, YOLO e Faster R-CNN para identificação de pragas em imagens de drone. O Google Earth Engine permite processar petabytes de imagens satelitais na nuvem."},
        {"titulo": "Drones e Veículos Aéreos Não Tripulados (VANT)", "fonte": "Módulo 4",
         "texto": "Drones agrícolas operam em baixa altitude (30–120m) com resolução centimétrica. Modelos populares: DJI Agras T40 (pulverização) e senseFly eBee (mapeamento). Geram ortomosaicos, modelos digitais de elevação, mapas NDVI e nuvens de pontos 3D. Reduzem uso de agroquímicos em até 30-40%."},
        {"titulo": "Economia Espacial e o Agronegócio Brasileiro", "fonte": "Módulo 5",
         "texto": "A nova economia espacial movimentou US$ 546 bilhões em 2023, com projeções de US$ 1 trilhão até 2030. O Brasil possui o Centro de Lançamento de Alcântara (CLA) e opera o satélite CBERS. Startups agtech captaram mais de R$ 2,5 bilhões entre 2020 e 2024."},
        {"titulo": "Irrigação Inteligente Baseada em Dados Satelitais", "fonte": "Módulo 6",
         "texto": "A irrigação responde por 70% do consumo global de água doce. A irrigação inteligente reduz o desperdício em até 50%. O índice ET calculado por dados de satélite permite estimar a demanda hídrica real. Sistemas IoT ajustam automaticamente a lâmina d'água com base em umidade do solo, previsão meteorológica e NDWI via satélite."},
        {"titulo": "Monitoramento de Desmatamento e Rastreabilidade", "fonte": "Módulo 7",
         "texto": "O Brasil monitora o desmatamento com PRODES, DETER e MapBiomas. A UE exige certificação de não desmatamento (EUDR) para soja, carne bovina e cacau. O CAR integrado a dados satelitais permite fiscalização automatizada do Código Florestal."},
        {"titulo": "Modelos de Previsão de Safra", "fonte": "Módulo 8",
         "texto": "Modelos de previsão integram dados satelitais, climáticos e agronômicos. O DSSAT simula crescimento de culturas. A CONAB usa MODIS/NDVI para estimativas mensais. IA supervisionada atinge acurácia acima de 90% na previsão de soja e milho com 2-3 meses de antecedência."},
        {"titulo": "Blockchain e Contratos Inteligentes no Agro", "fonte": "Módulo 9",
         "texto": "Blockchain com dados satelitais cria rastreabilidade imutável na cadeia agroalimentar. Smart contracts automatizam pagamentos por créditos de carbono com verificação via satélite. O crédito de carbono voluntário foi negociado a US$ 8-15 por tonelada de CO2 em 2023."},
        {"titulo": "Geotecnologias e SIG", "fonte": "Módulo 10",
         "texto": "SIG integram camadas espaciais para tomada de decisão agrícola. QGIS e ArcGIS permitem cruzamento de dados de solo, clima e sensoriamento remoto. O ZARC define janelas de plantio de baixo risco para 40+ culturas. GNSS de alta precisão (RTK, PPP) é base para autopiloto de maquinários agrícolas."}
    ]
    docs = [Document(page_content=d["texto"], metadata={"titulo": d["titulo"], "fonte": d["fonte"]}) for d in DOCUMENTS]
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(chunks, emb)
    llm = InferenceClient(provider="novita", api_key=os.environ.get("HF_TOKEN", ""))
    return vs, llm, DOCUMENTS

def gerar_resposta_rag(pergunta, vectorstore, client):
    docs = vectorstore.similarity_search(pergunta, k=3)
    contexto = "\\n\\n".join([f"[{d.metadata['titulo']}]\\n{d.page_content}" for d in docs])
    fontes = list({d.metadata["titulo"] for d in docs})
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "Você é um assistente especialista em agricultura inteligente e nova economia espacial. Responda em português, de forma clara, com base APENAS no contexto fornecido."},
            {"role": "user", "content": f"CONTEXTO:\\n{contexto}\\n\\nPERGUNTA: {pergunta}"}
        ],
        max_tokens=512, temperature=0.3
    )
    return resp.choices[0].message.content.strip(), fontes

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0 6px 0;'>
  <div style='font-size:2.2em;'>🛰️</div>
  <div style='font-size:1.3em; font-weight:bold; color:#40c463;'>AgroSpace</div>
  <div style='font-size:0.8em; color:#a5d6a7;'>Nova Economia Espacial</div>
  <div style='font-size:0.75em; color:#81c784;'>Agricultura Inteligente</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("---")

menu = st.sidebar.radio("📋 Menu", [
    "🏠 Home",
    "📡 Sensoriamento Remoto",
    "🤖 IA & Machine Learning",
    "🌊 Irrigação Inteligente",
    "🚁 Drones & VANTs",
    "🌳 Sustentabilidade & Carbono",
    "🇧🇷 Economia Espacial Brasil",
    "🖼️ Análise de Imagem NDVI",
    "💬 Chat AgroSpace"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Fontes de Pesquisa:**")
st.sidebar.markdown("""
<small>
🔬 <a href='https://earthdata.nasa.gov' target='_blank'>NASA — MODIS, Landsat</a><br>
🛰️ <a href='https://www.copernicus.eu' target='_blank'>ESA Copernicus — Sentinel-2</a><br>
🌱 <a href='https://www.embrapa.br' target='_blank'>Embrapa — IRRIGAR, ZARC</a><br>
🌳 <a href='https://www.inpe.br' target='_blank'>INPE — PRODES, DETER</a><br>
🌍 <a href='https://cropmonitor.org' target='_blank'>FAO — CROP MONITOR</a><br>
🌊 <a href='https://www.noaa.gov' target='_blank'>NOAA — El Niño/ENSO</a><br>
💻 <a href='https://github.com/satellite-image-deep-learning' target='_blank'>GitHub — Satellite ML</a><br>
📊 <a href='https://www.kaggle.com/competitions/csiro-biomass' target='_blank'>Kaggle — CSIRO Biomass</a><br>
🤗 <a href='https://huggingface.co' target='_blank'>HuggingFace — Llama/MiniLM</a><br>
🌐 <a href='https://earthengine.google.com' target='_blank'>Google Earth Engine</a><br>
🗺️ <a href='https://terrabrasilis.dpi.inpe.br' target='_blank'>TerraBrasilis — INPE</a><br>
🧑‍🔬 <a href='https://colab.research.google.com/drive/1jciJ0UGDUnWwZDPB7aRTKZjlq6EbQQKb?usp=sharing' target='_blank'>RAG Agricultura Espacial — Geandro Dezordi</a>
</small>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**🛰️ Satélites para Agricultura:**")
st.sidebar.markdown("""
<small>
🌍 <a href='https://earth.google.com/web/' target='_blank'>Google Earth Web</a><br>
🗺️ <a href='https://maps.google.com/' target='_blank'>Google Maps — Satélite</a><br>
🔭 <a href='https://worldview.earthdata.nasa.gov/' target='_blank'>NASA Worldview</a><br>
🌀 <a href='https://zoom.earth/' target='_blank'>Zoom Earth</a><br>
📡 <a href='https://apps.sentinel-hub.com/eo-browser/' target='_blank'>Sentinel Hub EO Browser</a><br>
🇪🇺 <a href='https://browser.dataspace.copernicus.eu/' target='_blank'>Copernicus Browser</a>
</small>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tecnologias:**")
st.sidebar.markdown("🔹 Streamlit · Plotly")
st.sidebar.markdown("🔹 LangChain · FAISS")
st.sidebar.markdown("🔹 HuggingFace · Llama")
st.sidebar.markdown("🔹 NumPy · PIL")

# ══════════════════════════════════════════════════════════════════
# 🏠 HOME
# ══════════════════════════════════════════════════════════════════
if menu == "🏠 Home":
    st.title("🌾🛰️ AgroSpace Dashboard")
    st.markdown("### Nova Economia Espacial & Agricultura Inteligente")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Mercado Espacial 2023", "US$ 546 bi", "+15% a.a.")
    with col2:
        st.metric("🌾 Projeção 2030", "US$ 1 trilhão", "+83%")
    with col3:
        st.metric("🇧🇷 Agtech Brasil", "R$ 2,5 bi", "2020–2024")
    with col4:
        st.metric("💧 Economia de Água", "até 50%", "Irrigação inteligente")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Crescimento do Mercado Espacial")
        anos = [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]
        valores = [330,350,385,415,440,447,470,499,546,600,660,730,800,875,935,1000]
        df_mkt = pd.DataFrame({"Ano": anos, "US$ Bilhões": valores})
        fig = px.area(df_mkt, x="Ano", y="US$ Bilhões",
                      color_discrete_sequence=["#40c463"],
                      template="plotly_dark")
        fig.add_vline(x=2026, line_dash="dash", line_color="#e3b341",
                      annotation_text="Hoje", annotation_position="top right")
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🥧 Distribuição por Setor")
        setores = ["Satélites Comerciais","Lançamento","Serviços de Solo","Observação da Terra","Agtech Espacial"]
        valores_s = [38, 22, 18, 14, 8]
        fig2 = px.pie(values=valores_s, names=setores, hole=0.4,
                      color_discrete_sequence=px.colors.sequential.Greens_r,
                      template="plotly_dark")
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🗺️ Módulos deste Dashboard")
    modulos = [
        ("📡","Sensoriamento Remoto","NDVI, Sentinel-2, Landsat, índices vegetativos"),
        ("🤖","IA & Machine Learning","CNN, LSTM, Random Forest, previsão de safra"),
        ("🌊","Irrigação Inteligente","ET, NDWI, IoT, redução de 50% no consumo hídrico"),
        ("🚁","Drones & VANTs","DJI Agras, senseFly eBee, fotogrametria, Pix4D"),
        ("🌳","Sustentabilidade","Créditos de carbono, PRODES, EUDR, MapBiomas"),
        ("🇧🇷","Economia Brasil","CLA Alcântara, CBERS, R$ 2,5 bi em agtech"),
        ("🖼️","Análise NDVI","Upload de imagem e diagnóstico agronômico automático"),
        ("💬","Chat AgroSpace","Assistente IA com base nos documentos do projeto"),
    ]
    cols = st.columns(4)
    for i, (icon, titulo, desc) in enumerate(modulos):
        with cols[i % 4]:
            st.markdown(f"""
            <div style='background:#1a2744;border-radius:10px;padding:14px;margin:6px 0;border-left:3px solid #40c463'>
            <b style='color:#40c463'>{icon} {titulo}</b><br>
            <small style='color:#aaa'>{desc}</small></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 📡 SENSORIAMENTO REMOTO
# ══════════════════════════════════════════════════════════════════
elif menu == "📡 Sensoriamento Remoto":
    st.title("📡 Sensoriamento Remoto na Agricultura")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🛰️ Satélites Ativos", "800+", "2024")
    with col2: st.metric("📏 Resolução Sentinel-2", "10 metros", "por pixel")
    with col3: st.metric("🔄 Revisita", "5 dias", "Sentinel-2")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📊 Índices Vegetativos — Escala de Valores")
        indices = ["NDVI","EVI","NDWI","NDRE","SAVI","GNDVI"]
        v_min   = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
        v_max   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        v_ideal = [0.7, 0.6, 0.3, 0.5, 0.6, 0.65]
        df_idx = pd.DataFrame({"Índice":indices,"Valor Ideal":v_ideal})
        fig = px.bar(df_idx, x="Valor Ideal", y="Índice", orientation="h",
                     color="Valor Ideal", color_continuous_scale="Greens",
                     template="plotly_dark", title="Valor Ideal por Índice")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cultura_sel = st.radio("Selecione a cultura:", ["🌱 Soja", "🎋 Cana-de-açúcar"], horizontal=True)
        semanas = list(range(1, 21))
        np.random.seed(10)
        if cultura_sel == "🌱 Soja":
            ndvi_serie = np.concatenate([
                np.linspace(0.1, 0.85, 10) + np.random.normal(0, 0.02, 10),
                np.linspace(0.85, 0.2, 10) + np.random.normal(0, 0.02, 10)
            ])
            titulo_serie = "Evolução NDVI — Soja (20 semanas)"
            cor_serie    = "#40c463"
            desc_serie   = "Soja: crescimento rápido nas primeiras 10 semanas (NDVI até 0,85), seguido de senescência na colheita."
        else:
            ndvi_serie = np.concatenate([
                np.linspace(0.05, 0.45, 5) + np.random.normal(0, 0.015, 5),
                np.linspace(0.45, 0.80, 8) + np.random.normal(0, 0.02, 8),
                np.linspace(0.80, 0.55, 7) + np.random.normal(0, 0.02, 7)
            ])
            titulo_serie = "Evolução NDVI — Cana-de-açúcar (20 semanas)"
            cor_serie    = "#f4c430"
            desc_serie   = "Cana: crescimento lento inicial, pico de biomassa entre semanas 13-15 (NDVI ~0,80) e leve queda pré-colheita."

        df_ndvi = pd.DataFrame({"Semana": semanas, "NDVI": ndvi_serie})
        fig2 = px.line(df_ndvi, x="Semana", y="NDVI", markers=True,
                       color_discrete_sequence=[cor_serie],
                       template="plotly_dark", title=titulo_serie)
        fig2.add_hline(y=0.5, line_dash="dash", line_color="yellow",
                       annotation_text="Limiar saudável")
        fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=270)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(desc_serie)

    # Expander de explicação dos índices — abre ao clicar
    with st.expander("📖 Clique aqui para entender cada índice vegetativo do gráfico"):
        st.markdown("""
| Índice | Nome completo | O que mede | Valor ideal | Uso principal |
|--------|--------------|------------|-------------|---------------|
| **NDVI** | Normalized Difference Vegetation Index | Saúde geral da vegetação | ≥ 0,7 | Monitoramento de culturas, detecção de estresse |
| **EVI** | Enhanced Vegetation Index | Vigor vegetativo em dosséis densos | ≥ 0,6 | Cerrado, Amazônia — menos saturação que NDVI |
| **NDWI** | Normalized Difference Water Index | Conteúdo de água na vegetação | ≥ 0,3 | Estresse hídrico, irrigação de precisão |
| **NDRE** | Normalized Difference Red Edge | Teor de clorofila e nitrogênio | ≥ 0,5 | Fertilidade, deficiência nutricional precoce |
| **SAVI** | Soil Adjusted Vegetation Index | Vegetação em áreas com solo exposto | ≥ 0,6 | Regiões áridas, início do ciclo da cultura |
| **GNDVI** | Green Normalized Difference Vegetation Index | Atividade fotossintética (clorofila) | ≥ 0,65 | Maturidade da cultura, estimativa de produtividade |

**Como interpretar:** valores próximos de 1,0 indicam vegetação densa e saudável;
valores abaixo de 0,2 indicam solo exposto ou vegetação morta.
Fórmula base do NDVI: `(NIR - RED) / (NIR + RED)` — proposta por Rouse et al. (1974).
        """)

    st.markdown("#### 🛰️ Satélites de Observação Agrícola")
    df_sat = pd.DataFrame({
        "Satélite": ["Sentinel-2A/B","Landsat-9","CBERS-4A","Planet Dove","MODIS (Terra/Aqua)"],
        "Agência":  ["ESA","NASA/USGS","INPE/CAST","Planet Labs","NASA"],
        "Resolução":["10m","30m","8m","3m","250m"],
        "Revisita": ["5 dias","16 dias","5 dias","Diário","1–2 dias"],
        "Uso Agro": ["NDVI, EVI, NDWI","Histórico longo","Monitoramento BR","Alta frequência","Escala global"]
    })
    st.dataframe(df_sat, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# 🤖 IA & MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════
elif menu == "🤖 IA & Machine Learning":
    st.title("🤖 IA & Machine Learning na Agricultura")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🎯 Acurácia Previsão Safra", "90%+", "Redes neurais")
    with col2: st.metric("⏱️ Antecedência", "2–3 meses", "Antes da colheita")
    with col3: st.metric("📉 Redução Defensivos", "30–40%", "Com drones + IA")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🧠 Modelos de ML por Aplicação")
        modelos = ["CNN","LSTM","Random Forest","XGBoost","YOLO","U-Net","GPR"]
        acuracia = [92, 88, 91, 93, 89, 94, 87]
        aplicacao= ["Classif. Culturas","Previsão Temporal","Produtividade",
                    "Produtividade","Detecção Pragas","Segmentação","Previsão Safra"]
        df_ml = pd.DataFrame({"Modelo":modelos,"Acurácia (%)":acuracia,"Aplicação":aplicacao})
        fig = px.bar(df_ml, x="Modelo", y="Acurácia (%)", color="Acurácia (%)",
                     hover_data=["Aplicação"], color_continuous_scale="Greens",
                     template="plotly_dark", title="Acurácia por Modelo de IA")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 📊 Previsão de Produtividade — Soja (ton/ha)")
        anos = list(range(2018, 2031))
        real = [3.1,3.3,3.0,3.5,3.4,3.7,3.8,None,None,None,None,None,None]
        previsto = [None,None,None,None,None,3.7,3.9,4.1,4.3,4.4,4.6,4.8,5.0]
        df_prod = pd.DataFrame({"Ano":anos,"Real":real,"Previsto (IA)":previsto})
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=anos, y=real, name="Real", line=dict(color="#40c463",width=2)))
        fig2.add_trace(go.Scatter(x=anos, y=previsto, name="Previsto (IA)",
                                  line=dict(color="#1f6feb",width=2,dash="dot")))
        fig2.update_layout(template="plotly_dark", height=300,
                           margin=dict(l=0,r=0,t=10,b=0),
                           title="Produtividade Soja — Real vs Previsto por IA")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔬 Pipeline de IA Agrícola — clique em uma etapa para saber mais")

    PIPELINE_INFO = {
        "🛰️ Imagem Satélite/Drone": {
            "desc": "Coleta de dados brutos via sensores multiespectrais.",
            "numeros": "Sentinel-2 gera ~1 TB/dia. Planet Labs captura 3 milhões de km²/dia.",
            "conclusao": "Investir em acesso a dados satelitais é o ponto de partida. O custo de 1 cena Sentinel-2 caiu de US$ 20 (2010) para zero (2024) — tornando a análise acessível a qualquer produtor."
        },
        "⚙️ Pré-processamento": {
            "desc": "Correção atmosférica, remoção de nuvens, normalização de bandas.",
            "numeros": "Reduz ruído em até 40%. Sem pré-processamento, acurácia dos modelos cai de 90% para 60%.",
            "conclusao": "Vale cada centavo: sem pré-processamento correto, qualquer modelo de IA produz resultados incorretos. É a etapa mais crítica e muitas vezes mais negligenciada do pipeline."
        },
        "📐 Extração de Features": {
            "desc": "Cálculo de NDVI, EVI, NDWI e outras variáveis derivadas das imagens.",
            "numeros": "Um único pixel Sentinel-2 gera até 13 bandas e 20+ índices derivados.",
            "conclusao": "Extrair os índices certos multiplica a capacidade preditiva. NDRE supera NDVI em 15% na detecção de deficiência de nitrogênio — um dado que pode economizar R$ 200/ha em fertilizantes."
        },
        "🤖 Modelo de ML/DL": {
            "desc": "Treinamento de CNN, LSTM, XGBoost ou Random Forest com os dados extraídos.",
            "numeros": "XGBoost atinge 93% de acurácia. CNN + LSTM combinados chegam a 96% em previsão de safra.",
            "conclusao": "Modelos de IA pagam seu custo de desenvolvimento em 1 safra: uma previsão 90% precisa com 3 meses de antecedência permite negociar contratos futuros com vantagem de R$ 50-150/ton."
        },
        "📊 Predição": {
            "desc": "Geração de mapas de produtividade, alertas de pragas e recomendações de manejo.",
            "numeros": "Modelos bem treinados reduzem perdas por pragas em 25% e por clima em 18%.",
            "conclusao": "A predição transformou a agricultura reativa em proativa. Detectar uma praga 2 semanas antes do dano visível reduz o custo de controle em até 60% e preserva a produtividade."
        },
        "✅ Decisão Agronômica": {
            "desc": "Ação baseada nas predições: aplicação localizada, ajuste de irrigação, manejo diferenciado.",
            "numeros": "Agricultores que usam decisões baseadas em IA têm ROI médio de 320% sobre o investimento em tecnologia.",
            "conclusao": "Este é o elo que transforma dados em lucro. A decisão agronômica assistida por IA reduz custos operacionais em 15-30% e aumenta a produtividade em 10-25% — justificando completamente o investimento em todo o pipeline."
        },
    }

    etapas_labels = list(PIPELINE_INFO.keys())
    cols_pipe = st.columns(len(etapas_labels))
    for i, etapa in enumerate(etapas_labels):
        with cols_pipe[i]:
            if st.button(etapa, key=f"pipe_{i}", use_container_width=True):
                st.session_state["etapa_sel"] = etapa

    if "etapa_sel" in st.session_state:
        info = PIPELINE_INFO[st.session_state["etapa_sel"]]
        st.markdown(f"""
<div style='background:#1a2744; border-left:4px solid #40c463; border-radius:10px;
            padding:18px; margin:12px 0;'>
  <h4 style='color:#40c463; margin:0 0 8px 0;'>{st.session_state["etapa_sel"]}</h4>
  <p style='color:#ccc; margin:4px 0;'><b>O que é:</b> {info["desc"]}</p>
  <p style='color:#40c463; margin:4px 0;'><b>Números:</b> {info["numeros"]}</p>
  <p style='color:#e8f5e9; margin:8px 0 0 0;'><b>Por que vale investir:</b> {info["conclusao"]}</p>
</div>
        """, unsafe_allow_html=True)

    # Funil visual abaixo dos botões
    fig3 = go.Figure(go.Funnel(
        y=["Imagem Satélite/Drone","Pré-processamento","Extração de Features",
           "Modelo de ML/DL","Predição","Decisão Agronômica"],
        x=[100,95,85,80,78,75],
        marker_color=["#40c463","#2ea84d","#1f8a3a","#1f6feb","#1a5ec4","#e3b341"]
    ))
    fig3.update_layout(template="plotly_dark", height=260, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# 🌊 IRRIGAÇÃO INTELIGENTE
# ══════════════════════════════════════════════════════════════════
elif menu == "🌊 Irrigação Inteligente":
    st.title("🌊 Irrigação Inteligente por Satélite")
    st.markdown("---")

    col1,col2,col3 = st.columns(3)
    with col1: st.metric("💧 Consumo Agrícola Global","70%","da água doce")
    with col2: st.metric("📉 Redução com IoT+Satélite","até 50%","no desperdício")
    with col3: st.metric("🌡️ Evapotranspiração","ET calculada","via SEBAL/METRIC")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 💧 Consumo Hídrico: Tradicional vs Inteligente")
        culturas = ["Soja","Milho","Cana-de-açúcar","Café","Algodão"]
        trad = [550,600,1800,900,700]
        intel = [290,330,950,500,380]
        df_agua = pd.DataFrame({"Cultura":culturas,"Tradicional (mm)":trad,"Inteligente (mm)":intel})
        fig = go.Figure()
        fig.add_bar(x=culturas, y=trad, name="Tradicional", marker_color="#e74c3c")
        fig.add_bar(x=culturas, y=intel, name="Inteligente", marker_color="#40c463")
        fig.update_layout(barmode="group", template="plotly_dark",
                          height=300, margin=dict(l=0,r=0,t=10,b=0),
                          title="Lâmina d'água aplicada (mm/ciclo)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 📡 Índices Hídricos por Satélite")
        meses = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        ndwi = [0.3,0.25,0.4,0.35,0.2,0.1,0.05,0.08,0.15,0.28,0.35,0.32]
        et   = [5.5,5.2,4.8,3.9,3.1,2.5,2.2,2.8,3.5,4.5,5.0,5.4]
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=meses, y=et, name="ET (mm/dia)", marker_color="#1f6feb"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=meses, y=ndwi, name="NDWI", line=dict(color="#40c463",width=2),mode="lines+markers"), secondary_y=True)
        fig2.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# 🚁 DRONES & VANTs
# ══════════════════════════════════════════════════════════════════
elif menu == "🚁 Drones & VANTs":
    st.title("🚁 Drones & VANTs na Agricultura")
    st.markdown("---")

    col1,col2,col3 = st.columns(3)
    with col1: st.metric("📐 Resolução máx.","1–2 cm/pixel","câmeras multiespectrais")
    with col2: st.metric("📉 Redução Defensivos","30–40%","aplicação localizada")
    with col3: st.metric("💰 Mercado Drones Agro","US$ 11,2 bi","projeção 2027")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🚁 Comparativo de Modelos")
        df_drones = pd.DataFrame({
            "Modelo":   ["DJI Agras T40","senseFly eBee X","DJI Phantom 4 Multi","Parrot Sequoia+","AgDrone"],
            "Tipo":     ["Pulverização","Mapeamento","Multiespectral","Sensor","Mapeamento"],
            "Autonomia":["10 min","90 min","30 min","N/A","60 min"],
            "Área/dia": ["40 ha","500 ha","100 ha","N/A","200 ha"],
            "Câmera":   ["RGB","RGB+Multi","5 bandas","4 bandas","RGB+Multi"]
        })
        st.dataframe(df_drones, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### 📈 Crescimento do Mercado de Drones Agro")
        anos_d = [2019,2020,2021,2022,2023,2024,2025,2026,2027]
        mkt_d  = [1.2, 1.5, 2.1, 3.0, 4.5, 6.0, 7.8, 9.5, 11.2]
        df_mkt_d = pd.DataFrame({"Ano":anos_d,"US$ Bilhões":mkt_d})
        fig = px.bar(df_mkt_d, x="Ano", y="US$ Bilhões",
                     color="US$ Bilhões", color_continuous_scale="Greens",
                     template="plotly_dark", title="Mercado Global de Drones Agrícolas")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# 🌳 SUSTENTABILIDADE & CARBONO
# ══════════════════════════════════════════════════════════════════
elif menu == "🌳 Sustentabilidade & Carbono":
    st.title("🌳 Sustentabilidade & Mercado de Carbono")
    st.markdown("---")

    col1,col2,col3 = st.columns(3)
    with col1: st.metric("💰 Preço Carbono Agro","US$ 8–15","por tonelada CO₂")
    with col2: st.metric("🎯 Mercado 2030","US$ 50 bi","créditos voluntários")
    with col3: st.metric("🌿 EUDR","Em vigor","desde 2023")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🌱 Práticas de Sequestro de Carbono")
        praticas = ["Plantio Direto","ILPF","Recuperação Pastagens","Reflorestamento","Biochar"]
        tonco2   = [1.5, 3.2, 2.1, 5.0, 2.8]
        df_carb  = pd.DataFrame({"Prática":praticas,"t CO₂/ha/ano":tonco2})
        fig = px.bar(df_carb, x="t CO₂/ha/ano", y="Prática", orientation="h",
                     color="t CO₂/ha/ano", color_continuous_scale="Greens",
                     template="plotly_dark", title="Sequestro de Carbono por Prática")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 📊 Desmatamento Amazônia — PRODES/INPE (km²/ano)")
        anos_d  = [2020, 2021, 2022, 2023, 2024, 2025]
        desmata = [10851, 13038, 11594, 9064, 6518, 5796]
        df_dmt  = pd.DataFrame({"Ano": anos_d, "km² desmatados": desmata})
        fig2 = px.line(df_dmt, x="Ano", y="km² desmatados", markers=True,
                       color_discrete_sequence=["#e74c3c"],
                       template="plotly_dark", title="Desmatamento Amazônia Legal — PRODES/INPE")
        fig2.add_annotation(x=2025, y=5796,
                            text="5.796 km²<br>Menor em 11 anos",
                            showarrow=True, arrowhead=2,
                            arrowcolor="#40c463", font=dict(color="#40c463", size=11),
                            bgcolor="#0d2137", bordercolor="#40c463")
        fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=300)
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Dados completos e fontes PRODES/INPE"):
            st.markdown("""
**Série histórica recente — Amazônia Legal (km²):**

| Ano | Desmatamento (km²) | Variação |
|-----|--------------------|---------|
| 2025 | **5.796** (estimativa) | -11,1% |
| 2024 | 6.518 | -28,2% |
| 2023 | 9.064 | -21,8% |
| 2022 | 11.594 | -11,1% |
| 2021 | 13.038 | +20,2% |
| 2020 | 10.851 | — |

2025 marca o **4° ano consecutivo de queda** e o menor valor dos últimos 11 anos.
O período PRODES vai de **1° de agosto** de um ano a **31 de julho** do seguinte.

**Fontes:**
- [Nota Técnica PRODES 2025 — INPE](https://data.inpe.br/biomasbr/wp-content/uploads/sites/3/2025/10/20251015Nota_tecnica_EstimativaPRODES_2025_F.pdf)
- [TerraBrasilis — Visualizador Interativo](https://terrabrasilis.dpi.inpe.br/app/dashboard/deforestation/biomes/legal_amazon/increments)
- [G1 — Amazônia: desmatamento 2025](https://g1.globo.com/meio-ambiente/noticia/2025/10/30/brasil-desmatamento.ghtml)
            """)

# ══════════════════════════════════════════════════════════════════
# 🇧🇷 ECONOMIA ESPACIAL BRASIL
# ══════════════════════════════════════════════════════════════════
elif menu == "🇧🇷 Economia Espacial Brasil":
    st.title("🇧🇷 Economia Espacial Brasileira")
    st.markdown("---")

    col1,col2,col3 = st.columns(3)
    with col1: st.metric("🚀 CLA Alcântara","2° melhor","economia 30% combustível")
    with col2: st.metric("🛰️ Satélites BR","CBERS + SGDC","operacionais")
    with col3: st.metric("🌱 Agtech captações","R$ 2,5 bi","2020–2024")

    st.markdown("---")

    with st.expander("🚀 Por que o CLA Alcântara é estratégico? — Clique para saber"):
        st.markdown("""
O **Centro de Lançamento de Alcântara (CLA)** é considerado um dos melhores espaçoportos
do mundo, superando o Cabo Canaveral (EUA) e o Cosmódromo de Baikonur (Cazaquistão)
em eficiência energética e economia de custos.

**Vantagens competitivas:**

- **Economia de combustível:** Fica a apenas 2° ao sul do Equador, onde a velocidade
  de rotação da Terra é máxima (~465 m/s). Isso impulsiona o foguete naturalmente,
  gerando economia de até **30% de propelente**.

- **Maior carga útil:** A economia de combustível permite transportar cargas mais pesadas
  para a mesma órbita — vantagem direta para satélites agrícolas e de observação.

- **Clima favorável:** Não está em rota de furacões (como nos EUA) e não sofre com
  frio extremo. O clima quente reduz a densidade do ar e o atrito aerodinâmico.

- **Segurança:** Localizado em uma península com baixa densidade populacional,
  voltado para o Oceano Atlântico — minimiza riscos e facilita recuperação de estágios.

**Fontes:**
[Wikipedia — CLA](https://pt.wikipedia.org/wiki/Centro_Espacial_de_Alc%C3%A2ntara) |
[TecMundo — Base de Alcântara](https://www.tecmundo.com.br/ciencia/261195-base-alcantara-melhores-mundo-lancar-foguetes.htm) |
[SpaceLab — Por que a melhor base é brasileira?](https://www.youtube.com/watch?v=OxBz9LAEgpQ)
        """)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 💰 Investimento em Agtech no Brasil")
        anos_ag = [2018,2019,2020,2021,2022,2023,2024]
        invest  = [0.3, 0.5, 0.8, 1.2, 1.8, 2.2, 2.5]
        df_ag = pd.DataFrame({"Ano":anos_ag,"R$ Bilhões":invest})
        fig = px.area(df_ag, x="Ano", y="R$ Bilhões",
                      color_discrete_sequence=["#009c3b"],
                      template="plotly_dark", title="Captações Agtech — Brasil")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🌾 Produção Agro Brasileira — Principais Culturas")
        culturas_br = ["Soja","Milho","Cana-de-açúcar","Café","Algodão","Laranja"]
        producao_mt = [162, 137, 715, 3.5, 6.8, 16]
        fig2 = px.bar(x=culturas_br, y=producao_mt,
                      color=culturas_br,
                      color_discrete_sequence=px.colors.sequential.Greens,
                      template="plotly_dark", labels={"x":"Cultura","y":"Produção (Mt)"},
                      title="Produção em Milhões de Toneladas (2023/24)")
        fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=280, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)



# ══════════════════════════════════════════════════════════════════
# 💬 CHAT RAG
# ══════════════════════════════════════════════════════════════════
elif menu == "💬 Chat AgroSpace":
    st.title("💬 Chat AgroSpace — Assistente de Agricultura Espacial")
    st.markdown("Faça perguntas sobre **agricultura inteligente, satélites, drones, IoT e geotecnologias**.")
    st.markdown("---")

    with st.spinner("⏳ Inicializando RAG (apenas na primeira vez)..."):
        vectorstore, llm_client, _ = init_rag()

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    exemplos = [
        "O que é NDVI e como é calculado?",
        "Quais satélites são usados na agricultura de precisão?",
        "Como funciona a irrigação inteligente com dados satelitais?",
        "O que é LoRaWAN e qual sua aplicação no agro?",
        "Como o blockchain é usado na rastreabilidade de commodities?",
        "Qual é o futuro do mercado de créditos de carbono no Brasil?"
    ]

    st.markdown("**💡 Perguntas de exemplo:**")
    cols_ex = st.columns(3)
    for i, ex in enumerate(exemplos):
        with cols_ex[i % 3]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.pergunta_rapida = ex

    st.markdown("---")

    for msg in st.session_state.mensagens:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <b>Você:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🛰️ <b>Assistente:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            if "fontes" in msg:
                st.caption("📚 Fontes: " + " · ".join(msg["fontes"]))

    pergunta_default = st.session_state.pop("pergunta_rapida", "") if "pergunta_rapida" in st.session_state else ""
    pergunta = st.text_input("✏️ Digite sua pergunta:", value=pergunta_default, key="input_chat")

    col_btn1, col_btn2 = st.columns([1,5])
    with col_btn1:
        enviar = st.button("Enviar 🚀")
    with col_btn2:
        if st.button("🗑️ Limpar conversa"):
            st.session_state.mensagens = []
            st.rerun()

    if enviar and pergunta.strip():
        st.session_state.mensagens.append({"role":"user","content":pergunta})
        with st.spinner("🤖 Gerando resposta..."):
            try:
                resposta, fontes = gerar_resposta_rag(pergunta, vectorstore, llm_client)
                st.session_state.mensagens.append({"role":"assistant","content":resposta,"fontes":fontes})
            except Exception as e:
                st.session_state.mensagens.append({"role":"assistant","content":f"Erro: {str(e)}","fontes":[]})
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# 🖼️ ANÁLISE DE IMAGEM NDVI
# ══════════════════════════════════════════════════════════════════
elif menu == "🖼️ Análise de Imagem NDVI":
    st.title("🖼️ Análise de Imagem NDVI")
    st.markdown("Faça upload de uma imagem agrícola para análise automática de vegetação e diagnóstico do solo.")
    st.markdown("---")

    modo = st.radio("Escolha o modo de análise:",
                    ["📂 Upload de imagem real", "🔬 Simulação sintética (sem upload)"],
                    horizontal=True)

    def calcular_ndvi_e_diagnostico(RED, NIR):
        NDVI = np.clip((NIR - RED) / (NIR + RED + 1e-8), -1, 1)
        aptidao = np.zeros_like(NDVI, dtype=int)
        aptidao[NDVI >= 0.5]                       = 3
        aptidao[(NDVI >= 0.3) & (NDVI < 0.5)]     = 2
        aptidao[(NDVI >= 0.1) & (NDVI < 0.3)]     = 1
        pct = {
            "✅ Apto (NDVI ≥ 0.5)":        round((NDVI>=0.5).mean()*100,1),
            "🟡 Moderado (0.3–0.5)":       round(((NDVI>=0.3)&(NDVI<0.5)).mean()*100,1),
            "🟠 Estressado (0.1–0.3)":     round(((NDVI>=0.1)&(NDVI<0.3)).mean()*100,1),
            "🔴 Impróprio (< 0.1)":        round((NDVI<0.1).mean()*100,1),
        }
        return NDVI, aptidao, pct

    def mostrar_resultados(img_rgb, RED, NIR, titulo=""):
        NDVI, aptidao, pct = calcular_ndvi_e_diagnostico(RED, NIR)

        col1,col2,col3,col4 = st.columns(4)
        with col1: st.metric("NDVI Médio", f"{NDVI.mean():.3f}")
        with col2: st.metric("NDVI Máx",   f"{NDVI.max():.3f}")
        with col3: st.metric("NDVI Mín",   f"{NDVI.min():.3f}")
        with col4: st.metric("✅ % Apto",  f"{pct['✅ Apto (NDVI ≥ 0.5)']:.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor("#0d1117")
            if img_rgb is not None:
                axes[0].imshow(img_rgb)
                axes[0].set_title("Original RGB", color="white", fontsize=9)
            else:
                axes[0].imshow(RED, cmap="Reds")
                axes[0].set_title("Banda RED", color="white", fontsize=9)
            axes[0].axis("off")
            im = axes[1].imshow(NDVI, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
            axes[1].set_title("Mapa NDVI", color="white", fontsize=9)
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046).ax.tick_params(colors="white",labelsize=7)
            st.pyplot(fig, use_container_width=True)

        with col_b:
            df_pct = pd.DataFrame({"Categoria":list(pct.keys()),"Percentual (%)":list(pct.values())})
            fig2 = px.pie(df_pct, values="Percentual (%)", names="Categoria",
                          color_discrete_sequence=["#27ae60","#f1c40f","#e67e22","#c0392b"],
                          template="plotly_dark", title="Distribuição de Aptidão Agrícola", hole=0.35)
            fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📋 Diagnóstico Agronômico")
        apto = pct["✅ Apto (NDVI ≥ 0.5)"]
        mod  = pct["🟡 Moderado (0.3–0.5)"]
        if apto >= 60:
            st.success(f"✅ **SOLO PRONTO PARA AGRICULTURA** — {apto:.1f}% da área com vegetação saudável. Condições ideais de plantio.")
        elif apto + mod >= 50:
            st.warning(f"⚠️ **SOLO PARCIALMENTE APTO** — {apto+mod:.1f}% com cobertura razoável. Recomenda-se adubação e monitoramento.")
        else:
            st.error(f"❌ **SOLO IMPRÓPRIO / NECESSITA INTERVENÇÃO** — apenas {apto:.1f}% apto. Verificar irrigação, pragas e fertilidade.")

        with st.expander("📘 Ver recomendações detalhadas"):
            if apto >= 60:
                st.markdown("- Manter manejo atual\n- Monitorar NDVI mensalmente via satélite\n- Planejar colheita com base na produtividade estimada")
            elif apto + mod >= 50:
                st.markdown("- Aplicar adubação de cobertura nas áreas moderadas\n- Ajustar irrigação — verificar balanço hídrico\n- Nova análise em 15 dias")
            else:
                st.markdown("- Coletar amostras de solo para análise laboratorial\n- Aplicar defensivo localizado com drone\n- Verificar sistema de irrigação\n- Considerar plantio de cobertura")

    if modo == "📂 Upload de imagem real":
        uploaded = st.file_uploader("📁 Selecione uma ou mais imagens (JPG, PNG)",
                                    type=["jpg","jpeg","png"], accept_multiple_files=True)
        if uploaded:
            for arq in uploaded:
                st.markdown(f"#### 🖼️ {arq.name}")
                img = Image.open(arq).convert("RGB")
                img_array = np.array(img) / 255.0
                RED = img_array[:,:,0]
                NIR = img_array[:,:,1]
                mostrar_resultados(img_array, RED, NIR, arq.name)
                st.markdown("---")
        else:
            st.info("👆 Faça upload de uma imagem agrícola para iniciar a análise.")

    else:
        st.info("🔬 Usando imagem sintética com 4 zonas agrícolas distintas.")
        np.random.seed(42)
        SIZE = 100
        RED = np.zeros((SIZE,SIZE)); NIR = np.zeros((SIZE,SIZE))
        RED[:50,:50] = np.random.normal(0.08,0.015,(50,50)).clip(0,1)
        NIR[:50,:50] = np.random.normal(0.72,0.06,(50,50)).clip(0,1)
        RED[:50,50:] = np.random.normal(0.18,0.02,(50,50)).clip(0,1)
        NIR[:50,50:] = np.random.normal(0.50,0.07,(50,50)).clip(0,1)
        RED[50:,:50] = np.random.normal(0.30,0.03,(50,50)).clip(0,1)
        NIR[50:,:50] = np.random.normal(0.32,0.05,(50,50)).clip(0,1)
        RED[50:,50:] = np.random.normal(0.40,0.04,(50,50)).clip(0,1)
        NIR[50:,50:] = np.random.normal(0.22,0.03,(50,50)).clip(0,1)
        mostrar_resultados(None, RED, NIR, "Simulação")
