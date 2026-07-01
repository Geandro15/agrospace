# 🛰️ AgroSpace Dashboard

Dashboard interativo que une **agricultura de precisão**, **nova economia espacial** e **inteligência artificial**, construído com Streamlit. O projeto reúne sensoriamento remoto, análise de imagem NDVI, automação de coleta de dados (scraping/RPA), análise de documentos e planilhas com IA, e um assistente conversacional (RAG) especializado em agro espacial.

## 📋 Sobre o projeto

O AgroSpace foi desenvolvido como projeto acadêmico para explorar como tecnologias espaciais e de IA — satélites, drones, IoT, machine learning — vêm transformando o agronegócio. O dashboard combina conteúdo educacional, calculadoras técnicas (como NDVI) e ferramentas de automação em uma única interface.

## 🗺️ Módulos

| Módulo | Descrição |
|---|---|
| 🏠 Home | Visão geral do mercado espacial e dos módulos do dashboard |
| 📡 Sensoriamento Remoto | NDVI, satélites Sentinel-2 e Landsat, índices vegetativos |
| 🤖 IA & Machine Learning | CNN, LSTM, Random Forest e outros modelos usados na previsão de safra |
| 🌊 Irrigação Inteligente | Evapotranspiração (ET), NDWI, IoT aplicado à gestão hídrica |
| 🚁 Drones & VANTs | Drones agrícolas, fotogrametria, mapeamento aéreo |
| 🌳 Sustentabilidade & Carbono | Créditos de carbono, PRODES, MapBiomas, EUDR |
| 🇧🇷 Economia Espacial Brasil | CLA Alcântara, satélite CBERS, investimento em agtech |
| 🖼️ Análise de Imagem NDVI | Upload de imagem agrícola com cálculo automático de NDVI e diagnóstico de aptidão do solo |
| 🔄 Automação Robótica de Processos | Web scraping com XPath, parsing de XML/KML, demonstração de RPA com Selenium |
| 📄 Pesquisa Doc | Upload de PDF/Word com extração e análise automática de conteúdo |
| 📊 Análise de Planilhas IA | Upload de CSV/XLSX com geração automática de gráficos, estatísticas e insights |
| 💬 Chat AgroSpace | Assistente conversacional (RAG) que responde perguntas com base na própria documentação do projeto |

## 🧠 Como funciona o assistente (RAG)

O Chat AgroSpace e os Insights de Planilhas usam uma arquitetura de **Retrieval-Augmented Generation**:

1. Uma base de conhecimento própria (módulos sobre sensoriamento remoto, IoT, IA agrícola, drones, economia espacial, irrigação, sustentabilidade etc.) é dividida em trechos com `RecursiveCharacterTextSplitter`.
2. Os trechos são vetorizados com o modelo de embeddings multilíngue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` e indexados em um banco vetorial **FAISS**.
3. Cada pergunta do usuário busca os trechos mais relevantes (similarity search) e os envia como contexto para um LLM (`meta-llama/Llama-3.1-8B-Instruct`, via Hugging Face Inference API) gerar a resposta.
4. As fontes usadas na resposta são exibidas ao usuário para rastreabilidade.

## 🌱 Análise de Imagem NDVI

O módulo de NDVI aceita upload de imagens reais (extraindo bandas RED/NIR a partir dos canais RGB) ou uma simulação sintética com quatro zonas agrícolas distintas. A partir das bandas, calcula:

```
NDVI = (NIR - RED) / (NIR + RED)
```

O resultado classifica a área em zonas de aptidão agrícola (apto, moderado, estressado, impróprio) e gera um diagnóstico textual com recomendações práticas.

## ⚙️ Tecnologias utilizadas

- **Interface:** Streamlit
- **Dados e visualização:** Pandas, NumPy, Plotly, Matplotlib, Pillow
- **IA/NLP:** LangChain, Hugging Face Hub (embeddings + LLM via Inference API), FAISS
- **Documentos:** pdfplumber, PyMuPDF, pypdf, python-docx
- **Coleta de dados:** requests, BeautifulSoup, lxml (XPath)

## 🚀 Como executar

### Pré-requisitos
- Python 3.10+
- Uma chave de API da Hugging Face (necessária para o Chat e os Insights de IA)

### Instalação

```bash
# Clonar o repositório
git clone <url-do-repositorio>
cd agroSpace

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Configuração

O projeto usa a API de inferência da Hugging Face para o LLM do chat. Defina sua chave como variável de ambiente antes de executar:

```bash
export HF_TOKEN="sua_chave_aqui"        # Linux/Mac
set HF_TOKEN=sua_chave_aqui             # Windows (cmd)
```

> Sem essa variável configurada, os módulos de Chat e Insights de IA não conseguirão gerar respostas — os demais módulos (NDVI, gráficos, scraping) funcionam normalmente sem ela.

### Executar

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente no navegador, em `http://localhost:8501`.

## 📁 Estrutura do projeto

```
agroSpace/
├── app.py                       # Aplicação principal (todos os módulos)
├── requirements.txt             # Dependências do projeto
├── AgroSpace_Documentacao.docx  # Documentação acadêmica do projeto
└── README.md
```

## 📚 Contexto acadêmico

Este projeto foi desenvolvido como trabalho acadêmico, integrando conceitos de inteligência artificial, sensoriamento remoto e automação aplicados ao agronegócio.
