# 03 - Arquitetura do Sistema

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura em Camadas](#arquitetura-em-camadas)
- [API RESTful com Flask](#api-restful-com-flask)
- [Endpoints da API](#endpoints-da-api)
- [Fluxo de Requisições](#fluxo-de-requisições)
- [Containerização com Docker](#containerização-com-docker)
- [Configuração e Variáveis de Ambiente](#configuração-e-variáveis-de-ambiente)
- [Gerenciamento de Arquivos](#gerenciamento-de-arquivos)
- [Integração dos Componentes](#integração-dos-componentes)

---

## Visão Geral

O PneumoFinder foi projetado como uma **API RESTful** modular seguindo princípios de **programação funcional** e **separação de responsabilidades**. A arquitetura permite que diferentes componentes (CNN, Grad-CAM, LLM, banco de dados) operem de forma independente e testável, facilitando manutenção e extensibilidade.

### Princípios Arquiteturais

1. **Modularidade**: Cada componente possui responsabilidade única e bem definida
2. **Funcionalidade pura**: Funções sem efeitos colaterais sempre que possível
3. **Configuração centralizada**: Todas as configurações em um único ponto
4. **Stateless**: API não mantém estado entre requisições (RESTful)
5. **Containerização**: Deploy simplificado com Docker para produção

### Stack Tecnológico

| Camada              | Tecnologia              | Versão | Função                        |
| ------------------- | ----------------------- | ------ | ----------------------------- |
| **Framework Web**   | Flask                   | 3.1.0  | Servidor HTTP e roteamento    |
| **WSGI**            | Werkzeug                | 3.x    | Interface WSGI (dev server)   |
| **CORS**            | Flask-CORS              | -      | Cross-Origin Resource Sharing |
| **Deep Learning**   | TensorFlow              | 2.19.0 | Inferência CNN                |
| **Computer Vision** | OpenCV                  | 4.x    | Processamento de imagens      |
| **Vision LLM**      | LLaVA-Med               | 7B     | Análise visual médica         |
| **Text LLM**        | BioMistral-7B           | 7B     | Geração de laudos médicos     |
| **RAG**             | ChromaDB + 31 guidelines| -      | Grounding de conhecimento     |
| **Containerização** | Docker + Docker Compose | -      | Deploy e orquestração         |

---

## Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────────┐
│                      CAMADA DE API                          │
│  Flask + Flask-CORS + Werkzeug                              │
│  (src/api/routes.py)                                        │
│  - Roteamento HTTP (/diagnose, /analyze)                    │
│  - Validação de entrada                                     │
│  - Serialização JSON                                        │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CAMADA DE NEGÓCIO (CORE)                  │
│  ┌──────────────┐ ┌─────────────┐ ┌───────────────────┐     │
│  │ diagnosis.py │ │visualizat.. │ │medical_pipeline.py│     │
│  │  - CNN       │ │  - Grad-CAM │ │                   │     │
│  │  - ResNet50  │ │  - Heatmap  │ │ Stage 1: LLaVA-Med│     │
│  └──────────────┘ └─────────────┘ │ Stage 2: BioMistral│    │
│                                   └───────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      CAMADA RAG                             │
│  ┌───────────────────┐          ┌──────────────────────┐    │
│  │  retriever.py     │          │  vector_store.py     │    │
│  │  - Guidelines (31)│          │  - ChromaDB          │    │
│  │  - Sample Reports │          │  - Embeddings        │    │
│  │  - Terminology    │          │  - Relevance Filter  │    │
│  └───────────────────┘          └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CAMADA DE PERSISTÊNCIA                    │
│  ┌───────────────────┐          ┌──────────────────────┐    │
│  │  database.py      │          │  Modelos HuggingFace │    │
│  │  repositories.py  │          │  - LLaVA-Med (13GB)  │    │
│  │  - SQLite         │          │  - BioMistral (13GB) │    │
│  │  - CRUD           │          │  - device_map="auto" │    │
│  └───────────────────┘          └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA DE UTILIDADES                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐      │
│  │  config.py  │ │ file_utils.py│ │ medical_prompts.py│     │
│  │  - Env vars │ │ - Upload     │ │ - Vision prompt  │      │
│  │  - Paths    │ │ - Cleanup    │ │ - Medical prompt │      │
│  └─────────────┘ └──────────────┘ └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Estrutura de Diretórios

```
src/
├── api/
│   ├── __init__.py
│   └── routes.py                   # Flask app + endpoints
├── core/                           # Lógica de negócio
│   ├── __init__.py
│   ├── diagnosis.py                # Funções CNN
│   ├── visualization.py            # Funções Grad-CAM
│   ├── clinical_description.py     # Funções LLM (legado)
│   └── medical_pipeline.py         # Pipeline 2 estágios (LLaVA-Med + BioMistral)
├── rag/                            # Sistema RAG
│   ├── __init__.py
│   ├── retriever.py                # MedicalRetriever (guidelines, reports)
│   └── vector_store.py             # ChromaDB wrapper
├── prompts/                        # Prompt Engineering
│   ├── __init__.py
│   └── medical_prompts.py          # Vision + Medical prompts
├── db/                             # Persistência
│   ├── __init__.py
│   ├── database.py                 # Conexão SQLite
│   └── repositories.py             # CRUD + deduplicação
├── utils/                          # Utilitários
│   ├── __init__.py
│   ├── config.py                   # Configuração centralizada
│   ├── file_utils.py               # Manipulação de arquivos
│   └── image_utils.py              # Processamento de imagens
└── bots/
    ├── __init__.py
    └── whatsapp_bot.py             # Integração Twilio (futuro)
```

---

## API RESTful com Flask

### Escolha do Framework

Flask foi escolhido por:

1. **Leveza**: Não impõe estrutura rígida, ideal para aplicações funcionais
2. **Simplicidade**: Roteamento intuitivo e decoradores Python nativos
3. **Flexibilidade**: Fácil integração com TensorFlow, OpenCV e Ollama
4. **Comunidade**: Vasta documentação e extensões (Flask-CORS)
5. **Produção**: Compatível com WSGI servers (Gunicorn, uWSGI)

### Inicialização da Aplicação

O ponto de entrada (`app.py` na raiz) delega para `src/api/app.py`:

```python
# app.py (raiz do projeto)
if __name__ == "__main__":
    from src.api.app import app, config, ensure_directory

    # Garante que diretório temporário existe
    ensure_directory(config.temp_dir)

    # Inicia servidor Flask
    app.run(debug=config.debug_mode, port=config.api_port)
```

**Código:** `app.py:1-21`

### Carregamento de Modelos na Inicialização

Para evitar latência em cada requisição, os modelos são carregados **uma única vez** no startup:

```python
# src/api/app.py
from flask import Flask
from src.core.diagnosis import load_cnn_model
from src.core.visualization import find_resnet_base, find_last_conv_layer
from src.utils.config import config

app = Flask(__name__)
CORS(app)

# Carrega CNN (ResNet50 + camadas customizadas)
cnn_model = load_cnn_model(config.cnn_model_path)

# Extrai ResNet base para Grad-CAM
resnet_base = find_resnet_base(cnn_model)
last_conv_layer = find_last_conv_layer(resnet_base)

print(f"✓ Modelos carregados. Grad-CAM target: {last_conv_layer.name}")
```

**Código:** `src/api/app.py:42-47`

Essa estratégia reduz o tempo de resposta de ~15s (com carregamento) para ~2s (CNN) + ~13s (LLM).

---

## Endpoints da API

A API expõe **9 endpoints** divididos em 3 categorias:

### 1. Endpoints Principais (Inglês)

#### `POST /diagnose` - Diagnóstico Simples

**Descrição:** Classifica radiografia como `PNEUMONIA` ou `NORMAL` usando CNN.

**Entrada:**

```bash
curl -X POST http://localhost:5001/diagnose \
  -F "image=@radiografia.jpg"
```

**Saída:**

```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 0.87,
  "diagnosis_id": 42
}
```

**Código:** `src/api/app.py:56-105`

---

#### `POST /diagnose/explained` - Diagnóstico Completo

**Descrição:** Além do diagnóstico, gera **Grad-CAM** e **descrição clínica com LLM**.

**Entrada:**

```bash
curl -X POST http://localhost:5001/diagnose/explained \
  -F "image=@radiografia.jpg"
```

**Saída:**

```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 0.87,
  "diagnosis_id": 42,
  "description": "Radiografia de tórax revela consolidações bilaterais...",
  "heatmap_url": "/static/temp/radiografia_heatmap.png",
  "overlay_url": "/static/temp/radiografia_overlay.png"
}
```

**Código:** `src/api/routes.py:120-206`

**Fluxo interno:**

1. Upload e deduplicação (SHA-256)
2. CNN + Grad-CAM (`diagnose_with_visualization`)
3. LLM (`generate_clinical_description`)
4. Salvar BLOBs e embeddings no banco
5. Retornar URLs para visualizações

---

#### `POST /analyze` - Análise Completa com Pipeline 2 Estágios (NOVO)

**Descrição:** Endpoint principal que executa o pipeline completo de 2 estágios com LLaVA-Med e BioMistral.

**Entrada:**

```bash
curl -X POST http://localhost:5001/analyze \
  -F "image=@radiografia.jpg"
```

**Saída:**

```json
{
  "success": true,
  "diagnosis": "PNEUMONIA",
  "confidence": 0.9621,
  "vision_findings": "Bilateral lower lobe consolidations consistent with pneumonia. No pleural effusion.",
  "medical_report": "Findings: Bilateral airspace opacities in the lower lobes with air bronchograms. Impression: Community-acquired pneumonia. Clinical correlation recommended.",
  "latency_s": 49.2,
  "stage1_latency_s": 38.5,
  "stage2_latency_s": 8.2,
  "model": {
    "vision": "llava-med",
    "medical": "BioMistral-7B"
  }
}
```

**Fluxo interno:**

```
1. Upload e preprocessamento da imagem
      ↓
2. CNN Classification (ResNet50) + Grad-CAM
      ↓
3. Stage 1: LLaVA-Med Vision Analysis
   ├─→ Build RAG context (guidelines)
   ├─→ Build vision prompt
   └─→ Generate findings description
      ↓
4. Model Unload (LLaVA-Med → free MPS memory)
      ↓
5. Stage 2: BioMistral-7B Medical Report
   ├─→ Build RAG context (guidelines + sample reports)
   ├─→ Build medical prompt with vision findings
   └─→ Generate professional radiology report
      ↓
6. Model Unload (BioMistral → free memory)
      ↓
7. Return JSON response
```

**Código:** `src/api/routes.py` + `src/core/medical_pipeline.py`

---

#### `GET /health` - Health Check

**Descrição:** Verifica se a API está operacional (útil para orquestradores como Kubernetes).

**Saída:**

```json
{
  "status": "healthy",
  "service": "PneumoFinder API"
}
```

**Código:** `src/api/app.py:50-53`

---

### 2. Endpoints de Banco de Dados

#### `GET /api/diagnoses/<id>` - Recuperar Diagnóstico

**Descrição:** Busca diagnóstico por ID com metadados completos.

**Exemplo:**

```bash
curl http://localhost:5001/api/diagnoses/42
```

**Saída:**

```json
{
  "id": 42,
  "diagnosis": "PNEUMONIA",
  "confidence": 0.87,
  "created_at": "2024-01-15T10:30:00",
  "metadata": { "endpoint": "/diagnose/explained", "filename": "xray.jpg" },
  "description": "Consolidações bilaterais...",
  "has_visualizations": true
}
```

**Código:** `src/api/app.py:214-250`

---

#### `POST /api/search/similar` - Busca Semântica

**Descrição:** Encontra diagnósticos similares usando embeddings de texto (ChromaDB + HNSW).

**Entrada:**

```bash
curl -X POST http://localhost:5001/api/search/similar \
  -H "Content-Type: application/json" \
  -d '{"query": "infiltrados pulmonares bilaterais", "top_k": 3}'
```

**Saída:**

```json
{
  "query": "infiltrados pulmonares bilaterais",
  "top_k": 3,
  "results": [
    {
      "diagnosis_id": 38,
      "diagnosis": "PNEUMONIA",
      "confidence": 0.92,
      "similarity": 0.83,
      "description": "Consolidações bilaterais predominantes...",
      "created_at": "2024-01-14T08:15:22"
    },
    ...
  ]
}
```

**Código:** `src/api/app.py:253-294`

---

#### `GET /api/diagnoses/recent?limit=10` - Diagnósticos Recentes

**Descrição:** Lista últimos N diagnósticos ordenados por data.

**Saída:**

```json
{
  "count": 10,
  "diagnoses": [
    {
      "id": 50,
      "diagnosis": "NORMAL",
      "confidence": 0.95,
      "created_at": "..."
    },
    {
      "id": 49,
      "diagnosis": "PNEUMONIA",
      "confidence": 0.88,
      "created_at": "..."
    }
  ]
}
```

**Código:** `src/api/app.py:297-329`

---

### 3. Endpoints Legados (Retrocompatibilidade)

Para manter compatibilidade com versões anteriores da API (em português):

- `POST /diagnosticar_pneumonia` → redireciona para `/diagnose`
- `POST /diagnostico_completo` → redireciona para `/diagnose/complete`
- `POST /diagnosticar_com_descricao` → redireciona para `/diagnose/explained`

**Código:** `src/api/app.py:332-348`

---

## Fluxo de Requisições

### Fluxo 1: Diagnóstico Simples (`POST /diagnose`)

```
1. Cliente envia POST com imagem (multipart/form-data)
      ↓
2. Flask salva upload em temp/ (src/utils/file_utils.py)
      ↓
3. Calcula SHA-256 do arquivo para deduplicação
      ↓
4. Verifica se hash já existe no banco (src/db/repositories.py)
      ├─→ Existe: retorna diagnosis_id existente (cache hit)
      └─→ Não existe: continua processamento
      ↓
5. Carrega e preprocessa imagem (src/utils/image_utils.py)
      ↓
6. Executa inferência CNN (src/core/diagnosis.py)
      ↓
7. Salva resultado no SQLite (src/db/repositories.py)
      ↓
8. Retorna JSON: {diagnosis, confidence, diagnosis_id}
      ↓
9. Cleanup: remove arquivo temporário (src/utils/file_utils.py)
```

**Tempo médio:** ~2 segundos (CNN) + 50ms (I/O)

---

### Fluxo 2: Diagnóstico Completo (`POST /diagnose/explained`)

```
1-4. [Idêntico ao Fluxo 1]
      ↓
5. Executa CNN + Grad-CAM simultaneamente
      └─→ diagnose_with_visualization() retorna:
          - diagnosis, confidence (CNN)
          - heatmap_path, overlay_path (Grad-CAM)
      ↓
6. Gera descrição clínica com LLaVA:
      └─→ generate_clinical_description(
            diagnosis, confidence, original_img, overlay_img
          )
      ↓
7. Salva no banco:
      ├─→ SQLite: diagnosis, confidence, metadata
      ├─→ SQLite BLOBs: heatmap + overlay (PNG binário)
      └─→ ChromaDB: embedding 384-dim da descrição clínica
      ↓
8. Retorna JSON com:
      - diagnosis, confidence, diagnosis_id
      - description (texto LLM)
      - heatmap_url, overlay_url (URLs para servir imagens)
      ↓
9. Cleanup: remove arquivos temporários
```

**Tempo médio:** ~2s (CNN) + ~1s (Grad-CAM) + ~13s (LLM) = **16 segundos**

---

### Fluxo 3: Pipeline 2 Estágios (`POST /analyze`)

```
1. Cliente envia POST com imagem (multipart/form-data)
      ↓
2. Flask salva upload em temp/
      ↓
3. Executa CNN + Grad-CAM (diagnose_with_visualization)
      ↓
4. Inicializa MedicalPipeline com RAG
      ↓
5. Stage 1: LLaVA-Med Vision Analysis
      ├─→ Carrega LLaVA-Med (device_map="auto", ~13GB)
      ├─→ Busca guidelines relevantes via RAG
      ├─→ Gera vision prompt com contexto
      ├─→ Processa imagem + gera findings
      └─→ Descarrega LLaVA-Med (gc.collect + MPS cache clear)
      ↓
6. Stage 2: BioMistral Medical Report
      ├─→ Carrega BioMistral-7B (device_map="auto", ~13GB)
      ├─→ Busca guidelines + sample reports via RAG
      ├─→ Gera medical prompt com vision findings
      ├─→ Gera laudo radiológico profissional
      └─→ Descarrega BioMistral (gc.collect + MPS cache clear)
      ↓
7. Retorna JSON com todos os resultados
      ↓
8. Cleanup: remove arquivos temporários
```

**Tempo médio:** ~0.5s (CNN) + ~38s (LLaVA-Med) + ~2s (unload) + ~8s (BioMistral) = **~49 segundos**

**Memória:**
- Peak: ~14GB (um modelo por vez)
- Baseline: ~4GB (RAG + overhead)

---

### Fluxo 4: Busca Semântica (`POST /api/search/similar`)

```
1. Cliente envia query text: "infiltrados bilaterais"
      ↓
2. Gera embedding 384-dim com sentence-transformers
      └─→ modelo: all-MiniLM-L6-v2
      ↓
3. Consulta ChromaDB (HNSW index):
      └─→ query_embeddings=[embedding]
      └─→ n_results=top_k
      ↓
4. ChromaDB retorna IDs de diagnósticos + scores de similaridade
      ↓
5. Para cada ID, busca dados completos no SQLite:
      └─→ diagnosis, confidence, description, created_at
      ↓
6. Ordena resultados por similaridade (maior → menor)
      ↓
7. Retorna JSON com lista ranqueada
```

**Tempo médio:** ~200ms (embedding) + ~50ms (HNSW) + ~10ms (SQL) = **260ms**

---

## Containerização com Docker

### Arquitetura de Containers

```
┌─────────────────────────────────────────────────────────────┐
│                     Host Machine (M4 Pro 24GB)              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              HuggingFace Models (Host)               │   │
│  │  ┌─────────────────┐     ┌────────────────────┐      │   │
│  │  │   LLaVA-Med     │     │   BioMistral-7B    │      │   │
│  │  │   (Stage 1)     │     │   (Stage 2)        │      │   │
│  │  │   ~13GB MPS+CPU │     │   ~13GB MPS+CPU    │      │   │
│  │  └─────────────────┘     └────────────────────┘      │   │
│  │  (Carregamento sequencial - apenas 1 por vez)        │   │
│  └──────────────────────────────────────────────────────┘   │
│                               │                             │
│  ┌──────────────────────┐     ▼                             │
│  │   PneumoFinder API   │◄────────────────────────┐         │
│  │   Port: 5001         │                         │         │
│  │   - Flask            │     ┌───────────────────┴──────┐  │
│  │   - MedicalPipeline  │     │       ChromaDB           │  │
│  │   - CNN (ResNet50)   │◄────┤       Port: 8000         │  │
│  │                      │     │       - Guidelines (31)  │  │
│  └──────────────────────┘     │       - Sample Reports   │  │
│                               └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Nota:** O pipeline atual executa no host (não containerizado) para:
1. **Acesso MPS**: Apple Silicon requer drivers nativos
2. **Memória**: 24GB unified memory não é virtualizável
3. **Performance**: Evita overhead de container para modelos de 13GB

### Dockerfile Multi-Stage

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

# Usuário não-root para segurança
RUN useradd -m -u 1000 pneumofinder

# Dependências de sistema (OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Copia artefatos do builder
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

# Cria diretórios com permissões corretas
RUN mkdir -p database temp && \
    chown -R pneumofinder:pneumofinder /app

USER pneumofinder

# Health check para orquestradores
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:5001/health')"

EXPOSE 5001
CMD ["python", "app.py"]
```

**Arquivo:** `Dockerfile:1-50`

### Docker Compose

Orquestra API + ChromaDB:

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "5001:5001"
    environment:
      - FLASK_ENV=production
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    volumes:
      - ./models:/app/models:ro # Modelo CNN (read-only)
      - api-database:/app/database # SQLite persistente
      - api-temp:/app/temp # Uploads temporários
    depends_on:
      - chromadb
    extra_hosts:
      - "host.docker.internal:host-gateway" # Acessa Ollama no host
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma # Vetores persistentes
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    restart: unless-stopped

volumes:
  api-database:
  api-temp:
  chroma-data:
```

**Arquivo:** `docker-compose.yml:1-45`

### Por que os LLMs ficam no Host?

Os modelos LLaVA-Med e BioMistral **NÃO** são containerizados porque:

1. **MPS Access**: Apple Silicon Metal Performance Shaders requer drivers nativos
2. **Memória**: 24GB unified memory precisa de acesso direto (não virtualizado)
3. **Device Map**: `device_map="auto"` divide modelo entre MPS e CPU/RAM
4. **Performance**: Overhead de container adiciona latência significativa em modelos de 13GB

---

## Configuração e Variáveis de Ambiente

### Classe `Config` Centralizada

```python
# src/utils/config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuração centralizada com suporte a variáveis de ambiente."""

    # API
    api_port: int = int(os.getenv("API_PORT", "5001"))
    debug_mode: bool = os.getenv("FLASK_DEBUG", "0") == "1"

    # Modelos CNN
    cnn_model_path: str = os.getenv("CNN_MODEL_PATH", "models/pneumonia_model.keras")

    # Vision Model (Stage 1)
    vision_model: str = os.getenv("VISION_MODEL", "llava-med")  # ou "llava-llama3"

    # Medical Model (Stage 2)
    medical_model: str = os.getenv("MEDICAL_MODEL", "BioMistral/BioMistral-7B")

    # Device
    device: str = os.getenv("DEVICE", "mps")  # "mps", "cuda", "cpu"
    max_mps_memory: str = os.getenv("MAX_MPS_MEMORY", "10GiB")
    max_cpu_memory: str = os.getenv("MAX_CPU_MEMORY", "16GiB")

    # RAG
    use_rag: bool = os.getenv("USE_RAG", "1") == "1"
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")

    # Diretórios
    temp_dir: str = "temp"
    database_dir: str = "database"

config = Config()
```

**Código:** `src/utils/config.py`

### Variáveis de Ambiente Suportadas

| Variável          | Padrão                         | Descrição                    |
| ----------------- | ------------------------------ | ---------------------------- |
| `API_PORT`        | `5001`                         | Porta HTTP do Flask          |
| `FLASK_DEBUG`     | `0`                            | Debug mode (0=off, 1=on)     |
| `CNN_MODEL_PATH`  | `models/pneumonia_model.keras` | Caminho do modelo TensorFlow |
| `VISION_MODEL`    | `llava-med`                    | Modelo de visão (Stage 1)    |
| `MEDICAL_MODEL`   | `BioMistral/BioMistral-7B`     | Modelo médico (Stage 2)      |
| `DEVICE`          | `mps`                          | Device (mps/cuda/cpu)        |
| `MAX_MPS_MEMORY`  | `10GiB`                        | Limite MPS para device_map   |
| `MAX_CPU_MEMORY`  | `16GiB`                        | Limite CPU para device_map   |
| `USE_RAG`         | `1`                            | Usar RAG grounding (1=on)    |
| `CHROMA_PERSIST_DIR` | `data/chroma`               | Diretório ChromaDB           |

### Arquivo `.env` (Exemplo)

```bash
# .env (não commitado no Git)
FLASK_DEBUG=1
VISION_MODEL=llava-med
DEVICE=mps
USE_RAG=1

# Limites de memória (M4 Pro 24GB)
MAX_MPS_MEMORY=10GiB
MAX_CPU_MEMORY=16GiB

# Twilio (opcional, para WhatsApp)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
```

**Arquivo:** `.env.example`

---

## Gerenciamento de Arquivos

### Upload e Deduplicação

```python
# src/utils/file_utils.py
import hashlib
import os
from werkzeug.utils import secure_filename

def save_uploaded_file(file, directory: str) -> str:
    """Salva arquivo enviado e retorna path seguro."""
    filename = secure_filename(file.filename)
    filepath = os.path.join(directory, filename)
    file.save(filepath)
    return filepath

def cleanup_file(filepath: str) -> None:
    """Remove arquivo temporário."""
    if os.path.exists(filepath):
        os.remove(filepath)
```

**Código:** `src/utils/file_utils.py:10-28`

### Cálculo de Hash (Deduplicação)

```python
# src/api/app.py (exemplo de uso)
with open(temp_path, "rb") as f:
    image_hash = hashlib.sha256(f.read()).hexdigest()

# Verifica se já existe no banco
existing = repo.get_by_hash(image_hash)
if existing:
    return jsonify({"diagnosis_id": existing["id"], ...})
```

**Código:** `src/api/app.py:76-77`

Isso evita reprocessar imagens idênticas, economizando ~16s por requisição duplicada.

---

## Integração dos Componentes

### Fluxo Completo End-to-End

```
      Cliente HTTP
         │
         ▼
  ┌─────────────┐
  │ Flask Router│ (src/api/app.py)
  └──────┬──────┘
         │
         ├─→ Validação de entrada (Flask request)
         │
         ├─→ Upload e hash (src/utils/file_utils.py)
         │
         ├─→ Deduplicação (src/db/repositories.py)
         │
         ├─→ CNN Inference (src/core/diagnosis.py)
         │       └─→ TensorFlow + ResNet50
         │
         ├─→ Grad-CAM (src/core/visualization.py)
         │       └─→ OpenCV + NumPy
         │
         ├─→ LLM Explicação (src/core/clinical_description.py)
         │       └─→ HTTP request para Ollama (host)
         │
         ├─→ Salvar SQLite (src/db/repositories.py)
         │       └─→ Diagnosis + BLOBs (heatmap/overlay)
         │
         ├─→ Salvar ChromaDB (src/db/vector_store.py)
         │       └─→ Embeddings da descrição clínica
         │
         ├─→ Cleanup (src/utils/file_utils.py)
         │
         └─→ JSON Response
```

### Exemplo de Código Integrado

```python
# src/api/app.py:150-165
diagnosis, confidence, heatmap_path, overlay_path = diagnose_with_visualization(
    cnn_model, resnet_base, last_conv_layer, temp_path, config.temp_dir
)

description = generate_clinical_description(
    diagnosis, confidence, temp_path, overlay_path,
    config.medical_prompt_path, config.ollama_host, config.ollama_model
)

diagnosis_id = save_diagnosis(
    image_hash=image_hash,
    diagnosis=diagnosis,
    confidence=confidence,
    metadata={"endpoint": "/diagnose/explained", "filename": image_file.filename}
)

# Salva visualizações como BLOBs
with open(heatmap_path, "rb") as f:
    heatmap_bytes = f.read()
with open(overlay_path, "rb") as f:
    overlay_bytes = f.read()
save_visualization(diagnosis_id, heatmap_bytes, overlay_bytes)

# Salva descrição + embeddings
save_clinical_description(diagnosis_id, description, diagnosis, confidence)
```

**Código:** `src/api/app.py:151-182`

---

## Considerações de Produção

### Escalabilidade

1. **Horizontal**: Múltiplas instâncias do container API atrás de load balancer (Nginx/HAProxy)
2. **Vertical**: Aumentar workers do Gunicorn (1 worker por núcleo CPU)
3. **Cache**: Redis para diagnósticos frequentes (Time-To-Live de 1 hora)
4. **Batch Processing**: Fila de mensagens (Celery + RabbitMQ) para processamento assíncrono

### Segurança

1. **HTTPS**: Certificados SSL/TLS (Let's Encrypt)
2. **Rate Limiting**: Flask-Limiter para prevenir DoS
3. **Autenticação**: API keys ou JWT para acesso controlado
4. **CORS**: Restrição de origens permitidas (production)
5. **Validação**: Limite de tamanho de upload (5MB máximo)

### Monitoramento

1. **Logs**: Estruturados em JSON (ELK stack ou Loki)
2. **Métricas**: Prometheus + Grafana
   - Latência por endpoint
   - Taxa de erro (5xx)
   - Uso de memória/CPU
3. **Health checks**: `/health` endpoint para Kubernetes liveness probe

---

## Referências

- Flask Documentation: https://flask.palletsprojects.com/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- HuggingFace Accelerate (device_map): https://huggingface.co/docs/accelerate/
- LLaVA-Med: https://github.com/microsoft/LLaVA-Med
- BioMistral: https://huggingface.co/BioMistral/BioMistral-7B
- ChromaDB: https://www.trychroma.com/
- TensorFlow Serving: https://www.tensorflow.org/tfx/guide/serving (alternativa futura)

---

**Documento atualizado para TCC - PneumoFinder v3.0**
**Última atualização:** Fevereiro 2026
