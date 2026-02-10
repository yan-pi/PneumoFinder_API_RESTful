# 02 - Fundamentos Teóricos

## Sumário

- [1. Redes Neurais Convolucionais](#1-redes-neurais-convolucionais)
- [2. Transfer Learning e ResNet50](#2-transfer-learning-e-resnet50)
- [3. Técnicas de Explicabilidade](#3-técnicas-de-explicabilidade)
- [4. Modelos de Linguagem Multimodais](#4-modelos-de-linguagem-multimodais)
- [5. LLMs Especializados em Medicina](#5-llms-especializados-em-medicina)
- [6. Retrieval-Augmented Generation (RAG)](#6-retrieval-augmented-generation-rag)
- [7. Embeddings e Busca Semântica](#7-embeddings-e-busca-semântica)

---

## 1. Redes Neurais Convolucionais

### 1.1 Arquitetura Básica

CNNs são especializadas em processamento de imagens através de camadas:

```
Input Image (224x224x3)
        ↓
┌─────────────────┐
│ Convolutional   │ → Extração de features locais (bordas, texturas)
│ Layers          │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Pooling Layers  │ → Redução de dimensionalidade
└────────┬────────┘
         ↓
┌─────────────────┐
│ Fully Connected │ → Classificação
│ Layers          │
└────────┬────────┘
         ↓
Output: [NORMAL, PNEUMONIA]
```

**Operações principais:**

- **Convolução:** Filtros deslizantes detectam padrões locais
- **Pooling:** Max/Average pooling reduz dimensões (invariância a translação)
- **Ativação (ReLU):** Não-linearidade f(x) = max(0, x)
- **Dropout:** Regularização para evitar overfitting

### 1.2 Aplicação em Imagens Médicas

**Vantagens de CNNs para radiografias:**

- ✅ Aprende features hierárquicas automaticamente
- ✅ Invariante a pequenas translações/rotações
- ✅ Compartilhamento de pesos (eficiência)
- ✅ Estado-da-arte em classificação de imagens

**Desafios:**

- Dataset médico limitado (milhares vs milhões de imagens gerais)
- Necessidade de aumento de dados (data augmentation)
- Overfitting em datasets pequenos

---

## 2. Transfer Learning e ResNet50

### 2.1 Transfer Learning

**Conceito:** Reutilizar pesos de modelo pré-treinado em dataset grande (ImageNet) e fine-tunar para tarefa específica (pneumonia).

```
┌──────────────────────────────────────────────────────┐
│          Transfer Learning Pipeline                  │
└──────────────────────────────────────────────────────┘

ImageNet (1.4M imagens)  →  ResNet50 pré-treinado
                                    ↓
                           [Congela camadas iniciais]
                                    ↓
                        [Fine-tune últimas camadas]
                                    ↓
                    Dataset Pneumonia (5K imagens)
                                    ↓
                         Modelo especializado
```

**Vantagens:**

- Convergência mais rápida (menos épocas)
- Melhor performance com poucos dados
- Features genéricas (bordas, texturas) já aprendidas

### 2.2 Arquitetura ResNet50

**Residual Networks (He et al., 2015):**

Inovação principal: **Conexões residuais** (skip connections)

```
       x (input)
        │
        ├──────────────────┐ (shortcut)
        │                  │
        ▼                  │
    Conv 1x1              │
        │                  │
        ▼                  │
    Conv 3x3              │
        │                  │
        ▼                  │
    Conv 1x1              │
        │                  │
        ▼                  │
        ⊕ ←────────────────┘ (element-wise addition)
        │
        ▼
     ReLU
        │
        ▼
    F(x) + x (output)
```

**Benefícios:**

- Resolve problema de vanishing gradient
- Permite treinar redes muito profundas (50+ camadas)
- Melhor flow de gradientes durante backpropagation

**ResNet50 no PneumoFinder:**

```python
# Nossa implementação
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(
    weights='imagenet',        # Pesos pré-treinados
    include_top=False,         # Remove camada final (1000 classes)
    input_shape=(224, 224, 3)
)

# Adiciona camada customizada para pneumonia
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Classificação binária

model = Model(inputs=base_model.input, outputs=output)
```

---

## 3. Técnicas de Explicabilidade

### 3.1 O Problema da Caixa-Preta

CNNs são opacas: milhões de parâmetros tornam impossível interpretar decisões manualmente.

**Técnicas de explicabilidade existentes:**

| Técnica           | Tipo               | Vantagens          | Limitações              |
| ----------------- | ------------------ | ------------------ | ----------------------- |
| **Saliency Maps** | Gradient-based     | Simples            | Ruído visual            |
| **Occlusion**     | Perturbation-based | Intuitivo          | Computacionalmente caro |
| **Grad-CAM**      | Gradient-weighted  | Preciso, eficiente | Apenas visualização     |
| **LRP**           | Decomposition      | Detalhado          | Complexo                |

### 3.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

**Selvaraju et al. (2017)** - ICCV Best Paper

**Ideia central:** Usar gradientes da camada convolucional final para identificar regiões importantes.

**Algoritmo:**

```python
# Pseudo-código Grad-CAM
def gradcam(model, image, target_class):
    # 1. Forward pass
    conv_output, predictions = model(image)

    # 2. Backward pass para classe alvo
    loss = predictions[target_class]
    gradients = compute_gradients(loss, conv_output)

    # 3. Pesos = média global dos gradientes
    weights = global_average_pool(gradients)

    # 4. Combinação linear das feature maps
    heatmap = sum(weights[i] * conv_output[i] for i in range(channels))

    # 5. ReLU (apenas ativações positivas)
    heatmap = relu(heatmap)

    # 6. Normaliza para [0, 1]
    heatmap = normalize(heatmap)

    return heatmap
```

**Visualização:**

```
Original Image    Grad-CAM Heatmap       Overlay
┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │
│    🫁     │ →   │  🔴🔴     │ →   │  🔴🫁     │
│           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘
Radiografia     Regiões ativas   Sobreposição
```

**Nossa implementação:**

- Camada alvo: `conv5_block3_3_conv` (última camada convolucional do ResNet50)
- Upsampling: Bilinear para 224x224
- Colormap: Vermelho (alta ativação) → Transparente (baixa ativação)

### 3.3 Limitações do Grad-CAM

❌ **Apenas visual:** Não fornece explicação textual  
❌ **Requer interpretação:** Médicos ainda precisam analisar heatmap  
❌ **Sem contexto:** Não relaciona com terminologia médica

**Solução:** Integração com LLM multimodal para gerar descrições clínicas.

---

## 4. Modelos de Linguagem Multimodais

### 4.1 Evolução dos LLMs

**Timeline:**

- **2018:** BERT (Google) - 340M parâmetros
- **2020:** GPT-3 (OpenAI) - 175B parâmetros
- **2022:** GPT-4 (OpenAI) - ~1.7T parâmetros (estimativa)
- **2023:** LLaMA (Meta) - 7B-65B parâmetros (open-source)
- **2023:** LLaVA (UW-Madison) - Multimodal baseado em LLaMA

### 4.2 Arquitetura CLIP

**CLIP** (Contrastive Language-Image Pretraining) - OpenAI (2021)

**Treinamento contrastivo:**

```
Batch de 32,768 pares (imagem, texto)
        ↓
┌──────────────┐         ┌──────────────┐
│ Image Encoder│         │ Text Encoder │
│  (Vision     │         │  (Language   │
│  Transformer)│         │  Transformer)│
└──────┬───────┘         └──────┬───────┘
       │                        │
       ▼                        ▼
   Image Emb                Text Emb
   (512-dim)                (512-dim)
       │                        │
       └────────────┬───────────┘
                    ▼
              Cosine Similarity
                    ▼
         Maximize for correct pairs
         Minimize for incorrect pairs
```

**Resultado:** Embeddings alinhados entre visão e linguagem.

### 4.3 Arquitetura LLaVA

**LLaVA** (Large Language and Vision Assistant) - Liu et al. (2023)

**Componentes:**

```
┌────────────────────────────────────────────────────┐
│              LLaVA Architecture                     │
└────────────────────────────────────────────────────┘

Image (224x224)                    Text Prompt
      │                                  │
      ▼                                  │
┌────────────┐                          │
│   CLIP     │                           │
│   Vision   │ → Image Embeddings        │
│   Encoder  │    (576 tokens)           │
└─────┬──────┘                           │
      │                                  │
      └──────────┬───────────────────────┘
                 ▼
        ┌────────────────┐
        │  Projection    │ (Linear layer)
        │  Matrix (W)    │
        └────────┬───────┘
                 │
                 ▼
        [Combined Sequence]
                 │
                 ▼
        ┌────────────────┐
        │   Vicuna 7B    │
        │   (LLM)        │
        │   13B layers   │
        └────────┬───────┘
                 │
                 ▼
        Generated Text
    "The X-ray shows..."
```

**Treinamento:**

1. **Stage 1:** Congela CLIP e Vicuna, treina apenas projeção (W)
   - Dataset: 595K pares (imagem, legenda) do LAION
2. **Stage 2:** Fine-tune end-to-end com instruções multimodais
   - Dataset: 150K instruções geradas com GPT-4

**Vantagens sobre GPT-4V:**

- ✅ Open-source (pesos públicos)
- ✅ Roda localmente (sem custos de API)
- ✅ Privacidade (dados não saem do servidor)
- ✅ Customizável (fine-tuning possível)

**Trade-offs:**

- ❌ Menor que GPT-4 (7B vs ~1.7T parâmetros)
- ❌ Performance ligeiramente inferior (~90% da GPT-4V)
- ❌ Latência maior (10-15s vs 2-3s)

---

## 5. LLMs Especializados em Medicina

### 5.1 A Necessidade de Modelos Médicos

LLMs generalistas (GPT, LLaMA, Vicuna) apresentam limitações em domínios médicos:

| Problema | Exemplo | Impacto |
|----------|---------|---------|
| **Terminologia imprecisa** | "Mancha branca" vs "consolidação" | Comunicação inadequada |
| **Hallucinations** | Inventa achados não presentes | Risco clínico |
| **Falta de grounding** | Não cita guidelines | Baixa confiabilidade |
| **Generalização excessiva** | "Pode ser pneumonia" | Não-acionável |

**Solução:** Modelos treinados especificamente em dados médicos.

### 5.2 LLaVA-Med

**LLaVA-Med** (Li et al., 2023) - Microsoft Research

LLaVA-Med é uma versão do LLaVA especializada em imagens médicas através de treinamento em datasets biomédicos.

| Característica | Valor |
|---------------|-------|
| **Modelo** | `microsoft/llava-med-v1.5-mistral-7b` |
| **Base LLM** | Mistral-7B (não Vicuna) |
| **Vision Encoder** | CLIP ViT-L/14 |
| **Dataset de treino** | 600K+ pares imagem-texto médicos |
| **Especialidades** | Radiologia, patologia, dermatologia, oftalmologia |

**Treinamento em 3 estágios:**

```
┌────────────────────────────────────────────────────────────┐
│           LLaVA-Med Training Pipeline                       │
└────────────────────────────────────────────────────────────┘

Stage 1: Biomedical Figure-Caption Alignment
├─→ Dataset: PMC-15M (figuras de papers médicos)
├─→ Objetivo: Alinhar visão com vocabulário médico
└─→ Resultado: Modelo entende anatomia básica

Stage 2: Medical Visual Question Answering
├─→ Dataset: VQA-RAD, PathVQA, SLAKE
├─→ Objetivo: Responder perguntas sobre imagens médicas
└─→ Resultado: Modelo descreve achados específicos

Stage 3: Instruction Tuning Médico
├─→ Dataset: 60K instruções médicas (GPT-4 generated)
├─→ Objetivo: Seguir instruções clínicas complexas
└─→ Resultado: Modelo gera laudos profissionais
```

**Vantagens sobre LLaVA genérico:**

| Aspecto | LLaVA 7B | LLaVA-Med |
|---------|---------|-----------|
| Vocabulário | "White area" | "Consolidation" |
| Anatomia | "Lung" | "Right lower lobe" |
| Achados | Vago | Específico (air bronchogram) |
| Hallucinations | Frequentes | Reduzidas |
| Acurácia (nosso teste) | ~70% | **80%** |

### 5.3 BioMistral-7B

**BioMistral** (Labrak et al., 2024) - Especialização biomédica do Mistral-7B

| Característica | Valor |
|---------------|-------|
| **Modelo** | `BioMistral/BioMistral-7B` |
| **Base** | Mistral-7B-v0.1 |
| **Treinamento adicional** | PubMed abstracts, guidelines médicas |
| **Parâmetros** | 7.24B |
| **Contexto** | 32K tokens |

**Por que BioMistral para geração de laudos?**

```
┌────────────────────────────────────────────────────────────┐
│           Comparação de Saída de Texto                      │
└────────────────────────────────────────────────────────────┘

Input: "Describe findings: bilateral opacities in lower lobes"

Mistral-7B (genérico):
"There are some white areas in the lungs that could indicate
an infection or other problem."

BioMistral-7B (especializado):
"Findings: Bilateral airspace opacities in the lower lobes
with air bronchograms, suggestive of community-acquired
pneumonia. No pleural effusion. Heart size is normal.
Impression: Pneumonia. Clinical correlation recommended."
```

**Vantagens:**

- ✅ Formato de laudo radiológico (Findings / Impression)
- ✅ Terminologia padronizada (ACR guidelines)
- ✅ Achados negativos relevantes ("No pleural effusion")
- ✅ Recomendações clínicas ("Clinical correlation")

### 5.4 Pipeline de 2 Estágios no PneumoFinder

O PneumoFinder combina LLaVA-Med e BioMistral em pipeline sequencial:

```
┌────────────────────────────────────────────────────────────┐
│              Pipeline de 2 Estágios                         │
└────────────────────────────────────────────────────────────┘

Radiografia de Tórax
        │
        ▼
┌──────────────────────────────────────┐
│ STAGE 1: Análise Visual              │
│ Modelo: LLaVA-Med                    │
│                                       │
│ Input: Imagem + RAG context          │
│ Output: Descrição de achados visuais │
│         "Bilateral consolidations    │
│          in lower lobes..."          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ STAGE 2: Síntese de Laudo            │
│ Modelo: BioMistral-7B                │
│                                       │
│ Input: Achados Stage 1 + CNN + RAG   │
│ Output: Laudo radiológico profissional│
│         "Findings: ... Impression: ...│
│          Clinical correlation..."    │
└──────────────────────────────────────┘
```

**Justificativa da separação:**

1. **Especialização:** LLaVA-Med é melhor em visão, BioMistral em texto
2. **Memória:** Carregamento sequencial permite rodar em 24GB
3. **Qualidade:** Cada modelo faz o que sabe fazer melhor

---

## 6. Retrieval-Augmented Generation (RAG)

### 6.1 O Problema das Hallucinations

LLMs podem "inventar" informações que parecem corretas mas são falsas:

```
Prompt: "Describe the pneumonia in this X-ray"

Hallucination:
"The X-ray shows Legionella pneumonia with characteristic
cavitation and pleural involvement..."  ← INVENTADO!

Problema: O modelo não sabe o que REALMENTE está na imagem
```

### 6.2 Solução: RAG (Retrieval-Augmented Generation)

**RAG** (Lewis et al., 2020) combina recuperação de conhecimento com geração:

```
┌────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
└────────────────────────────────────────────────────────────┘

Query: "pneumonia findings"
        │
        ▼
┌──────────────────────────────────────┐
│ 1. RETRIEVAL                         │
│                                       │
│ Vector Store (ChromaDB)              │
│ ┌─────────────────────────────────┐  │
│ │ 31 Medical Guidelines:          │  │
│ │ - ACR Appropriateness Criteria  │  │
│ │ - Fleischner Society 2017       │  │
│ │ - WHO Pneumonia Guidelines      │  │
│ └─────────────────────────────────┘  │
│                                       │
│ Similarity Search → Top-K relevantes │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 2. AUGMENTATION                      │
│                                       │
│ Prompt = Original + Retrieved Context│
│                                       │
│ "Describe findings in this X-ray.    │
│  REFERENCE:                          │
│  [Fleischner 2017] Lobar consolida- │
│  tion with air bronchograms suggests │
│  bacterial pneumonia..."             │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 3. GENERATION                        │
│                                       │
│ LLM gera resposta GROUNDED nos       │
│ documentos recuperados               │
│                                       │
│ Output: "This X-ray shows lobar      │
│ consolidation consistent with        │
│ bacterial pneumonia (per Fleischner  │
│ 2017 guidelines)."                   │
└──────────────────────────────────────┘
```

### 6.3 Base de Conhecimento do PneumoFinder

| Coleção | Documentos | Conteúdo |
|---------|-----------|----------|
| **medical_guidelines** | 31 | Guidelines ACR, Fleischner, WHO |
| **sample_reports** | 50+ | Laudos exemplo (normal/pneumonia) |
| **medical_terminology** | 200+ | Termos técnicos EN↔PT |

**Exemplo de guideline:**

```
[Fleischner Society 2017 - Pneumonia Patterns]

Lobar consolidation with air bronchograms typically indicates
bacterial pneumonia, most commonly Streptococcus pneumoniae.

Interstitial pattern with ground-glass opacities suggests
viral or atypical pneumonia (Mycoplasma, Chlamydia).

Cavitation should raise suspicion for anaerobic infection,
tuberculosis, or necrotizing pneumonia.
```

### 6.4 Filtragem por Relevância

ChromaDB usa distância L2 para similaridade. O PneumoFinder implementa thresholds:

```python
# src/rag/retriever.py

DEFAULT_RELEVANCE_THRESHOLD = 1.2  # Max L2 distance
STRICT_RELEVANCE_THRESHOLD = 0.8   # For high-precision

def _filter_by_relevance(self, results, threshold):
    """Remove resultados com baixa similaridade."""
    return [r for r in results if r["distance"] <= threshold]
```

**Benefícios:**

- ✅ Evita contexto irrelevante no prompt
- ✅ Reduz ruído que pode confundir o LLM
- ✅ Melhora qualidade das respostas

### 6.5 RAG no Pipeline de 2 Estágios

```
Stage 1 (LLaVA-Med):
├─→ RAG: Guidelines de radiologia (como descrever achados)
└─→ Output: Descrição visual precisa

Stage 2 (BioMistral):
├─→ RAG: Guidelines + Sample Reports (formato de laudo)
└─→ Output: Laudo profissional padronizado
```

---

## 7. Embeddings e Busca Semântica

### 5.1 Representação Vetorial de Texto

**Embeddings** convertem texto em vetores numéricos que capturam significado semântico:

```
Texto: "Consolidação pulmonar bilateral"
         ↓
Sentence Transformer (all-MiniLM-L6-v2)
         ↓
Vetor: [0.23, -0.45, 0.12, ..., 0.67] (384 dimensões)
```

**Propriedade matemática:**

```
similarity("pneumonia", "infiltrado") > similarity("pneumonia", "fratura")
```

### 5.2 Sentence-Transformers

**Modelo:** `all-MiniLM-L6-v2`

- Baseado em BERT (Bidirectional Encoder Representations from Transformers)
- 22M parâmetros (leve e rápido)
- 384 dimensões de saída
- Treinado com 1B+ pares de sentenças

**Geração de embeddings:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "The model detected pneumonia with 87% confidence"
embedding = model.encode(text)  # shape: (384,)
```

### 5.3 ChromaDB

**ChromaDB** é um banco de dados vetorial otimizado para busca semântica.

**Índice HNSW** (Hierarchical Navigable Small World):

```
┌─────────────────────────────────────────┐
│     HNSW Index Structure                │
└─────────────────────────────────────────┘

Layer 2:  ○────○          Sparse
           │    │
Layer 1:  ○─○──○─○        Medium
           │││││││
Layer 0:  ○○○○○○○○○       Dense (todos os vetores)
         [Greedy search top-down]
```

**Vantagens:**

- ✅ Busca em O(log N) (sub-linear)
- ✅ Alta recall (>95% dos k-vizinhos mais próximos)
- ✅ Baixo uso de memória (compressão HNSW)

**Métricas de similaridade:**

- **Cosine:** cos(θ) = (A · B) / (||A|| ||B||) [Usado no PneumoFinder]
- **Euclidean:** ||A - B||
- **Dot product:** A · B

### 5.4 Pipeline de Busca Semântica

```
Query: "Infiltrados no lobo inferior direito"
        ↓
1. Gera embedding (384-dim)
        ↓
2. Busca no ChromaDB (cosine similarity)
        ↓
3. Retorna top-k diagnósticos similares
        ↓
[
  {id: 42, diagnosis: "PNEUMONIA", similarity: 0.92},
  {id: 17, diagnosis: "PNEUMONIA", similarity: 0.87},
  ...
]
```

**Exemplo real:**

```python
# Query
query = "consolidation in lower right lung field"

# Results
[
  {
    "diagnosis_id": 2,
    "description": "...potential lung abnormalities in the lower right...",
    "similarity": 0.89  # Muito similar!
  },
  {
    "diagnosis_id": 15,
    "description": "...bilateral infiltrates...",
    "similarity": 0.71  # Moderadamente similar
  }
]
```

---

## 8. Integração dos Conceitos

### 8.1 Pipeline Completo do PneumoFinder v3

```
┌────────────────────────────────────────────────────────────────┐
│           Fundamentos Teóricos Integrados (2 Estágios)          │
└────────────────────────────────────────────────────────────────┘

Radiografia de Tórax
        │
        ▼
┌────────────────────────────────────────┐
│ CNN + Grad-CAM                         │ → Transfer Learning (ResNet50)
│ (Seções 1, 2, 3)                       │ → Classificação + Heatmap
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ RAG (Seção 6)                          │ → Recupera guidelines relevantes
│                                         │ → ChromaDB + 31 documentos
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ STAGE 1: LLaVA-Med (Seção 5.2)         │ → Vision model médico
│                                         │ → Descrição visual + RAG
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ STAGE 2: BioMistral (Seção 5.3)        │ → Text model biomédico
│                                         │ → Laudo profissional + RAG
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Embeddings + Busca (Seção 7)           │ → Sentence-Transformers
│                                         │ → ChromaDB HNSW
└────────────────────────────────────────┘
```

### 8.2 Contribuições Teóricas

Este trabalho combina múltiplas técnicas estado-da-arte:

1. **CNN (Deep Learning clássico)** → Alta acurácia (87.3%)
2. **Grad-CAM (XAI)** → Explicabilidade visual
3. **LLMs Médicos Especializados** → LLaVA-Med + BioMistral
4. **RAG (Retrieval-Augmented Generation)** → Grounding com guidelines
5. **Embeddings vetoriais** → Busca semântica inteligente

**Novidade:** Pipeline de 2 estágios com modelos biomédicos especializados e RAG para diagnóstico médico explicável e confiável.

---

## 9. Trabalhos Relacionados

### 9.1 Comparação com Estado-da-Arte

| Trabalho                            | CNN         | Explicabilidade | LLM           | RAG | Open-Source |
| ----------------------------------- | ----------- | --------------- | ------------- | --- | ----------- |
| **Rajpurkar et al. (2018) CheXNet** | ✅ DenseNet | ❌              | ❌            | ❌  | ❌          |
| **Irvin et al. (2019) CheXpert**    | ✅ DenseNet | ❌              | ❌            | ❌  | ✅ (Dados)  |
| **Selvaraju et al. (2017)**         | ✅ VGG      | ✅ Grad-CAM     | ❌            | ❌  | ✅          |
| **Liu et al. (2023) LLaVA**         | N/A         | N/A             | ✅ Genérico   | ❌  | ✅          |
| **Li et al. (2023) LLaVA-Med**      | N/A         | N/A             | ✅ Médico     | ❌  | ✅          |
| **Labrak et al. (2024) BioMistral** | N/A         | N/A             | ✅ Biomédico  | ❌  | ✅          |
| **PneumoFinder v3 (Este trabalho)** | ✅ ResNet50 | ✅ Grad-CAM     | ✅ LLaVA-Med + BioMistral | ✅ 31 guidelines | ✅ Completo |

### 9.2 Diferenciais

- ✅ **Pipeline especializado:** LLaVA-Med (visão) + BioMistral (texto)
- ✅ **RAG grounding:** 31 guidelines médicas para evitar hallucinations
- ✅ **Deployment local:** Privacidade garantida, sem API externa
- ✅ **Apple Silicon:** Funciona em M4 Pro 24GB (device_map="auto")
- ✅ **Open-source:** Código completo reprodutível

---

## 10. Referências

1. **He et al. (2015).** "Deep Residual Learning for Image Recognition". CVPR 2016.

2. **Selvaraju et al. (2017).** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". ICCV 2017.

3. **Radford et al. (2021).** "Learning Transferable Visual Models From Natural Language Supervision". ICML 2021.

4. **Liu et al. (2023).** "Visual Instruction Tuning". NeurIPS 2023.

5. **Li et al. (2023).** "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day". NeurIPS 2023.

6. **Labrak et al. (2024).** "BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains". arXiv:2402.10373.

7. **Lewis et al. (2020).** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS 2020.

8. **Jiang et al. (2023).** "Mistral 7B". arXiv:2310.06825.

9. **Rajpurkar et al. (2018).** "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays". arXiv:1711.05225.

10. **Reimers & Gurevych (2019).** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". EMNLP 2019.

11. **Malkov & Yashunin (2018).** "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". IEEE TPAMI.

---

**Documento atualizado para TCC - PneumoFinder v3.0**
**Última atualização:** Fevereiro 2026
