# 02 - Fundamentos Teóricos

## Sumário

- [1. Redes Neurais Convolucionais](#1-redes-neurais-convolucionais)
- [2. Transfer Learning e ResNet50](#2-transfer-learning-e-resnet50)
- [3. Técnicas de Explicabilidade](#3-técnicas-de-explicabilidade)
- [4. Modelos de Linguagem Multimodais](#4-modelos-de-linguagem-multimodais)
- [5. Embeddings e Busca Semântica](#5-embeddings-e-busca-semântica)

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

### 4.4 Ollama

**Ollama** é um servidor que simplifica deployment de LLMs locais:

```bash
# Instala Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Baixa modelo LLaVA 7B
ollama pull llava:7b

# Inicia servidor
ollama serve  # Roda na porta 11434
```

**API REST simples:**

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llava:7b",
    "prompt": "Describe this medical image",
    "images": ["base64_encoded_image"]
  }'
```

---

## 5. Embeddings e Busca Semântica

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

## 6. Integração dos Conceitos

### 6.1 Pipeline Completo do PneumoFinder

```
┌──────────────────────────────────────────────────────────┐
│           Fundamentos Teóricos Integrados                 │
└──────────────────────────────────────────────────────────┘

Radiografia
    ↓
┌──────────────┐
│ CNN          │ → Transfer Learning (ResNet50)
│ (Seção 1, 2) │ → Classificação: PNEUMONIA (87%)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Grad-CAM     │ → Explicabilidade Visual (Seção 3)
│ (Seção 3)    │ → Heatmap + Overlay
└──────┬───────┘
       │
       ▼
┌───────────────┐
│ LLM Multimodal│ → CLIP + Vicuna (Seção 4)
│ (Seção 4)     │ → Descrição clínica em linguagem natural
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ Embeddings    │ → Sentence-Transformers (Seção 5)
│ (Seção 5)     │ → Busca semântica no ChromaDB
└───────────────┘
```

### 6.2 Contribuições Teóricas

Este trabalho combina múltiplas técnicas estado-da-arte:

1. **CNN (Deep Learning clássico)** → Alta acurácia
2. **Grad-CAM (XAI)** → Explicabilidade visual
3. **LLM Multimodal** → Explicações em linguagem natural
4. **Embeddings vetoriais** → Busca semântica inteligente

**Novidade:** Integração end-to-end dessas técnicas em sistema unificado para diagnóstico médico explicável.

---

## 7. Trabalhos Relacionados

### 7.1 Comparação com Estado-da-Arte

| Trabalho                            | CNN         | Explicabilidade | LLM           | Busca Semântica | Open-Source |
| ----------------------------------- | ----------- | --------------- | ------------- | --------------- | ----------- |
| **Rajpurkar et al. (2018) CheXNet** | ✅ DenseNet | ❌              | ❌            | ❌              | ❌          |
| **Irvin et al. (2019) CheXpert**    | ✅ DenseNet | ❌              | ❌            | ❌              | ✅ (Dados)  |
| **Selvaraju et al. (2017)**         | ✅ VGG      | ✅ Grad-CAM     | ❌            | ❌              | ✅          |
| **Liu et al. (2023) LLaVA**         | N/A         | N/A             | ✅ Multimodal | ❌              | ✅          |
| **PneumoFinder (Este trabalho)**    | ✅ ResNet50 | ✅ Grad-CAM     | ✅ LLaVA 7B   | ✅ ChromaDB     | ✅ Completo |

### 7.2 Diferenciais

- ✅ Único com integração completa CNN + Grad-CAM + LLM + Busca Semântica
- ✅ Deployment local (privacidade garantida)
- ✅ Open-source com Docker (reprodutível)
- ✅ Foco em explicabilidade prática para uso clínico

---

## 8. Referências

1. **He et al. (2015).** "Deep Residual Learning for Image Recognition". CVPR 2016.

2. **Selvaraju et al. (2017).** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". ICCV 2017.

3. **Radford et al. (2021).** "Learning Transferable Visual Models From Natural Language Supervision". ICML 2021.

4. **Liu et al. (2023).** "Visual Instruction Tuning". NeurIPS 2023.

5. **Rajpurkar et al. (2018).** "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays". arXiv:1711.05225.

6. **Reimers & Gurevych (2019).** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". EMNLP 2019.

7. **Malkov & Yashunin (2018).** "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". IEEE TPAMI.

---

**Documento elaborado para TCC/Monografia - PneumoFinder v2.0**  
**Última atualização:** Dezembro 2026
