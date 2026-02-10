# RAG System - PneumoFinder

## Overview

O sistema RAG (Retrieval-Augmented Generation) do PneumoFinder usa ChromaDB para armazenar e buscar conhecimento médico que é injetado nos prompts dos LLMs para melhorar a qualidade das respostas.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│  Query (diagnosis/findings)                                  │
│           ↓                                                  │
│  Embedding Model (sentence-transformers)                     │
│           ↓                                                  │
│  ChromaDB Semantic Search                                    │
│           ↓                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ medical_guidelines│  │sample_reports   │  │terminology  │ │
│  │ (31 docs)        │  │(12 docs)        │  │(147 terms)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           ↓                                                  │
│  Relevance Filtering (threshold: 1.2)                        │
│           ↓                                                  │
│  Diversity Ranking (dedupe similar docs)                     │
│           ↓                                                  │
│  RAG Context → Injected into LLM prompts                     │
└─────────────────────────────────────────────────────────────┘
```

## Collections

### 1. medical_guidelines (31 documentos)

Guidelines radiológicas baseadas em evidências:

| Categoria | Exemplos |
|-----------|----------|
| `pathology` | Lobar Pneumonia, Bronchopneumonia, Interstitial Pneumonia, Aspiration Pneumonia, Viral Pneumonia, HAP/VAP |
| `normal` | Normal Chest X-ray Criteria, Normal Variants |
| `anatomy` | Left vs Right Lung, Lung Zones, Hilar Anatomy |
| `signs` | Air Bronchogram, Silhouette Sign, Spine Sign, Deep Sulcus Sign |
| `differential` | Atelectasis vs Pneumonia, Pulmonary Edema vs Pneumonia, Mass vs Consolidation |
| `severity` | CURB-65 Score, Radiographic Progression Indicators |
| `complications` | Parapneumonic Effusion, Lung Abscess, ARDS |
| `management` | Follow-up Imaging, When to Recommend CT |
| `technical` | Technical Quality Assessment, AP vs PA Differences |

**Fontes:**
- American College of Radiology (ACR)
- Fleischner Society
- Radiological Society of North America (RSNA)
- British Thoracic Society (BTS)
- IDSA/ATS Guidelines

### 2. sample_reports (12 documentos)

Laudos radiológicos de exemplo em inglês e português:

| Categoria | Idioma | Quantidade |
|-----------|--------|------------|
| Normal | EN | 3 |
| Normal | PT-BR | 2 |
| Pneumonia | EN | 5 |
| Pneumonia | PT-BR | 2 |

### 3. medical_terminology (147 termos)

Mapeamento EN→PT de terminologia médica:

| Categoria | Exemplos |
|-----------|----------|
| Anatomical | right lung → pulmão direito, costophrenic angle → ângulo costofrênico |
| Pathology | consolidation → consolidação, opacity → opacidade |
| Clinical | pneumonia → pneumonia, follow-up → acompanhamento |
| Negative | no evidence of → sem evidência de, clear lungs → pulmões limpos |

---

## Embedding Models Benchmark

Benchmark realizado em 2026-02-10 comparando modelos de embedding para retrieval médico.

### Metodologia

- **Queries de teste:** 10 consultas médicas típicas
- **Documentos:** 15 guidelines da knowledge base
- **Métricas:**
  - Top-1 Score: Similaridade do melhor resultado
  - Top-3 Score: Média de similaridade dos 3 melhores
  - Recall@3: % de queries com resultado esperado no top 3

### Resultados

| Modelo | Dimensões | Load Time | Top-1 Score | Recall@3 |
|--------|-----------|-----------|-------------|----------|
| `all-MiniLM-L6-v2` (atual) | 384 | 3.8s | 0.54 | **90%** |
| `S-PubMedBert-MS-MARCO` | 768 | 19.1s | **0.91** | **90%** |
| `all-mpnet-base-v2` | 768 | 11.9s | 0.56 | 80% |

### Análise

1. **S-PubMedBert-MS-MARCO** (modelo médico):
   - Scores de similaridade **68% maiores** (0.91 vs 0.54)
   - Melhor entendimento de terminologia médica
   - Trade-off: Load time 5x maior (19s vs 4s)

2. **all-MiniLM-L6-v2** (atual):
   - Rápido para carregar
   - Bom Recall@3 (90%)
   - Scores de similaridade mais baixos

3. **all-mpnet-base-v2**:
   - Pior performance geral (80% Recall)
   - Não recomendado para uso médico

### Recomendação

Para produção com foco em qualidade médica, considerar trocar para `S-PubMedBert-MS-MARCO`. O aumento no tempo de carregamento (apenas na inicialização) é compensado pela melhor precisão semântica.

**Configuração via variável de ambiente (TODO):**
```bash
export EMBEDDING_MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
```

---

## Configuração

### Arquivos Principais

| Arquivo | Propósito |
|---------|-----------|
| `src/rag/vector_store.py` | ChromaDB wrapper |
| `src/rag/retriever.py` | Interface de retrieval com filtering |
| `data/knowledge_base/guidelines/radiology_guidelines.py` | Guidelines médicas |
| `data/knowledge_base/terminology/medical_terms_en_pt.py` | Terminologia EN→PT |
| `scripts/populate_knowledge_base.py` | Script de população |

### Parâmetros de Retrieval

```python
# src/rag/retriever.py
DEFAULT_RELEVANCE_THRESHOLD = 1.2  # Max distance para considerar relevante
STRICT_RELEVANCE_THRESHOLD = 0.8   # Para queries de alta precisão
```

### Repopular Knowledge Base

```bash
python scripts/populate_knowledge_base.py
```

---

## Uso no Pipeline

### Stage 1 (Vision Analysis)

```python
rag_context = retriever.build_rag_context(
    diagnosis="pneumonia",
    findings=None,
    include_guidelines=True,
    include_reports=False,  # Sem findings ainda
)
```

### Stage 2 (Medical Text)

```python
rag_context = retriever.build_rag_context(
    diagnosis="pneumonia",
    findings="bilateral consolidation with air bronchograms",
    include_guidelines=True,
    include_reports=True,  # Agora inclui reports similares
    prefer_english_reports=True,  # LLM é em inglês
)
```

### Exemplo de RAG Context

```
=== RELEVANT GUIDELINES ===
[Lobar Pneumonia - Classic Presentation]
- Lobar pneumonia presents as homogeneous consolidation involving an entire
  lobe, with sharp demarcation at fissures. Air bronchograms are typically
  present. Most commonly affects lower lobes.

[Air Bronchogram Sign - Consolidation Indicator]
- Air bronchograms appear as dark, branching air-filled bronchi visible
  against opacified lung parenchyma. Indicates alveolar consolidation.

=== SIMILAR REPORTS ===
[Right Lower Lobe Pneumonia - English]
- There is consolidation in the right lower lobe with air bronchograms.
  The remainder of the lungs is clear. No pleural effusion.
```

---

## Melhorias Futuras

1. **Hybrid Search**: Combinar busca semântica com BM25 (keyword)
2. **Medical Embeddings**: Trocar para S-PubMedBert-MS-MARCO
3. **Query Expansion**: Expandir queries com sinônimos médicos
4. **Feedback Loop**: Rastrear quais contextos melhoram diagnósticos
5. **Dynamic Updates**: Endpoint para adicionar novas guidelines
