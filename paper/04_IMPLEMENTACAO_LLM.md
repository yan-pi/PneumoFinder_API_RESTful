# 04 - Implementação do Pipeline de LLMs Médicos

## Sumário

- [1. Visão Geral](#1-visão-geral)
- [2. Arquitetura do Pipeline de 2 Estágios](#2-arquitetura-do-pipeline-de-2-estágios)
- [3. LLaVA-Med: Análise Visual Médica](#3-llava-med-análise-visual-médica)
- [4. BioMistral-7B: Geração de Laudos](#4-biomistral-7b-geração-de-laudos)
- [5. Sistema RAG de Conhecimento Médico](#5-sistema-rag-de-conhecimento-médico)
- [6. Gerenciamento de Memória](#6-gerenciamento-de-memória)
- [7. Prompt Engineering Médico](#7-prompt-engineering-médico)
- [8. Pipeline Completo: Código](#8-pipeline-completo-código)
- [9. Resultados e Métricas](#9-resultados-e-métricas)
- [10. Conclusões](#10-conclusões)

---

## 1. Visão Geral

### 1.1 Problema Abordado

Redes Neurais Convolucionais (CNNs) para classificação médica enfrentam um problema crítico de **explicabilidade**: embora alcancem alta acurácia, funcionam como "caixas-pretas", tornando difícil para profissionais de saúde compreenderem o raciocínio por trás das predições.

### 1.2 Solução: Pipeline de 2 Estágios

O PneumoFinder implementa um pipeline de **2 estágios** utilizando modelos de linguagem especializados em medicina:

| Estágio | Modelo | Função |
|---------|--------|--------|
| **Stage 1: Vision** | LLaVA-Med (7B) | Análise visual da radiografia com descrição de achados |
| **Stage 2: Medical** | BioMistral-7B | Síntese de laudo radiológico profissional |

Ambos os estágios são enriquecidos com **RAG (Retrieval-Augmented Generation)** utilizando 31 guidelines médicas.

### 1.3 Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pipeline Médico de 2 Estágios                         │
└─────────────────────────────────────────────────────────────────────────┘

Radiografia (Input)
      │
      ▼
┌─────────────────┐
│  CNN (ResNet50) │ → Diagnóstico: PNEUMONIA (87.3%)
│   + Grad-CAM    │ → Heatmap de atenção
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ESTÁGIO 1: Análise Visual (LLaVA-Med)                                   │
│                                                                          │
│ Modelo: microsoft/llava-med-v1.5-mistral-7b                             │
│                                                                          │
│ Inputs:                                                                  │
│ • Radiografia original                                                   │
│ • Contexto RAG (guidelines de radiologia)                               │
│                                                                          │
│ Output:                                                                  │
│ "The chest X-ray shows bilateral lower lobe consolidations consistent   │
│  with pneumonia. No pleural effusion or pneumothorax is present."       │
└────────┬────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ESTÁGIO 2: Geração de Laudo (BioMistral-7B)                             │
│                                                                          │
│ Modelo: BioMistral/BioMistral-7B                                        │
│                                                                          │
│ Inputs:                                                                  │
│ • Achados do Estágio 1                                                  │
│ • Diagnóstico CNN + confiança                                           │
│ • Contexto RAG (guidelines + laudos exemplo)                            │
│                                                                          │
│ Output:                                                                  │
│ "Findings: Bilateral lower lobe airspace opacities with air             │
│  bronchograms, consistent with pneumonia. Heart size is normal.         │
│  Impression: Community-acquired pneumonia. Clinical correlation         │
│  recommended."                                                           │
└────────┬────────────────────────────────────────────────────────────────┘
         │
         ▼
    Laudo Final (JSON Response)
```

---

## 2. Arquitetura do Pipeline de 2 Estágios

### 2.1 Por que 2 Estágios?

A arquitetura de 2 estágios foi escolhida por:

| Vantagem | Descrição |
|----------|-----------|
| **Especialização** | LLaVA-Med é treinado especificamente em imagens médicas |
| **Qualidade** | BioMistral gera texto biomédico de alta qualidade |
| **Grounding** | RAG em cada estágio previne alucinações |
| **Memória** | Carregamento sequencial permite rodar em 24GB |

### 2.2 Comparação com Abordagem Anterior

| Aspecto | Pipeline Anterior (3 estágios) | Pipeline Atual (2 estágios) |
|---------|--------------------------------|------------------------------|
| Modelos | LLaVA 7B (genérico) + Tradução | LLaVA-Med + BioMistral |
| Tradução | Estágio separado (GPT-4) | Removido (output em inglês) |
| Latência | ~45s | ~39s |
| Acurácia | ~70% | **80%** |

### 2.3 Fluxo de Dados

```python
# Classe principal: MedicalPipeline
class MedicalPipeline:
    def analyze_xray(self, image_path: str, cnn_diagnosis: dict) -> dict:
        # Stage 1: LLaVA-Med analisa a radiografia
        vision_result = self.stage1_vision_analysis(image_path, cnn_diagnosis)

        # Stage 2: BioMistral gera laudo profissional
        medical_result = self.stage2_medical_text(vision_result, cnn_diagnosis)

        return {
            "stage1_vision": vision_result,
            "stage2_medical": medical_result,
            "final_report_en": medical_result["report_en"],
        }
```

---

## 3. LLaVA-Med: Análise Visual Médica

### 3.1 O que é LLaVA-Med?

**LLaVA-Med** (Large Language and Vision Assistant for Medicine) é um modelo multimodal desenvolvido pela Microsoft, especializado em análise de imagens médicas.

| Característica | Valor |
|---------------|-------|
| **Modelo** | `microsoft/llava-med-v1.5-mistral-7b` |
| **Base LLM** | Mistral-7B |
| **Vision Encoder** | CLIP ViT-L/14 |
| **Treinamento** | 600K+ pares imagem-texto médicos |
| **Especialidade** | Radiologia, patologia, dermatologia |

### 3.2 Por que LLaVA-Med?

| Critério | LLaVA 7B (genérico) | LLaVA-Med |
|----------|---------------------|-----------|
| Treinamento médico | ❌ Generalista | ✅ 600K+ imagens médicas |
| Terminologia | Imprecisa | Precisa (consolidation, infiltrate) |
| Anatomia | Básica | Detalhada (lobos, segmentos) |
| Hallucinations | Frequentes | Reduzidas |
| Acurácia (nosso teste) | ~70% | **80%** |

### 3.3 Arquitetura Interna

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLaVA-Med Architecture                               │
└─────────────────────────────────────────────────────────────────────────┘

Radiografia (336x336)                      Prompt + RAG Context
     │                                            │
     ▼                                            ▼
┌──────────────┐                         ┌──────────────┐
│  CLIP ViT-L  │                         │   Mistral    │
│  Vision      │ → Image Embeddings      │     7B       │
│  Encoder     │   (576 patches)         │     LLM      │
└──────┬───────┘                         └──────┬───────┘
       │                                        │
       │         ┌────────────────────┐         │
       └────────►│  MLP Projection    │◄────────┘
                 │  (Vision→Text)     │
                 └─────────┬──────────┘
                           │
                           ▼
                  Texto Gerado (tokens)
       "Bilateral consolidations in lower lobes..."
```

### 3.4 Configuração no Apple Silicon (MPS)

LLaVA-Med requer configurações específicas para rodar em Macs com Apple Silicon:

```python
# Correções aplicadas em llava/model/builder.py

# 1. Desabilitar Flash Attention (não suportado no MPS)
model = LlavaMistralForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",  # Substituiu use_flash_attention_2=False
    **kwargs,
)

# 2. Device mapping para split MPS + CPU
if device == "mps":
    kwargs["device_map"] = "auto"
    kwargs["max_memory"] = {"mps": "10GiB", "cpu": "20GiB"}

# 3. Manter vision tower no CPU para evitar overflow
model.model.vision_tower.to("cpu")
```

### 3.5 Código de Inferência

```python
# src/core/medical_pipeline.py - Método _run_llava_med()

def _run_llava_med(self, image_path: str, prompt: str) -> str:
    """Run LLaVA-Med inference using Python API."""

    # Fix protobuf compatibility
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # Load model (lazy loading)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        device="mps",
    )

    # Process image
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to("cpu", dtype=torch.float16)

    # Build conversation (vicuna_v1 template for Mistral base)
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)

    # Generate
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            images=image_tensor,
            max_new_tokens=512,
            temperature=0.2,
        )

    # Extract assistant response
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.split("ASSISTANT:")[-1].strip()
```

---

## 4. BioMistral-7B: Geração de Laudos

### 4.1 O que é BioMistral?

**BioMistral-7B** é um modelo de linguagem especializado em texto biomédico, baseado no Mistral-7B e continuado com literatura médica.

| Característica | Valor |
|---------------|-------|
| **Modelo** | `BioMistral/BioMistral-7B` |
| **Base** | Mistral-7B-v0.1 |
| **Treinamento** | PubMed, guidelines médicas |
| **Parâmetros** | 7.24B |
| **Contexto** | 32K tokens |

### 4.2 Por que BioMistral para Síntese?

| Critério | Mistral-7B (base) | BioMistral-7B |
|----------|-------------------|---------------|
| Vocabulário médico | Limitado | Extenso |
| Formato de laudos | Inconsistente | Padronizado |
| Recomendações | Genéricas | Clinicamente relevantes |
| Referências | Ausentes | Baseadas em guidelines |

### 4.3 Código de Carregamento

```python
# src/core/medical_pipeline.py - Método _load_biomistral()

def _load_biomistral(self) -> None:
    """Lazy-load BioMistral model with intelligent device mapping.

    Strategy: MPS has ~10GB single allocation limit, but BioMistral @ FP16 = 13GB.
    Use device_map="auto" to split across MPS + CPU/RAM intelligently.
    """
    model_id = "BioMistral/BioMistral-7B"

    self._biomistral_tokenizer = AutoTokenizer.from_pretrained(model_id)
    self._biomistral_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",  # Let transformers split intelligently
        low_cpu_mem_usage=True,
        max_memory={"mps": "10GiB", "cpu": "16GiB"},  # MPS limit + CPU fallback
    )
```

### 4.4 Geração de Laudo

```python
# src/core/medical_pipeline.py - Método stage2_medical_text()

def stage2_medical_text(self, vision_result: dict, cnn_diagnosis: dict) -> dict:
    """Stage 2: Medical text generation with BioMistral."""

    # Unload LLaVA-Med first (free MPS memory)
    self._unload_llava_med()

    # Load BioMistral
    self._load_biomistral()

    # Build prompt with RAG context
    medical_prompt = build_medical_text_prompt(
        vision_description=vision_result["findings"],
        cnn_diagnosis=cnn_diagnosis,
        use_rag=True,
        rag_context=self.rag_retriever.build_rag_context(...)
    )

    # Format with Mistral chat template
    formatted_prompt = self._biomistral_tokenizer.apply_chat_template(
        [{"role": "user", "content": medical_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Generate
    inputs = self._biomistral_tokenizer(formatted_prompt, return_tensors="pt")
    outputs = self._biomistral_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )

    return {
        "report_en": self._biomistral_tokenizer.decode(outputs[0], skip_special_tokens=True),
        "model": "BioMistral-7B",
    }
```

---

## 5. Sistema RAG de Conhecimento Médico

### 5.1 Visão Geral do RAG

O sistema RAG (Retrieval-Augmented Generation) enriquece os prompts com conhecimento médico relevante, prevenindo alucinações e garantindo terminologia correta.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Sistema RAG - Fluxo                                   │
└─────────────────────────────────────────────────────────────────────────┘

Diagnóstico CNN: "PNEUMONIA"
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MedicalRetriever.build_rag_context()                                     │
│                                                                          │
│ 1. Busca guidelines relevantes:                                         │
│    → "Lobar consolidation indicates bacterial pneumonia"                │
│    → "Air bronchograms suggest alveolar process"                        │
│                                                                          │
│ 2. Busca laudos exemplo:                                                │
│    → "Impression: Right lower lobe pneumonia"                           │
│                                                                          │
│ 3. Formata contexto estruturado                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
=== RELEVANT GUIDELINES ===
[Fleischner Society 2017]
- Lobar consolidation with air bronchograms suggests bacterial pneumonia

[ACR Appropriateness Criteria]
- Follow-up imaging recommended in 6-8 weeks for resolution

=== SIMILAR REPORTS ===
[Pneumonia - Typical Case]
- Impression: Right lower lobe consolidation, likely community-acquired pneumonia
```

### 5.2 Base de Conhecimento

| Coleção | Documentos | Conteúdo |
|---------|-----------|----------|
| **medical_guidelines** | 31 | Guidelines de radiologia (ACR, Fleischner) |
| **sample_reports** | 50+ | Laudos exemplo (normal e pneumonia) |
| **medical_terminology** | 200+ | Termos técnicos EN→PT |

### 5.3 Implementação do Retriever

```python
# src/rag/retriever.py

class MedicalRetriever:
    """Retriever for medical knowledge to ground LLM generation."""

    def __init__(self, relevance_threshold: float = 1.2):
        self.vector_store = ChromaVectorStore()
        self.relevance_threshold = relevance_threshold

    def get_relevant_guidelines(self, diagnosis: str, n_results: int = 3) -> list:
        """Retrieve relevant radiology guidelines with quality filtering."""

        results = self.vector_store.query(
            collection_name="medical_guidelines",
            query_text=diagnosis,
            n_results=n_results * 2,  # Request extra for filtering
        )

        # Apply relevance filtering (L2 distance threshold)
        guidelines = self._filter_by_relevance(results)

        # Deduplicate for diversity
        guidelines = self._deduplicate_results(guidelines)

        return guidelines[:n_results]

    def build_rag_context(
        self,
        diagnosis: str,
        findings: str | None = None,
        include_guidelines: bool = True,
        include_reports: bool = True,
    ) -> str:
        """Build RAG context for LLM prompts."""

        context_parts = []

        if include_guidelines:
            guidelines = self.get_relevant_guidelines(diagnosis)
            context_parts.append("=== RELEVANT GUIDELINES ===")
            for g in guidelines:
                context_parts.append(f"[{g['metadata']['title']}]")
                context_parts.append(f"- {g['text']}")

        if include_reports and findings:
            reports = self.get_similar_reports(findings)
            context_parts.append("\n=== SIMILAR REPORTS ===")
            for r in reports:
                context_parts.append(f"- {r['text']}")

        return "\n".join(context_parts)
```

### 5.4 Filtragem por Relevância

O sistema utiliza **threshold de distância** para garantir qualidade:

```python
# Relevance thresholds (ChromaDB uses L2 distance - lower is better)
DEFAULT_RELEVANCE_THRESHOLD = 1.2  # Max distance to consider relevant
STRICT_RELEVANCE_THRESHOLD = 0.8   # For high-precision queries

def _filter_by_relevance(self, results: list, threshold: float = 1.2) -> list:
    """Filter results by L2 distance threshold."""
    return [r for r in results if r.get("distance", 0) <= threshold]
```

---

## 6. Gerenciamento de Memória

### 6.1 Desafio: M4 Pro 24GB

O ambiente de desenvolvimento possui 24GB de memória unificada (MPS), mas:

- LLaVA-Med @ FP16: ~13GB
- BioMistral @ FP16: ~13GB
- MPS single allocation limit: ~10GB

**Solução: Carregamento Sequencial com Split MPS/CPU**

### 6.2 Estratégia de Device Mapping

```python
# Ambos os modelos usam device_map="auto" com limites:
max_memory = {"mps": "10GiB", "cpu": "16GiB"}

# HuggingFace Accelerate distribui layers:
# - Layers 0-20: MPS (GPU)
# - Layers 21-32: CPU/RAM
```

### 6.3 Carregamento Sequencial

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Memory Timeline                                       │
└─────────────────────────────────────────────────────────────────────────┘

Tempo →

│ RAG Retriever │ LLaVA-Med  │     │ BioMistral │     │
│    (~1GB)     │   (13GB)   │ GC  │   (13GB)   │ GC  │
└───────────────┴────────────┴─────┴────────────┴─────┘
                      ↑            ↑             ↑
                  Stage 1      Unload       Stage 2

Peak Memory: ~13GB (dentro do limite de 24GB)
```

### 6.4 Código de Unload

```python
# src/core/medical_pipeline.py

def _unload_llava_med(self) -> None:
    """Unload LLaVA-Med to free MPS memory before loading BioMistral."""
    if hasattr(self, "_llava_med_model"):
        tokenizer, model, image_processor, _ = self._llava_med_model

        # Move to CPU and delete
        model.cpu()
        del tokenizer, model, image_processor
        del self._llava_med_model

        # Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

def _unload_model(self, model_name: str) -> None:
    """Explicitly unload model and free memory."""
    if model_name == "biomistral":
        del self._biomistral_model
        del self._biomistral_tokenizer
        self._biomistral_model = None
        self._biomistral_tokenizer = None

    gc.collect()
    torch.mps.empty_cache()
```

---

## 7. Prompt Engineering Médico

### 7.1 Prompt do Estágio 1 (Vision)

```python
# src/prompts/medical_prompts.py

def build_vision_prompt(use_rag: bool = False, rag_context: str = "") -> str:
    """Build prompt for vision models (LLaVA-Med)."""

    base_prompt = """You are a medical AI assistant analyzing a chest X-ray.

TASK: Describe the radiological findings in this chest X-ray.

INSTRUCTIONS:
1. Identify anatomical structures (left lung, right lung, heart, diaphragm)
2. Note ANY opacities, consolidations, or abnormal patterns
3. If the X-ray is NORMAL, explicitly state "no acute findings"
4. CRITICAL: Use correct left/right orientation (right side appears left on PA view)
5. Be specific about location (e.g., "right lower lobe", "left hilum")
6. Mention presence/absence of: pleural effusion, pneumothorax, cardiomegaly
"""

    if use_rag and rag_context:
        base_prompt += f"""
MEDICAL REFERENCE KNOWLEDGE:
{rag_context}

Use the above guidelines to inform your analysis, but describe ONLY what you observe.
"""

    base_prompt += """
OUTPUT FORMAT:
- 2-3 concise sentences
- English only
- Professional medical terminology

Describe the X-ray findings:"""

    return base_prompt
```

### 7.2 Prompt do Estágio 2 (Medical)

```python
def build_medical_text_prompt(
    vision_description: str,
    cnn_diagnosis: dict,
    use_rag: bool = False,
    rag_context: str = "",
) -> str:
    """Build prompt for medical text generation (BioMistral-7B)."""

    diagnosis = cnn_diagnosis.get("prediction", "Unknown")
    confidence = cnn_diagnosis.get("confidence", 0.0)

    prompt = f"""A vision AI analyzed a chest X-ray and reported: "{vision_description}"

A CNN classifier predicted: {diagnosis} with {confidence:.0%} confidence.
"""

    if use_rag and rag_context:
        prompt += f"""
Relevant medical knowledge:
{rag_context}
"""

    prompt += """Write a professional 2-3 sentence radiology report that synthesizes these findings.
Use standard medical terminology and be specific about:
- Anatomical locations (e.g., "right lower lobe", "bilateral bases")
- Pattern of abnormality if present (consolidation, infiltrate, opacity)
- Clinical assessment or recommendation

If the findings are normal, state "No acute cardiopulmonary abnormality."
If findings contradict the CNN prediction, note "Clinical correlation recommended."

Radiology Report:"""

    return prompt
```

### 7.3 Estrutura dos Prompts

| Elemento | Estágio 1 (Vision) | Estágio 2 (Medical) |
|----------|-------------------|---------------------|
| **Persona** | "Medical AI assistant" | Implícito (radiologist) |
| **Task** | Descrever achados | Sintetizar laudo |
| **Context** | RAG guidelines | Vision output + CNN + RAG |
| **Format** | 2-3 sentenças, EN | 2-3 sentenças, EN |
| **Constraints** | Só o que observa | Terminologia padrão |

---

## 8. Pipeline Completo: Código

### 8.1 Classe MedicalPipeline

```python
# src/core/medical_pipeline.py

class MedicalPipeline:
    """Complete medical analysis pipeline with RAG grounding."""

    def __init__(
        self,
        use_rag: bool = True,
        vision_model: str = "llava-med",
        device: str = "mps",
    ) -> None:
        self.use_rag = use_rag
        self.vision_model_name = vision_model
        self.device = device

        # Lazy-loaded models
        self._rag_retriever = None
        self._biomistral_model = None
        self._biomistral_tokenizer = None

    def analyze_xray(
        self, image_path: str, cnn_diagnosis: dict | None = None
    ) -> dict:
        """Complete end-to-end X-ray analysis.

        Memory Management:
        - Stage 1: LLaVA-Med (load → generate → unload)
        - Stage 2: BioMistral (load → generate → unload)

        Peak memory: ~13GB
        """
        pipeline_start = time.time()

        if cnn_diagnosis is None:
            cnn_diagnosis = {"prediction": "UNKNOWN", "confidence": 0.0}

        # Stage 1: Vision analysis with LLaVA-Med
        vision_result = self.stage1_vision_analysis(image_path, cnn_diagnosis)

        # Stage 2: Medical text with BioMistral
        medical_result = self.stage2_medical_text(vision_result, cnn_diagnosis)

        # Cleanup
        self._unload_model("biomistral")

        return {
            "success": True,
            "image_path": image_path,
            "cnn_diagnosis": cnn_diagnosis,
            "stage1_vision": vision_result,
            "stage2_medical": medical_result,
            "final_report_en": medical_result["report_en"],
            "total_latency_s": time.time() - pipeline_start,
        }
```

### 8.2 Integração com API

```python
# src/api/routes.py

@router.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    """Complete X-ray analysis with 2-stage pipeline."""

    # 1. Save uploaded file
    image_path = save_upload(file)

    # 2. CNN classification
    cnn_result = get_diagnosis(image_path)

    # 3. Run medical pipeline
    pipeline = MedicalPipeline(use_rag=True, vision_model="llava-med")
    analysis = pipeline.analyze_xray(image_path, cnn_result)

    return {
        "diagnosis": cnn_result["prediction"],
        "confidence": cnn_result["confidence"],
        "vision_findings": analysis["stage1_vision"]["findings"],
        "medical_report": analysis["final_report_en"],
        "latency_s": analysis["total_latency_s"],
    }
```

---

## 9. Resultados e Métricas

### 9.1 Avaliação LLaVA-Med

Avaliação realizada com 5 casos de teste:

| Caso | Ground Truth | CNN | LLaVA-Med | Correto? |
|------|-------------|-----|-----------|----------|
| case_01_normal_clear | NORMAL | NORMAL (98.6%) | Normal findings | ✅ |
| case_02_pneumonia_severe | PNEUMONIA | PNEUMONIA (99.9%) | Consolidations detected | ✅ |
| case_03_normal_challenging | NORMAL | NORMAL (95.2%) | No acute findings | ✅ |
| case_04_pneumonia_moderate | PNEUMONIA | PNEUMONIA (96.2%) | Bilateral opacities | ✅ |
| case_05_false_negative | PNEUMONIA | NORMAL (98.6%) | No acute findings | ❌ |

**Métricas:**

| Métrica | Valor |
|---------|-------|
| **Acurácia LLaVA-Med** | 80% (4/5) |
| **Acurácia CNN** | 80% (4/5) |
| **Latência Média** | 39.3s |
| **Latência Vision (Stage 1)** | 35-40s |

### 9.2 Análise de Erros

O caso 5 (falso negativo) demonstra uma limitação importante:

> **Observação:** LLaVA-Med segue a orientação do RAG context, que inclui o diagnóstico da CNN. Quando a CNN erra (falso negativo), o LLaVA-Med tende a concordar.

**Mitigação futura:** Prompt adversarial que instrui "ignore CNN prediction, describe only what you see".

### 9.3 Performance por Estágio

```
Pipeline Breakdown (caso típico):
├── RAG Retrieval:    ~0.5s
├── Stage 1 (Vision): ~38s
│   └── LLaVA-Med inference
├── Model Unload:     ~2s
│   └── GC + MPS cache clear
├── Stage 2 (Medical): ~8s
│   └── BioMistral load + generate
└── Total:            ~48s
```

---

## 10. Conclusões

### 10.1 Contribuições Técnicas

1. ✅ **Pipeline de 2 estágios** com modelos médicos especializados
2. ✅ **LLaVA-Med** funcionando em Apple Silicon (MPS) com device_map
3. ✅ **BioMistral-7B** para geração de laudos profissionais
4. ✅ **Sistema RAG** com 31 guidelines médicas
5. ✅ **Gerenciamento de memória** para hardware limitado (24GB)

### 10.2 Limitações

1. ❌ Latência de ~40s (aceitável para diagnóstico, não tempo real)
2. ❌ Dependência do diagnóstico CNN (propagação de erros)
3. ❌ Limitado a hardware com 24GB+ de memória

### 10.3 Trabalhos Futuros

1. 🔬 **Prompt adversarial** para reduzir bias do CNN
2. 🚀 **Quantização 4-bit** para inferência mais rápida
3. 📊 **Validação clínica** com radiologistas
4. 🤖 **Multi-patologia** (TB, COVID-19, efusão pleural)
5. 🌐 **Tradução opcional** com modelos menores (NLLB)

---

## Referências

1. **LLaVA-Med:** Li et al. (2023). "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day". NeurIPS 2023.
2. **BioMistral:** Labrak et al. (2024). "BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains". arXiv:2402.10373.
3. **Mistral-7B:** Jiang et al. (2023). "Mistral 7B". arXiv:2310.06825.
4. **RAG:** Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS 2020.
5. **ChromaDB:** https://www.trychroma.com/
6. **HuggingFace Accelerate:** https://huggingface.co/docs/accelerate/

---

**Documento atualizado para TCC - PneumoFinder v3.0**
**Última atualização:** Fevereiro 2026
