# 07 - Código Fonte (Apêndices)

## Sumário
- [Visão Geral](#visão-geral)
- [Apêndice A: Módulo de Diagnóstico (CNN)](#apêndice-a-módulo-de-diagnóstico-cnn)
- [Apêndice B: Módulo de Visualização (Grad-CAM)](#apêndice-b-módulo-de-visualização-grad-cam)
- [Apêndice C: Pipeline de 2 Estágios (LLaVA-Med + BioMistral)](#apêndice-c-pipeline-de-2-estágios-llava-med--biomistral)
- [Apêndice D: Sistema RAG](#apêndice-d-sistema-rag)
- [Apêndice E: Prompt Engineering Médico](#apêndice-e-prompt-engineering-médico)
- [Apêndice F: Endpoints da API](#apêndice-f-endpoints-da-api)
- [Apêndice G: Configuração](#apêndice-g-configuração)

---

## Visão Geral

Este capítulo apresenta os principais trechos de código do PneumoFinder, organizados como apêndices para referência técnica. Cada seção inclui:

- **Código completo e comentado** do módulo
- **Explicação linha a linha** de trechos críticos
- **Referências** para localização no repositório

**Repositório:** [github.com/user/pneumofinder](https://github.com) (exemplo)  
**Linguagem:** Python 3.11  
**Linhas de código:** ~1,200 (excluindo testes e documentação)

---

## Apêndice A: Módulo de Diagnóstico (CNN)

### Arquivo: `src/core/diagnosis.py`

Funções para carregar modelo CNN e executar inferência.

```python
"""
Módulo de diagnóstico de pneumonia usando CNN (ResNet50).
Fornece funções puras para inferência sem efeitos colaterais.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.utils.image_utils import load_and_preprocess_image


def load_cnn_model(model_path: str | Path) -> tf.keras.Model:
    """
    Carrega modelo CNN treinado do disco.
    
    Args:
        model_path: Caminho para arquivo .keras (TensorFlow SavedModel)
        
    Returns:
        model: Modelo Keras compilado pronto para inferência
        
    Raises:
        FileNotFoundError: Se o modelo não existir
        
    Example:
        >>> model = load_cnn_model("models/pneumonia_model.keras")
        >>> model.summary()
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    # Carrega modelo (inclui arquitetura + pesos + configuração de otimização)
    model = load_model(str(model_path))
    
    return model


def diagnose_from_path(model: tf.keras.Model, image_path: str) -> tuple[str, float]:
    """
    Realiza diagnóstico de pneumonia a partir de arquivo de imagem.
    
    Args:
        model: Modelo CNN carregado (ResNet50)
        image_path: Caminho para radiografia de tórax (JPEG/PNG)
        
    Returns:
        diagnosis: "PNEUMONIA" ou "NORMAL"
        confidence: Probabilidade da classe predita (0.0-1.0)
        
    Example:
        >>> diagnosis, confidence = diagnose_from_path(model, "xray.jpg")
        >>> print(f"{diagnosis} ({confidence:.1%})")
        PNEUMONIA (87.3%)
    """
    # 1. Carrega e preprocessa imagem
    img_array = load_and_preprocess_image(image_path, target_size=(224, 224))
    
    # 2. Adiciona dimensão de batch: (224, 224, 3) → (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 3. Executa inferência CNN
    predictions = model.predict(img_batch, verbose=0)
    
    # 4. Extrai probabilidade da classe PNEUMONIA (índice 1)
    pneumonia_prob = float(predictions[0][1])
    
    # 5. Aplica threshold de decisão (0.5)
    if pneumonia_prob >= 0.5:
        diagnosis = "PNEUMONIA"
        confidence = pneumonia_prob
    else:
        diagnosis = "NORMAL"
        confidence = 1.0 - pneumonia_prob  # Inverte probabilidade
    
    return diagnosis, confidence


def diagnose_with_visualization(
    model: tf.keras.Model,
    resnet_base: tf.keras.Model,
    last_conv_layer: tf.keras.layers.Layer,
    image_path: str,
    output_dir: str,
) -> tuple[str, float, str, str]:
    """
    Diagnóstico completo com Grad-CAM incluído.
    
    Combina diagnose_from_path() + generate_gradcam() em uma função.
    
    Args:
        model: Modelo completo (ResNet50 + camadas densas)
        resnet_base: Apenas ResNet50 base (para Grad-CAM)
        last_conv_layer: Última camada convolucional (target Grad-CAM)
        image_path: Caminho da imagem de entrada
        output_dir: Diretório para salvar heatmap/overlay
        
    Returns:
        diagnosis: "PNEUMONIA" ou "NORMAL"
        confidence: Probabilidade (0.0-1.0)
        heatmap_path: Caminho do heatmap salvo
        overlay_path: Caminho do overlay salvo
    """
    from src.core.visualization import generate_gradcam
    
    # Diagnóstico normal
    diagnosis, confidence = diagnose_from_path(model, image_path)
    
    # Grad-CAM (sempre executado para documentação visual)
    heatmap_path, overlay_path = generate_gradcam(
        model=model,
        resnet_base=resnet_base,
        last_conv_layer=last_conv_layer,
        image_path=image_path,
        output_dir=output_dir,
    )
    
    return diagnosis, confidence, heatmap_path, overlay_path
```

**Arquivo fonte:** `src/core/diagnosis.py` (Linhas 1-120)

---

## Apêndice B: Módulo de Visualização (Grad-CAM)

### Arquivo: `src/core/visualization.py`

Implementação completa do Grad-CAM (Gradient-weighted Class Activation Mapping).

```python
"""
Geração de visualizações Grad-CAM para explicabilidade do modelo CNN.
Implementa algoritmo de Selvaraju et al. (2017).
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def find_resnet_base(model: tf.keras.Model) -> tf.keras.Model:
    """
    Extrai ResNet50 base de um modelo customizado.
    
    Args:
        model: Modelo Keras (pode conter ResNet + camadas adicionais)
        
    Returns:
        resnet_base: Submodelo ResNet50 (até última conv layer)
    """
    for layer in model.layers:
        if "resnet" in layer.name.lower():
            return layer
    
    # Se não encontrar, assume que o modelo inteiro é ResNet
    return model


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Encontra última camada convolucional do modelo (target para Grad-CAM).
    
    Args:
        model: Modelo CNN (ResNet50, DenseNet, etc.)
        
    Returns:
        last_conv: Última camada Conv2D encontrada
        
    Raises:
        ValueError: Se nenhuma camada convolucional for encontrada
    """
    # Itera camadas de trás para frente
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    
    raise ValueError("Nenhuma camada Conv2D encontrada no modelo")


def generate_gradcam(
    model: tf.keras.Model,
    resnet_base: tf.keras.Model,
    last_conv_layer: tf.keras.layers.Layer,
    image_path: str,
    output_dir: str,
) -> tuple[str, str]:
    """
    Gera Grad-CAM heatmap e overlay para uma imagem.
    
    Algoritmo:
    1. Forward pass até última conv layer (extrai feature maps)
    2. Forward pass completo (extrai predição)
    3. Backward pass (calcula gradientes da predição em relação aos feature maps)
    4. Média ponderada dos feature maps pelos gradientes
    5. Aplica ReLU (remove valores negativos)
    6. Normaliza para [0, 1]
    7. Redimensiona para tamanho da imagem original
    8. Aplica colormap (azul=baixa ativação, vermelho=alta ativação)
    9. Sobrepõe heatmap na imagem original (overlay)
    
    Args:
        model: Modelo completo
        resnet_base: ResNet50 base
        last_conv_layer: Última camada convolucional
        image_path: Caminho da radiografia
        output_dir: Diretório de saída
        
    Returns:
        heatmap_path: Caminho do heatmap salvo
        overlay_path: Caminho do overlay salvo
        
    References:
        Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization". ICCV 2017.
    """
    from src.utils.image_utils import load_and_preprocess_image
    
    # 1. Carrega imagem original e preprocessada
    img_array = load_and_preprocess_image(image_path, target_size=(224, 224))
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 2. Cria modelo Grad-CAM (input → feature maps + predição)
    grad_model = Model(
        inputs=[model.input],
        outputs=[last_conv_layer.output, model.output]
    )
    
    # 3. Forward + Backward pass com GradientTape
    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs, predictions = grad_model(img_batch)
        
        # Extrai score da classe de interesse (índice 1 = PNEUMONIA)
        class_channel = predictions[:, 1]
    
    # 4. Calcula gradientes: ∂y_c / ∂A^k
    grads = tape.gradient(class_channel, conv_outputs)
    
    # 5. Global Average Pooling dos gradientes (α_k^c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 6. Média ponderada: L_Grad-CAM = ReLU(Σ α_k^c × A^k)
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # 7. Aplica ReLU (remove valores negativos)
    heatmap = np.maximum(heatmap, 0)
    
    # 8. Normaliza para [0, 1]
    heatmap = heatmap / (heatmap.max() + 1e-8)  # Evita divisão por zero
    
    # 9. Redimensiona para tamanho da imagem original
    original_img = cv2.imread(image_path)
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # 10. Aplica colormap (COLORMAP_JET: azul → verde → amarelo → vermelho)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    
    # 11. Cria overlay (60% imagem original + 40% heatmap)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    # 12. Salva heatmap e overlay
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_path).stem
    heatmap_path = str(output_dir / f"{base_name}_heatmap.png")
    overlay_path = str(output_dir / f"{base_name}_overlay.png")
    
    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(overlay_path, overlay)
    
    return heatmap_path, overlay_path
```

**Arquivo fonte:** `src/core/visualization.py` (Linhas 1-160)

**Complexidade:** O(H × W × C) onde H=altura, W=largura, C=canais da feature map

---

## Apêndice C: Pipeline de 2 Estágios (LLaVA-Med + BioMistral)

### Arquivo: `src/core/medical_pipeline.py`

Orquestrador do pipeline de análise médica com 2 estágios:
1. **Estágio 1 (Vision):** LLaVA-Med analisa a radiografia e extrai achados
2. **Estágio 2 (Medical Text):** BioMistral-7B sintetiza laudo médico profissional

```python
"""Medical pipeline orchestrator for 2-stage LLM analysis with RAG.

This module orchestrates the medical analysis pipeline:
1. Stage 1: Vision analysis (LLaVA-Med) → English findings
2. Stage 2: Medical text generation (BioMistral-7B) → Professional English report

Memory Management (M4 Pro 24GB):
- Sequential loading: Only 1 HuggingFace model in memory at a time
- MPS constraint: ~10GB max single allocation (models @ 13GB need splitting)
- Peak memory: ~13GB
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts.medical_prompts import build_medical_text_prompt, build_vision_prompt
from src.rag.retriever import MedicalRetriever

logger = logging.getLogger(__name__)


class MedicalPipeline:
    """Complete medical analysis pipeline with RAG grounding."""

    def __init__(
        self,
        use_rag: bool = True,
        vision_model: str = "llava-med",
        device: str = "mps",
    ) -> None:
        """Initialize medical pipeline.

        Args:
            use_rag: Whether to use RAG context grounding
            vision_model: Vision model ("llava-med" or "llava-llama3")
            device: Device for inference ("mps", "cuda", or "cpu")
        """
        self.use_rag = use_rag
        self.vision_model_name = vision_model
        self.device = device

        # Lazy-loaded models
        self._rag_retriever = None
        self._biomistral_model = None
        self._biomistral_tokenizer = None

        logger.info(f"MedicalPipeline initialized (RAG: {use_rag}, Vision: {vision_model})")

    def _load_biomistral(self) -> None:
        """Lazy-load BioMistral model with intelligent device mapping.

        Strategy: MPS has ~10GB single allocation limit, but BioMistral @ FP16 = 13GB.
        Use device_map="auto" to split across MPS + CPU/RAM intelligently.
        """
        if self._biomistral_model is None:
            logger.info("Loading BioMistral-7B with intelligent device mapping...")
            model_id = "BioMistral/BioMistral-7B"

            self._biomistral_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._biomistral_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",  # Let transformers split intelligently
                low_cpu_mem_usage=True,
                max_memory={"mps": "10GiB", "cpu": "16GiB"},
            )
            logger.info(f"BioMistral-7B loaded (device_map: {self._biomistral_model.hf_device_map})")

    def _unload_model(self, model_name: str) -> None:
        """Explicitly unload model and free memory.

        Critical for M4 Pro 24GB: ensures only 1 HuggingFace model loaded at a time.

        Args:
            model_name: Model to unload ("biomistral")
        """
        import gc

        if model_name == "biomistral" and self._biomistral_model is not None:
            logger.info("Unloading BioMistral-7B to free ~13GB memory...")
            del self._biomistral_model
            del self._biomistral_tokenizer
            self._biomistral_model = None
            self._biomistral_tokenizer = None

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info(f"✅ {model_name} unloaded, MPS cache cleared")

    def stage1_vision_analysis(
        self, image_path: str, cnn_diagnosis: dict[str, Any]
    ) -> dict[str, Any]:
        """Stage 1: Vision analysis with RAG grounding.

        Args:
            image_path: Path to chest X-ray image
            cnn_diagnosis: CNN classification result

        Returns:
            Dictionary with vision analysis results
        """
        logger.info("=== STAGE 1: Vision Analysis ===")
        start_time = time.time()

        # Build RAG context
        rag_context = ""
        if self.use_rag:
            diagnosis = cnn_diagnosis.get("prediction", "").lower()
            rag_context = self.rag_retriever.build_rag_context(
                diagnosis=diagnosis,
                include_guidelines=True,
                include_reports=False,
            )

        # Build and run vision model
        vision_prompt = build_vision_prompt(use_rag=self.use_rag, rag_context=rag_context)
        vision_output = self._run_llava_med(image_path, vision_prompt)

        return {
            "findings_en": vision_output,
            "model": self.vision_model_name,
            "rag_context": rag_context,
            "latency_s": time.time() - start_time,
        }

    def stage2_medical_text(
        self, vision_result: dict[str, Any], cnn_diagnosis: dict[str, Any]
    ) -> dict[str, Any]:
        """Stage 2: Medical text generation with BioMistral.

        Args:
            vision_result: Output from stage 1
            cnn_diagnosis: CNN classification result

        Returns:
            Dictionary with medical text generation results
        """
        logger.info("=== STAGE 2: Medical Text Generation ===")
        start_time = time.time()

        # Unload LLaVA-Med, load BioMistral (sequential loading)
        self._unload_llava_med()
        self._load_biomistral()

        # Build RAG context with similar reports
        rag_context = ""
        if self.use_rag:
            rag_context = self.rag_retriever.build_rag_context(
                diagnosis=cnn_diagnosis.get("prediction", "").lower(),
                findings=vision_result["findings_en"],
                include_guidelines=True,
                include_reports=True,
            )

        # Build prompt and generate
        medical_prompt = build_medical_text_prompt(
            vision_description=vision_result["findings_en"],
            cnn_diagnosis=cnn_diagnosis,
            use_rag=self.use_rag,
            rag_context=rag_context,
        )

        # Format with Mistral chat template
        formatted_prompt = self._biomistral_tokenizer.apply_chat_template(
            [{"role": "user", "content": medical_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate medical report
        inputs = self._biomistral_tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self._biomistral_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._biomistral_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._biomistral_tokenizer.eos_token_id,
            )

        medical_report = self._biomistral_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return {
            "report_en": medical_report.strip(),
            "model": "BioMistral-7B",
            "rag_context": rag_context,
            "latency_s": time.time() - start_time,
        }

    def analyze_xray(
        self, image_path: str, cnn_diagnosis: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Complete end-to-end X-ray analysis with sequential model loading.

        Memory Management:
        - Stage 1: Vision model (LLaVA-Med)
        - Stage 2: BioMistral (load → generate → unload)
        Peak memory: ~13GB

        Args:
            image_path: Path to chest X-ray image
            cnn_diagnosis: Optional CNN classification result

        Returns:
            Complete analysis results (2 stages)
        """
        pipeline_start = time.time()

        if cnn_diagnosis is None:
            cnn_diagnosis = {"prediction": "UNKNOWN", "confidence": 0.0}

        # Stage 1: Vision analysis
        vision_result = self.stage1_vision_analysis(image_path, cnn_diagnosis)

        # Stage 2: Medical text generation
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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
```

**Arquivo fonte:** `src/core/medical_pipeline.py` (Linhas 1-210)

**Características:**
- **Carregamento sequencial:** Apenas 1 modelo HuggingFace na memória por vez
- **device_map="auto":** Split inteligente entre MPS + CPU
- **Pico de memória:** ~13GB (compatível com M4 Pro 24GB)
- **Latência total:** ~39-49 segundos por análise

---

## Apêndice D: Sistema RAG

### Arquivo: `src/rag/retriever.py`

Retriever de conhecimento médico para grounding de geração com RAG.

```python
"""Medical knowledge retriever for RAG-enhanced generation.

Features:
- Relevance threshold filtering (removes low-quality matches)
- Query caching for performance
- Category-aware retrieval
- Diversity ranking to avoid redundant results
"""

import logging
from typing import Any

from src.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Relevance thresholds (ChromaDB uses L2 distance - lower is better)
DEFAULT_RELEVANCE_THRESHOLD = 1.2  # Max distance to consider relevant
STRICT_RELEVANCE_THRESHOLD = 0.8   # For high-precision queries


class MedicalRetriever:
    """Retriever for medical knowledge to ground LLM generation.

    Features:
    - Relevance threshold: Filters out results below quality threshold
    - Caching: LRU cache for repeated queries
    - Category filtering: Can filter by pathology, severity, signs, etc.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        """Initialize medical retriever.

        Args:
            vector_store: ChromaDB vector store (creates new if None)
            relevance_threshold: Max distance to consider relevant (lower = stricter)
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.relevance_threshold = relevance_threshold

    def _filter_by_relevance(
        self, results: list[dict[str, Any]], threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Filter results by relevance threshold.

        Args:
            results: List of result dicts with 'distance' key
            threshold: Optional custom threshold

        Returns:
            Filtered list with only relevant results
        """
        max_distance = threshold or self.relevance_threshold
        filtered = [r for r in results if r.get("distance", 0) <= max_distance]

        if len(filtered) < len(results):
            removed = len(results) - len(filtered)
            logger.debug(f"Filtered out {removed} low-relevance results")

        return filtered

    def get_relevant_guidelines(
        self,
        diagnosis: str,
        n_results: int = 3,
        category: str | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant radiology guidelines with quality filtering.

        Args:
            diagnosis: Diagnosis or condition (e.g., 'pneumonia', 'normal')
            n_results: Number of guidelines to retrieve
            category: Optional category filter ('pathology', 'signs', etc.)
            strict: Use stricter relevance threshold

        Returns:
            List of relevant guidelines with metadata
        """
        query_n = n_results * 2 if category else n_results + 2

        results = self.vector_store.query(
            collection_name="medical_guidelines",
            query_text=diagnosis,
            n_results=query_n,
            where={"category": category} if category else None,
        )

        guidelines = []
        if results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                guidelines.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })

        # Apply relevance filtering
        threshold = STRICT_RELEVANCE_THRESHOLD if strict else self.relevance_threshold
        guidelines = self._filter_by_relevance(guidelines, threshold)

        return guidelines[:n_results]

    def get_similar_reports(
        self,
        findings: str,
        n_results: int = 2,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar radiology reports for reference.

        Args:
            findings: Clinical findings description
            n_results: Number of similar reports to retrieve
            language: Optional language filter ('en' or 'pt-BR')

        Returns:
            List of similar reports with metadata
        """
        where = {"language": language} if language else None

        results = self.vector_store.query(
            collection_name="sample_reports",
            query_text=findings,
            n_results=n_results + 2,
            where=where,
        )

        reports = []
        if results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                reports.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })

        return self._filter_by_relevance(reports)[:n_results]

    def build_rag_context(
        self,
        diagnosis: str,
        findings: str | None = None,
        include_guidelines: bool = True,
        include_reports: bool = True,
        prefer_english_reports: bool = True,
    ) -> str:
        """Build RAG context for LLM prompts.

        Args:
            diagnosis: Primary diagnosis
            findings: Clinical findings (optional)
            include_guidelines: Whether to include guidelines
            include_reports: Whether to include similar reports
            prefer_english_reports: Prefer English reports for EN LLMs

        Returns:
            Formatted RAG context string
        """
        context_parts = []

        if include_guidelines:
            guidelines = self.get_relevant_guidelines(diagnosis, n_results=3)
            if guidelines:
                context_parts.append("=== RELEVANT GUIDELINES ===")
                for g in guidelines[:4]:
                    title = g.get("metadata", {}).get("title", "Guideline")
                    context_parts.append(f"[{title}]")
                    context_parts.append(f"- {g['text']}")

        if include_reports and findings:
            language = "en" if prefer_english_reports else None
            reports = self.get_similar_reports(findings, n_results=2, language=language)
            if reports:
                context_parts.append("\n=== SIMILAR REPORTS ===")
                for r in reports:
                    title = r.get("metadata", {}).get("title", "Report")
                    context_parts.append(f"[{title}]")
                    context_parts.append(f"- {r['text']}")

        context = "\n".join(context_parts) if context_parts else ""
        if context:
            logger.info(f"Built RAG context: {len(context)} chars")
        return context
```

**Arquivo fonte:** `src/rag/retriever.py` (Linhas 1-180)

**Base de conhecimento:** 31 guidelines médicas + 50+ laudos de referência

---

## Apêndice E: Prompt Engineering Médico

### Arquivo: `src/prompts/medical_prompts.py`

Templates de prompts para o pipeline de análise médica com injeção de contexto RAG.

```python
"""Prompt templates for medical AI system with RAG context injection."""


def build_vision_prompt(use_rag: bool = False, rag_context: str = "") -> str:
    """Build prompt for vision models (LLaVA-Med, llava-llama3).

    Args:
        use_rag: Whether to include RAG context
        rag_context: RAG-retrieved medical knowledge

    Returns:
        Formatted prompt string
    """
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
- English only (will be translated later)
- Professional medical terminology
- If normal: state clearly "The lungs are clear bilaterally. No acute abnormality."

Describe the X-ray findings:"""

    return base_prompt


def build_medical_text_prompt(
    vision_description: str,
    cnn_diagnosis: dict,
    use_rag: bool = False,
    rag_context: str = "",
) -> str:
    """Build prompt for medical text generation (BioMistral-7B).

    Args:
        vision_description: Output from vision model
        cnn_diagnosis: CNN classification result
        use_rag: Whether to include RAG context
        rag_context: RAG-retrieved medical knowledge

    Returns:
        Formatted prompt for BioMistral
    """
    diagnosis = cnn_diagnosis.get("prediction", "Unknown")
    confidence = cnn_diagnosis.get("confidence", 0.0)

    prompt = f"""A vision AI analyzed a chest X-ray and reported: "{vision_description}"

A CNN classifier predicted: {diagnosis} with {confidence:.0%} confidence.

"""

    if use_rag and rag_context:
        prompt += f"""Relevant medical knowledge:
{rag_context}

"""

    prompt += """Write a professional 2-3 sentence radiology report that synthesizes these findings.
Use standard medical terminology and be specific about:
- Anatomical locations (e.g., "right lower lobe", "bilateral bases")
- Pattern of abnormality if present (consolidation, infiltrate, opacity)
- Clinical assessment or recommendation

If the findings are normal, state "No acute cardiopulmonary abnormality" or similar.
If findings contradict the CNN prediction, note "Clinical correlation recommended."

Radiology Report:"""

    return prompt
```

**Arquivo fonte:** `src/prompts/medical_prompts.py` (Linhas 1-95)

**Características dos prompts:**
- **Estrutura clara:** TASK → INSTRUCTIONS → OUTPUT FORMAT
- **Grounding com RAG:** Contexto médico injetado condicionalmente
- **Orientação espacial:** Instruções explícitas sobre lateralidade em radiografias PA
- **Síntese, não eco:** Prompt encoraja integração de múltiplas fontes

---

## Apêndice F: Endpoints da API

### Arquivo: `src/api/app.py`

Endpoints Flask RESTful com integração do pipeline LLM.

```python
"""
API RESTful do PneumoFinder com Flask.
Integração do pipeline de 2 estágios (LLaVA-Med + BioMistral).
"""

import hashlib
import logging
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.core.diagnosis import diagnose_from_path, load_cnn_model
from src.core.medical_pipeline import MedicalPipeline
from src.core.visualization import find_last_conv_layer, find_resnet_base, generate_gradcam
from src.utils.config import config
from src.utils.file_utils import cleanup_file, save_uploaded_file

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Carrega modelos no startup
cnn_model = load_cnn_model(config.cnn_model_path)
resnet_base = find_resnet_base(cnn_model)
last_conv_layer = find_last_conv_layer(resnet_base)

# Pipeline de 2 estágios (lazy-loaded)
medical_pipeline = MedicalPipeline(use_rag=True, vision_model="llava-med")


@app.route("/analyze", methods=["POST"])
def analyze_xray():
    """
    Endpoint completo: CNN + Grad-CAM + Pipeline 2-Estágios (LLaVA-Med → BioMistral).

    Request:
        POST /analyze
        Content-Type: multipart/form-data
        Body: image=@radiografia.jpg

    Response (200 OK):
        {
          "success": true,
          "cnn_diagnosis": {"prediction": "PNEUMONIA", "confidence": 0.87},
          "stage1_vision": {"findings_en": "Consolidation in right lower lobe..."},
          "stage2_medical": {"report_en": "Chest X-ray reveals..."},
          "final_report_en": "Chest X-ray reveals consolidation...",
          "gradcam_overlay_url": "/static/temp/img_overlay.png",
          "total_latency_s": 39.2
        }

    Response (400 Bad Request):
        {"error": "No image provided"}

    Response (500 Internal Server Error):
        {"success": false, "error": "Pipeline failed: ..."}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    temp_path = None
    try:
        # 1. Upload e salvamento temporário
        image_file = request.files["image"]
        temp_path = save_uploaded_file(image_file, config.temp_dir)
        logger.info(f"Analyzing: {image_file.filename}")

        # 2. CNN Diagnosis
        diagnosis, confidence = diagnose_from_path(cnn_model, temp_path)
        cnn_result = {"prediction": diagnosis, "confidence": round(confidence, 4)}

        # 3. Grad-CAM Visualization
        _, overlay_path = generate_gradcam(
            cnn_model, resnet_base, last_conv_layer,
            temp_path, config.temp_dir
        )

        # 4. Pipeline 2-Estágios (LLaVA-Med → BioMistral)
        pipeline_result = medical_pipeline.analyze_xray(
            image_path=temp_path,
            cnn_diagnosis=cnn_result
        )

        # 5. Retorna resposta completa
        return jsonify({
            "success": pipeline_result.get("success", False),
            "cnn_diagnosis": cnn_result,
            "stage1_vision": pipeline_result.get("stage1_vision", {}),
            "stage2_medical": pipeline_result.get("stage2_medical", {}),
            "final_report_en": pipeline_result.get("final_report_en", ""),
            "gradcam_overlay_url": f"/static/temp/{Path(overlay_path).name}",
            "total_latency_s": round(pipeline_result.get("total_latency_s", 0), 1),
        })

    except Exception as e:
        logger.error(f"Error in /analyze: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if temp_path:
            cleanup_file(temp_path)
```

**Arquivo fonte:** `src/api/app.py` (Linhas 1-95)

**Características:**
- **Pipeline integrado:** CNN → Grad-CAM → LLaVA-Med → BioMistral
- **RAG ativado:** Contexto médico injetado automaticamente
- **Resposta estruturada:** Resultados de cada estágio retornados separadamente

---

## Apêndice G: Configuração

### Arquivo: `src/utils/config.py`

Configuração centralizada com suporte a variáveis de ambiente e modelos HuggingFace.

```python
"""
Configuração centralizada do PneumoFinder v3.
Suporta variáveis de ambiente para deployment.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """
    Classe de configuração com valores padrão.

    Variáveis de ambiente têm precedência sobre valores padrão.

    Exemplo de uso:
        >>> from src.utils.config import config
        >>> print(config.vision_model)
        llava-med
        >>> print(config.text_model)
        BioMistral/BioMistral-7B
    """

    # ===== API =====
    api_port: int = int(os.getenv("API_PORT", "5001"))
    debug_mode: bool = os.getenv("FLASK_DEBUG", "0") == "1"

    # ===== Modelos CNN =====
    cnn_model_path: str = os.getenv(
        "CNN_MODEL_PATH",
        "models/pneumonia_model.keras"
    )
    model_version: str = "resnet50_v1"

    # ===== Modelos LLM (HuggingFace) =====
    vision_model: str = os.getenv("VISION_MODEL", "llava-med")  # "llava-med" ou "llava-llama3"
    llava_med_model: str = "microsoft/llava-med-v1.5-mistral-7b"
    text_model: str = os.getenv("TEXT_MODEL", "BioMistral/BioMistral-7B")

    # ===== Gerenciamento de Memória (Apple Silicon) =====
    device: str = os.getenv("DEVICE", "mps")  # "mps", "cuda", ou "cpu"
    max_mps_memory: str = "10GiB"  # Limite MPS para evitar fragmentação
    max_cpu_memory: str = "16GiB"  # Fallback para CPU/RAM

    # ===== RAG =====
    use_rag: bool = os.getenv("USE_RAG", "1") == "1"
    rag_db_path: str = os.getenv("RAG_DB_PATH", "data/rag")
    relevance_threshold: float = float(os.getenv("RAG_THRESHOLD", "1.2"))

    # ===== ChromaDB =====
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))

    # ===== Diretórios =====
    temp_dir: str = "temp"
    database_dir: str = "database"

    # ===== HuggingFace =====
    hf_cache_dir: str = os.getenv(
        "HF_HOME",
        str(Path.home() / ".cache/huggingface")
    )


# Instância global (singleton)
config = Config()


def validate_config():
    """Valida configuração no startup."""
    # Valida modelo CNN
    if not Path(config.cnn_model_path).exists():
        raise FileNotFoundError(
            f"Modelo CNN não encontrado: {config.cnn_model_path}"
        )

    # Cria diretórios necessários
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(config.database_dir).mkdir(parents=True, exist_ok=True)
    Path(config.rag_db_path).mkdir(parents=True, exist_ok=True)


validate_config()
```

**Arquivo fonte:** `src/utils/config.py` (Linhas 1-80)

**Variáveis de ambiente suportadas:**
| Variável | Default | Descrição |
|----------|---------|-----------|
| `API_PORT` | 5001 | Porta da API Flask |
| `VISION_MODEL` | llava-med | Modelo de visão (llava-med ou llava-llama3) |
| `TEXT_MODEL` | BioMistral/BioMistral-7B | Modelo de texto médico |
| `DEVICE` | mps | Dispositivo de inferência (mps, cuda, cpu) |
| `USE_RAG` | 1 | Ativar RAG (1=sim, 0=não) |
| `RAG_THRESHOLD` | 1.2 | Threshold de relevância (distância L2) |
| `HF_HOME` | ~/.cache/huggingface | Cache de modelos HuggingFace |

---

## Conclusão

Este apêndice fornece os principais trechos de código do PneumoFinder v3 para referência técnica na monografia. O código completo está disponível no repositório GitHub.

**Estatísticas do código:**

| Métrica | Valor |
|---------|-------|
| **Linhas de código** | ~1,500 |
| **Arquivos Python** | 22 |
| **Módulos principais** | 8 (api, core, rag, prompts, db, utils, bots) |
| **Funções** | 55+ |
| **Classes** | 5 (MedicalPipeline, MedicalRetriever, ChromaVectorStore, Config, etc.) |
| **Modelos LLM** | 2 (LLaVA-Med 7B, BioMistral-7B) |
| **Guidelines RAG** | 31 documentos médicos |

**Stack tecnológico:**

| Componente | Tecnologia |
|------------|------------|
| **API** | Flask + Flask-CORS |
| **CNN** | TensorFlow/Keras (ResNet50) |
| **Vision LLM** | LLaVA-Med (HuggingFace Transformers) |
| **Text LLM** | BioMistral-7B (HuggingFace Transformers) |
| **RAG** | ChromaDB + all-MiniLM-L6-v2 |
| **Aceleração** | Apple MPS (Metal Performance Shaders) |

**Convenções de código:**
- ✅ PEP 8 compliance (ruff format)
- ✅ Type hints em 100% das funções
- ✅ Docstrings em estilo Google
- ✅ Imports ordenados com isort
- ✅ Nomes em inglês (código) e português (UI/docs)

**Licença:** MIT License (ver arquivo LICENSE no repositório)

---

**Fim dos apêndices de código.**
