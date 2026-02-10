"""Medical pipeline orchestrator for 2-stage LLM analysis with RAG.

This module orchestrates the medical analysis pipeline:
1. Stage 1: Vision analysis (LLaVA-Med or llava-llama3) → English findings
2. Stage 2: Medical text generation (BioMistral-7B) → Professional English report

All stages are grounded with RAG context to prevent hallucinations.

Memory Management (M4 Pro 24GB):
- Sequential loading: Only 1 HuggingFace model in memory at a time
- Ollama (Stage 1): Auto-unloads after completion → ~3GB baseline
- HuggingFace models (Stage 2): Intelligent device_map splits across MPS+CPU/RAM
- MPS constraint: ~10GB max single allocation (models @ 13GB need splitting)
- Peak memory: ~13GB
- Design: Industry-standard pattern (HuggingFace Accelerate approach)
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts.medical_prompts import (
    build_medical_text_prompt,
    build_vision_prompt,
)
from src.rag.retriever import MedicalRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalPipeline:
    """Complete medical analysis pipeline with RAG grounding."""

    def __init__(
        self,
        use_rag: bool = True,
        vision_model: str = "llava-med",  # "llava-med" or "llava-llama3"
        device: str = "mps",  # "mps", "cuda", or "cpu"
    ) -> None:
        """Initialize medical pipeline.

        Args:
            use_rag: Whether to use RAG context grounding
            vision_model: Vision model to use ("llava-med" or "llava-llama3")
            device: Device for model inference
        """
        self.use_rag = use_rag
        self.vision_model_name = vision_model
        self.device = device

        # Lazy-loaded models
        self._rag_retriever = None
        self._biomistral_model = None
        self._biomistral_tokenizer = None
        self._llava_med_loaded = False

        logger.info(f"MedicalPipeline initialized (RAG: {use_rag}, Vision: {vision_model})")

    @property
    def rag_retriever(self) -> MedicalRetriever:
        """Lazy-load RAG retriever."""
        if self._rag_retriever is None:
            logger.info("Loading RAG retriever...")
            self._rag_retriever = MedicalRetriever()
        return self._rag_retriever

    def _load_biomistral(self) -> None:
        """Lazy-load BioMistral model with intelligent device mapping.

        Strategy: MPS has ~10GB single allocation limit, but BioMistral @ FP16 = 13GB.
        Use device_map="auto" to split across MPS + CPU/RAM intelligently.
        Since we unload models between stages, this avoids memory conflicts.
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
                max_memory={"mps": "10GiB", "cpu": "16GiB"},  # MPS limit + CPU fallback
            )
            logger.info(
                f"BioMistral-7B loaded (device_map: {self._biomistral_model.hf_device_map})"
            )

    def _unload_model(self, model_name: str) -> None:
        """Explicitly unload model and free memory.

        Critical for M4 Pro 24GB: ensures only 1 HuggingFace model loaded at a time.
        Ollama (Stage 1) auto-unloads, but HuggingFace models need explicit cleanup.

        Memory Management Strategy:
        - Delete model/tokenizer references (Python GC eligible)
        - Force garbage collection (gc.collect())
        - Clear MPS cache (Metal Performance Shaders on macOS)

        Expected memory drop: ~13GB per unload

        Args:
            model_name: Model to unload ("biomistral")
        """
        import gc

        if model_name == "biomistral":
            if self._biomistral_model is not None:
                logger.info("Unloading BioMistral-7B to free ~13GB memory...")
                del self._biomistral_model
                del self._biomistral_tokenizer
                self._biomistral_model = None
                self._biomistral_tokenizer = None
        else:
            logger.warning(f"Unknown model name for unload: {model_name}")
            return

        # Force Python garbage collection
        gc.collect()

        # Clear MPS (Metal Performance Shaders) cache on macOS
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info(f"✅ {model_name} unloaded, MPS cache cleared")
        else:
            logger.info(f"✅ {model_name} unloaded")

    def _run_llava_med(self, image_path: str, prompt: str) -> str:
        """Run LLaVA-Med inference using Python API.

        Args:
            image_path: Path to X-ray image
            prompt: Vision prompt

        Returns:
            LLaVA-Med response text
        """
        # Lazy import to avoid loading model on init
        import sys
        from pathlib import Path

        # Add LLaVA-Med to path
        llava_med_path = Path.home() / "www/tcc/LLaVA-Med"
        if not llava_med_path.exists():
            raise FileNotFoundError(
                f"LLaVA-Med not found at {llava_med_path}. "
                "Run: git clone https://github.com/microsoft/LLaVA-Med.git"
            )

        sys.path.insert(0, str(llava_med_path))

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path, process_images
        from llava.conversation import conv_templates
        from PIL import Image
        import torch

        model_path = "microsoft/llava-med-v1.5-mistral-7b"

        # Load model if not already loaded
        if not hasattr(self, "_llava_med_model"):
            logger.info("Loading LLaVA-Med model...")
            model_name = get_model_name_from_path(model_path)
            # Detect device: mps (Mac) or cpu (fallback)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device=device,  # Override CUDA default (device_map will be set internally)
            )
            self._llava_med_model = (tokenizer, model, image_processor, context_len)
            logger.info("✓ LLaVA-Med loaded")
        else:
            tokenizer, model, image_processor, context_len = self._llava_med_model

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Prepare conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], f"<image>\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer([prompt_formatted], return_tensors="pt").input_ids.to(model.device)

        # Generate
        logger.info(f"Running LLaVA-Med inference on {Path(image_path).name}...")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=512,
                use_cache=True,
                do_sample=True,
                temperature=0.2,
            )

        # Decode
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

    def _unload_llava_med(self) -> None:
        """Unload LLaVA-Med model to free MPS memory before loading BioMistral."""
        if hasattr(self, "_llava_med_model"):
            logger.info("Unloading LLaVA-Med to free MPS memory...")
            tokenizer, model, image_processor, context_len = self._llava_med_model

            # Move model to CPU and delete
            model.cpu()
            del tokenizer, model, image_processor, context_len
            del self._llava_med_model

            # Clear MPS cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Force garbage collection
            import gc

            gc.collect()

            logger.info("✓ LLaVA-Med unloaded, MPS memory freed")

    def _run_llava_ollama(self, image_path: str, prompt: str) -> str:
        """Run llava-llama3 via Ollama.

        Args:
            image_path: Path to X-ray image
            prompt: Vision prompt

        Returns:
            Ollama response text
        """
        cmd = [
            "ollama",
            "run",
            "llava-llama3",
            prompt,
            f"Image: {image_path}",
        ]

        logger.info(f"Running llava-llama3 on {image_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(f"Ollama error: {result.stderr}")
            raise RuntimeError(f"Ollama inference failed: {result.stderr}")

        response = result.stdout.strip()
        logger.info(f"Ollama response: {response[:100]}...")
        return response

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
                findings=None,
                include_guidelines=True,
                include_reports=False,
            )

        # Build vision prompt
        vision_prompt = build_vision_prompt(use_rag=self.use_rag, rag_context=rag_context)

        # Run vision model
        if self.vision_model_name == "llava-med":
            vision_output = self._run_llava_med(image_path, vision_prompt)
        else:
            vision_output = self._run_llava_ollama(image_path, vision_prompt)

        elapsed = time.time() - start_time
        logger.info(f"Stage 1 complete in {elapsed:.1f}s")

        return {
            "findings_en": vision_output,  # For evaluation compatibility
            "findings": vision_output,  # Legacy compatibility
            "model": self.vision_model_name,
            "rag_context": rag_context,
            "latency_s": elapsed,
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

        # Unload LLaVA-Med if it was used (free MPS memory before loading BioMistral)
        self._unload_llava_med()

        # Load BioMistral
        self._load_biomistral()

        # Build RAG context
        rag_context = ""
        if self.use_rag:
            diagnosis = cnn_diagnosis.get("prediction", "").lower()
            rag_context = self.rag_retriever.build_rag_context(
                diagnosis=diagnosis,
                findings=vision_result["findings"],
                include_guidelines=True,
                include_reports=True,
            )

        # Build medical text prompt
        medical_prompt = build_medical_text_prompt(
            vision_description=vision_result["findings"],
            cnn_diagnosis=cnn_diagnosis,
            use_rag=self.use_rag,
            rag_context=rag_context,
        )

        # Format prompt using BioMistral's chat template (Mistral format with [INST] tags)
        formatted_prompt = self._biomistral_tokenizer.apply_chat_template(
            [{"role": "user", "content": medical_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate medical report
        inputs = self._biomistral_tokenizer(formatted_prompt, return_tensors="pt")
        # device_map="auto" handles device placement automatically
        inputs = {k: v.to(self._biomistral_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._biomistral_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,  # Increased from 0.3 for better synthesis
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._biomistral_tokenizer.eos_token_id,
            )

        medical_report = self._biomistral_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        elapsed = time.time() - start_time
        logger.info(f"Stage 2 complete in {elapsed:.1f}s")
        logger.info(f"Medical report (EN): {medical_report}")

        return {
            "report_en": medical_report.strip(),
            "model": "BioMistral-7B",
            "rag_context": rag_context,
            "latency_s": elapsed,
        }

    def analyze_xray(
        self, image_path: str, cnn_diagnosis: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Complete end-to-end X-ray analysis with sequential model loading.

        Memory Management:
        - Stage 1: Vision model (LLaVA-Med or Ollama)
        - Stage 2: BioMistral (load → generate → unload)

        Peak memory: ~13GB

        Args:
            image_path: Path to chest X-ray image
            cnn_diagnosis: Optional CNN classification result

        Returns:
            Complete analysis results (2 stages)
        """
        logger.info(f"\n{'=' * 80}\nAnalyzing X-ray: {image_path}\n{'=' * 80}")

        # Memory monitoring (optional but helpful for debugging)
        if torch.backends.mps.is_available():
            logger.info("Memory: MPS backend available, monitoring enabled")

        pipeline_start = time.time()

        # Default CNN diagnosis if not provided
        if cnn_diagnosis is None:
            cnn_diagnosis = {"prediction": "UNKNOWN", "confidence": 0.0}

        try:
            # Stage 1: Vision analysis
            vision_result = self.stage1_vision_analysis(image_path, cnn_diagnosis)
            logger.info("Memory: Stage 1 complete")

            # Stage 2: Medical text generation (BioMistral)
            medical_result = self.stage2_medical_text(vision_result, cnn_diagnosis)
            logger.info("Memory: Stage 2 complete (BioMistral loaded)")

            # Cleanup: Unload BioMistral
            self._unload_model("biomistral")
            logger.info("Memory: BioMistral unloaded")

            # Combine results
            total_latency = time.time() - pipeline_start

            result = {
                "success": True,
                "image_path": image_path,
                "cnn_diagnosis": cnn_diagnosis,
                "stage1_vision": vision_result,
                "stage2_medical": medical_result,
                "final_report_en": medical_result["report_en"],
                "total_latency_s": total_latency,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(f"\n{'=' * 80}\nPipeline complete in {total_latency:.1f}s\n{'=' * 80}")
            logger.info(f"\nFINAL REPORT (EN):\n{result['final_report_en']}\n")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
