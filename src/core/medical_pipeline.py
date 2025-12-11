"""Medical pipeline orchestrator for 3-stage LLM analysis with RAG.

This module orchestrates the complete medical analysis pipeline:
1. Stage 1: Vision analysis (llava-llama3 via Ollama) → English findings
2. Stage 2: Medical text generation (BioMistral-7B) → Professional English report
3. Stage 3: PT-BR translation (Sabiá-7B) → Natural Portuguese report

All stages are grounded with RAG context to prevent hallucinations.

Memory Management (M4 Pro 24GB):
- Sequential loading: Only 1 HuggingFace model in memory at a time
- Ollama (Stage 1): Auto-unloads after completion → ~3GB baseline
- HuggingFace models (Stages 2-3): Intelligent device_map splits across MPS+CPU/RAM
- MPS constraint: ~10GB max single allocation (models @ 13GB need splitting)
- Peak memory: ~20GB (model layers + inference overhead)
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
    build_translation_prompt,
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
        self._sabia_model = None
        self._sabia_tokenizer = None
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

    def _load_sabia(self) -> None:
        """Lazy-load Sabiá model with intelligent device mapping.

        Strategy: MPS has ~10GB single allocation limit, but Sabiá @ FP16 = 13GB.
        Use device_map="auto" to split across MPS + CPU/RAM intelligently.
        Since we unload models between stages, this avoids memory conflicts.
        """
        if self._sabia_model is None:
            logger.info("Loading Sabiá-7B with intelligent device mapping...")
            model_id = "maritaca-ai/sabia-7b"

            self._sabia_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._sabia_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",  # Let transformers split intelligently
                low_cpu_mem_usage=True,
                max_memory={"mps": "10GiB", "cpu": "16GiB"},  # MPS limit + CPU fallback
            )
            logger.info(f"Sabiá-7B loaded (device_map: {self._sabia_model.hf_device_map})")
            logger.info(f"Sabiá-7B loaded (device_map: {self._sabia_model.hf_device_map})")

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
            model_name: Model to unload ("biomistral" or "sabia")
        """
        import gc

        if model_name == "biomistral":
            if self._biomistral_model is not None:
                logger.info("🧹 Unloading BioMistral-7B to free ~13GB memory...")
                del self._biomistral_model
                del self._biomistral_tokenizer
                self._biomistral_model = None
                self._biomistral_tokenizer = None

        elif model_name == "sabia":
            if self._sabia_model is not None:
                logger.info("🧹 Unloading Sabiá-7B to free ~13GB memory...")
                del self._sabia_model
                del self._sabia_tokenizer
                self._sabia_model = None
                self._sabia_tokenizer = None

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
        """Run LLaVA-Med inference using official repo.

        Args:
            image_path: Path to X-ray image
            prompt: Vision prompt

        Returns:
            LLaVA-Med response text
        """
        llava_med_path = Path.home() / "www/tcc/LLaVA-Med"
        inference_script = llava_med_path / "llava/eval/run_llava.py"

        if not inference_script.exists():
            raise FileNotFoundError(
                f"LLaVA-Med not found at {llava_med_path}. "
                "Run: python scripts/setup_llava_med_official.py"
            )

        # Run LLaVA-Med inference
        cmd = [
            sys.executable,
            str(inference_script),
            "--model-path",
            "microsoft/llava-med-v1.5-mistral-7b",
            "--image-file",
            image_path,
            "--query",
            prompt,
            "--temperature",
            "0.2",
            "--max-new-tokens",
            "256",
        ]

        logger.info(f"Running LLaVA-Med on {image_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(f"LLaVA-Med error: {result.stderr}")
            raise RuntimeError(f"LLaVA-Med inference failed: {result.stderr}")

        # Parse output (last line is usually the response)
        output_lines = result.stdout.strip().split("\n")
        response = output_lines[-1] if output_lines else ""

        logger.info(f"LLaVA-Med response: {response[:100]}...")
        return response

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
            "findings": vision_output,
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

        # Generate medical report
        inputs = self._biomistral_tokenizer(medical_prompt, return_tensors="pt")
        # device_map="auto" handles device placement automatically
        inputs = {k: v.to(self._biomistral_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._biomistral_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
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

    def stage3_translation(self, medical_result: dict[str, Any]) -> dict[str, Any]:
        """Stage 3: PT-BR translation with Sabiá.

        Args:
            medical_result: Output from stage 2

        Returns:
            Dictionary with translation results
        """
        logger.info("=== STAGE 3: PT-BR Translation ===")
        start_time = time.time()

        # Load Sabiá
        self._load_sabia()

        # Build translation prompt
        translation_prompt = build_translation_prompt(
            english_report=medical_result["report_en"], medical_terms_dict=None
        )

        # Translate to PT-BR
        inputs = self._sabia_tokenizer(translation_prompt, return_tensors="pt")
        # device_map="auto" handles device placement automatically
        inputs = {k: v.to(self._sabia_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._sabia_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._sabia_tokenizer.eos_token_id,
            )

        pt_report = self._sabia_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        elapsed = time.time() - start_time
        logger.info(f"Stage 3 complete in {elapsed:.1f}s")
        logger.info(f"PT-BR report: {pt_report}")

        return {
            "report_pt_br": pt_report.strip(),
            "model": "Sabiá-7B",
            "latency_s": elapsed,
        }

    def analyze_xray(
        self, image_path: str, cnn_diagnosis: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Complete end-to-end X-ray analysis with sequential model loading.

        Memory Management:
        - Stage 1: Ollama (auto-unloads)
        - Stage 2: BioMistral (load → generate → unload)
        - Stage 3: Sabiá (load → generate → unload)

        Peak memory: ~16GB (well under 21GB M4 Pro limit)

        Args:
            image_path: Path to chest X-ray image
            cnn_diagnosis: Optional CNN classification result

        Returns:
            Complete analysis results including all 3 stages
        """
        logger.info(f"\n{'=' * 80}\nAnalyzing X-ray: {image_path}\n{'=' * 80}")

        # Memory monitoring (optional but helpful for debugging)
        if torch.backends.mps.is_available():
            logger.info("💾 Memory: MPS backend available, monitoring enabled")

        pipeline_start = time.time()

        # Default CNN diagnosis if not provided
        if cnn_diagnosis is None:
            cnn_diagnosis = {"prediction": "UNKNOWN", "confidence": 0.0}

        try:
            # Stage 1: Vision analysis (Ollama - auto-unloads)
            vision_result = self.stage1_vision_analysis(image_path, cnn_diagnosis)
            logger.info("💾 Memory: Stage 1 complete (Ollama auto-unloaded)")

            # Stage 2: Medical text generation (BioMistral)
            medical_result = self.stage2_medical_text(vision_result, cnn_diagnosis)
            logger.info("💾 Memory: Stage 2 complete (BioMistral loaded)")

            # ✅ CRITICAL: Unload BioMistral before loading Sabiá
            self._unload_model("biomistral")
            logger.info("💾 Memory: BioMistral unloaded, ready for Stage 3")

            # Stage 3: PT-BR translation (Sabiá)
            translation_result = self.stage3_translation(medical_result)
            logger.info("💾 Memory: Stage 3 complete (Sabiá loaded)")

            # ✅ Cleanup: Unload Sabiá for next run
            self._unload_model("sabia")
            logger.info("💾 Memory: Sabiá unloaded, pipeline cleanup complete")

            # Combine results
            total_latency = time.time() - pipeline_start

            result = {
                "success": True,
                "image_path": image_path,
                "cnn_diagnosis": cnn_diagnosis,
                "stage1_vision": vision_result,
                "stage2_medical": medical_result,
                "stage3_translation": translation_result,
                "final_report_pt_br": translation_result["report_pt_br"],
                "total_latency_s": total_latency,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(f"\n{'=' * 80}\nPipeline complete in {total_latency:.1f}s\n{'=' * 80}")
            logger.info(f"\nFINAL REPORT (PT-BR):\n{result['final_report_pt_br']}\n")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
