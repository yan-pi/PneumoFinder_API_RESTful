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

Use the above guidelines to inform your analysis, but describe ONLY what you observe in this specific image.

"""

    base_prompt += """
OUTPUT FORMAT:
- 2-3 concise sentences
- English only (will be translated later)
- Professional medical terminology
- If normal: state clearly "The lungs are clear bilaterally. No acute cardiopulmonary abnormality."

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
    # Get diagnosis info
    diagnosis = cnn_diagnosis.get("prediction", "Unknown")
    confidence = cnn_diagnosis.get("confidence", 0.0)

    # Build conversational prompt that encourages synthesis, not echoing
    prompt = f"""A vision AI analyzed a chest X-ray and reported: "{vision_description}"

A CNN classifier predicted: {diagnosis} with {confidence:.0%} confidence.

"""

    if use_rag and rag_context:
        prompt += f"""Relevant medical knowledge:
{rag_context}

"""

    prompt += """Write a professional 2-3 sentence radiology report that synthesizes these findings. Use standard medical terminology and be specific about:
- Anatomical locations (e.g., "right lower lobe", "bilateral bases")
- Pattern of abnormality if present (consolidation, infiltrate, opacity)
- Clinical assessment or recommendation

If the findings are normal, state "No acute cardiopulmonary abnormality" or similar.
If findings contradict the CNN prediction, note "Clinical correlation recommended."

Radiology Report:"""

    return prompt


def build_rag_enhanced_pipeline_prompts(
    rag_retriever,
    diagnosis: str,
    vision_description: str = "",
    cnn_diagnosis: dict | None = None,
) -> dict[str, str]:
    """Build all pipeline prompts with RAG context.

    Args:
        rag_retriever: MedicalRetriever instance
        diagnosis: Predicted diagnosis (for RAG retrieval)
        vision_description: Vision model output (for medical text prompt)
        cnn_diagnosis: CNN classification result

    Returns:
        Dictionary with prompts for each pipeline stage
    """
    # Retrieve RAG context
    rag_context = rag_retriever.build_rag_context(
        diagnosis=diagnosis,
        findings=vision_description if vision_description else None,
        include_guidelines=True,
        include_reports=bool(vision_description),
    )

    # Get medical terminology for translation
    guidelines = rag_retriever.get_relevant_guidelines(diagnosis, n_results=5)

    # Build prompts for each stage
    return {
        "vision_prompt": build_vision_prompt(use_rag=True, rag_context=rag_context),
        "medical_text_prompt": build_medical_text_prompt(
            vision_description=vision_description,
            cnn_diagnosis=cnn_diagnosis or {},
            use_rag=True,
            rag_context=rag_context,
        ),
        "rag_context": rag_context,
    }
