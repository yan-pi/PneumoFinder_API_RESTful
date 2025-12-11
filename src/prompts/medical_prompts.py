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
    prompt = f"""You are a radiologist writing a professional chest X-ray report.

INPUT DATA:
1. Vision Analysis: {vision_description}
2. CNN Classification: {cnn_diagnosis.get("prediction", "Unknown")} (Confidence: {cnn_diagnosis.get("confidence", 0.0):.1%})

"""

    if use_rag and rag_context:
        prompt += f"""
MEDICAL REFERENCE KNOWLEDGE:
{rag_context}

"""

    prompt += """
TASK: Generate a concise, professional radiology report in English.

REQUIREMENTS:
1. Synthesize vision analysis and CNN classification
2. Use standard radiology terminology
3. If findings contradict (e.g., CNN says pneumonia but vision sees clear lungs):
   - Trust the vision analysis for anatomical details
   - Note the discrepancy: "Clinical correlation recommended"
4. For NORMAL X-rays:
   - Use phrases like "no acute cardiopulmonary abnormality"
   - Explicitly state "lungs are clear bilaterally"
   - Do NOT fabricate findings
5. For ABNORMAL X-rays:
   - Describe pattern (lobar, interstitial, patchy)
   - Specify location precisely (e.g., "right middle lobe consolidation")
   - Mention associated findings (effusion, air bronchograms)

OUTPUT FORMAT:
- 2-3 sentences maximum
- English only (translation happens next)
- Professional, concise, accurate

Generate the radiology report:"""

    return prompt


def build_translation_prompt(
    english_report: str,
    medical_terms_dict: dict[str, str] | None = None,
) -> str:
    """Build prompt for PT-BR translation (Sabiá-7B).

    Args:
        english_report: English radiology report
        medical_terms_dict: EN→PT medical terminology for reference

    Returns:
        Formatted prompt for Sabiá
    """
    prompt = f"""Você é um tradutor médico especializado em radiologia.

TAREFA: Traduzir o laudo radiológico abaixo para português brasileiro (PT-BR).

LAUDO EM INGLÊS:
{english_report}

"""

    if medical_terms_dict:
        terms_list = "\n".join(
            [f"- {en} → {pt}" for en, pt in list(medical_terms_dict.items())[:20]]
        )
        prompt += f"""
TERMINOLOGIA MÉDICA DE REFERÊNCIA:
{terms_list}
[...]

Use estes termos médicos corretos na tradução.

"""

    prompt += """
INSTRUÇÕES DE TRADUÇÃO:
1. Manter terminologia médica profissional em PT-BR
2. Não adicionar nem remover informações clínicas
3. Preservar a concisão do original (2-3 frases)
4. Usar termos padronizados:
   - "pulmões" (não "os pulmões")
   - "ausência de" (não "sem")
   - "campos pulmonares claros" para normal
   - "consolidação" para consolidation
   - "derrame pleural" para pleural effusion
5. CRÍTICO: Respeitar lateralidade:
   - "right lung" = "pulmão direito"
   - "left lower lobe" = "lobo inferior esquerdo"
6. Tradução natural, fluente, como escrita por radiologista brasileiro

FORMATO DE SAÍDA:
- Apenas o laudo traduzido
- Sem explicações adicionais
- 2-3 frases em PT-BR

Tradução:"""

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
        "translation_prompt": build_translation_prompt(
            english_report=vision_description,
            medical_terms_dict=None,  # Sabiá will use its training
        ),
        "rag_context": rag_context,
    }
