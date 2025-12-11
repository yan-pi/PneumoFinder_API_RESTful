"""Medical terminology database (EN→PT) for accurate translations.

This database prevents mistranslations of medical terms during PT-BR generation.
Covers radiology-specific anatomy, pathology, and clinical terminology.
"""

# Anatomical terms - CRITICAL for preventing left/right confusion
ANATOMICAL_TERMS = {
    # Chest anatomy
    "right lung": "pulmão direito",
    "left lung": "pulmão esquerdo",
    "right upper lobe": "lobo superior direito",
    "right middle lobe": "lobo médio direito",
    "right lower lobe": "lobo inferior direito",
    "left upper lobe": "lobo superior esquerdo",
    "left lower lobe": "lobo inferior esquerdo",
    "right hemithorax": "hemitórax direito",
    "left hemithorax": "hemitórax esquerdo",
    "right costophrenic angle": "ângulo costofrênico direito",
    "left costophrenic angle": "ângulo costofrênico esquerdo",
    "right heart border": "borda cardíaca direita",
    "left heart border": "borda cardíaca esquerda",
    "right hilum": "hilo direito",
    "left hilum": "hilo esquerdo",
    "right apex": "ápice direito",
    "left apex": "ápice esquerdo",
    "right base": "base direita",
    "left base": "base esquerda",
    "mediastinum": "mediastino",
    "trachea": "traqueia",
    "carina": "carina",
    "diaphragm": "diafragma",
    "pleura": "pleura",
    "pericardium": "pericárdio",
    "cardiac silhouette": "silhueta cardíaca",
    "aortic arch": "arco aórtico",
    "pulmonary vessels": "vasos pulmonares",
    "bronchi": "brônquios",
    "alveoli": "alvéolos",
    "interstitium": "interstício",
    "parenchyma": "parênquima",
    # Skeletal structures
    "ribs": "costelas",
    "clavicle": "clavícula",
    "scapula": "escápula",
    "sternum": "esterno",
    "vertebrae": "vértebras",
    "thoracic spine": "coluna torácica",
    "soft tissues": "tecidos moles",
}

# Pathological findings
PATHOLOGY_TERMS = {
    # Pneumonia-specific
    "consolidation": "consolidação",
    "opacity": "opacidade",
    "infiltrate": "infiltrado",
    "air bronchogram": "broncograma aéreo",
    "ground-glass opacity": "opacidade em vidro fosco",
    "patchy infiltrates": "infiltrados em placas",
    "lobar pneumonia": "pneumonia lobar",
    "bronchopneumonia": "broncopneumonia",
    "interstitial pneumonia": "pneumonia intersticial",
    "bacterial pneumonia": "pneumonia bacteriana",
    "viral pneumonia": "pneumonia viral",
    "aspiration pneumonia": "pneumonia aspirativa",
    # General pathology
    "effusion": "derrame",
    "pleural effusion": "derrame pleural",
    "atelectasis": "atelectasia",
    "edema": "edema",
    "pulmonary edema": "edema pulmonar",
    "fibrosis": "fibrose",
    "nodule": "nódulo",
    "mass": "massa",
    "cavitation": "cavitação",
    "abscess": "abscesso",
    "pneumothorax": "pneumotórax",
    "hemothorax": "hemotórax",
    "emphysema": "enfisema",
    "bronchiectasis": "bronquiectasia",
    "cardiomegaly": "cardiomegalia",
    "hyperinflation": "hiperinsuflação",
    "congestion": "congestão",
    "lymphadenopathy": "linfonodomegalia",
    # Findings descriptors
    "bilateral": "bilateral",
    "unilateral": "unilateral",
    "diffuse": "difuso",
    "focal": "focal",
    "multifocal": "multifocal",
    "confluent": "confluente",
    "scattered": "disperso",
    "peripheral": "periférico",
    "central": "central",
    "perihilar": "peri-hilar",
    "basal": "basal",
    "apical": "apical",
    "mild": "leve",
    "moderate": "moderado",
    "severe": "grave",
    "acute": "agudo",
    "chronic": "crônico",
    "subtle": "sutil",
    "prominent": "proeminente",
    "increased": "aumentado",
    "decreased": "diminuído",
    "normal": "normal",
    "abnormal": "anormal",
    "unremarkable": "sem alterações",
}

# Clinical and diagnostic terms
CLINICAL_TERMS = {
    # Diagnosis
    "pneumonia": "pneumonia",
    "infection": "infecção",
    "inflammation": "inflamação",
    "disease": "doença",
    "condition": "condição",
    "lesion": "lesão",
    "finding": "achado",
    "abnormality": "anormalidade",
    # Imaging terms
    "chest X-ray": "raio-X de tórax",
    "radiograph": "radiografia",
    "posteroanterior view": "incidência posteroanterior",
    "lateral view": "incidência lateral",
    "AP view": "incidência anteroposterior",
    "upright": "em pé",
    "supine": "em decúbito dorsal",
    "portable": "portátil",
    "bedside": "à beira do leito",
    # Assessment terms
    "compatible with": "compatível com",
    "consistent with": "consistente com",
    "suggestive of": "sugestivo de",
    "suspicious for": "suspeito para",
    "likely": "provável",
    "possible": "possível",
    "cannot exclude": "não se pode excluir",
    "rule out": "excluir",
    "differential diagnosis": "diagnóstico diferencial",
    "clinical correlation": "correlação clínica",
    "follow-up": "acompanhamento",
    "comparison": "comparação",
    "previous study": "exame anterior",
    "interval": "intervalo",
    "progression": "progressão",
    "improvement": "melhora",
    "stable": "estável",
    "unchanged": "inalterado",
    "resolved": "resolvido",
    # Recommendations
    "recommend": "recomenda-se",
    "suggest": "sugere-se",
    "correlation": "correlação",
    "clinical evaluation": "avaliação clínica",
    "further imaging": "investigação radiológica adicional",
    "CT scan": "tomografia computadorizada",
    "ultrasound": "ultrassonografia",
    "MRI": "ressonância magnética",
}

# Negative findings (critical for normal X-rays)
NEGATIVE_FINDINGS = {
    "no evidence of": "sem evidência de",
    "no signs of": "sem sinais de",
    "no acute": "sem alterações agudas",
    "no focal": "sem alterações focais",
    "clear lungs": "pulmões claros",
    "clear lung fields": "campos pulmonares claros",
    "unremarkable": "sem alterações",
    "within normal limits": "dentro dos limites da normalidade",
    "normal": "normal",
    "no abnormality": "sem anormalidades",
    "no pathology": "sem patologia",
    "negative": "negativo",
}

# Combine all dictionaries
MEDICAL_TERMINOLOGY_EN_PT = {
    **ANATOMICAL_TERMS,
    **PATHOLOGY_TERMS,
    **CLINICAL_TERMS,
    **NEGATIVE_FINDINGS,
}


def get_all_terms() -> list[tuple[str, str]]:
    """Get all medical terms as (EN, PT) tuples.

    Returns:
        List of (english_term, portuguese_term) tuples
    """
    return list(MEDICAL_TERMINOLOGY_EN_PT.items())


def translate_term(english_term: str) -> str | None:
    """Translate a medical term from English to Portuguese.

    Args:
        english_term: English medical term

    Returns:
        Portuguese translation or None if not found
    """
    return MEDICAL_TERMINOLOGY_EN_PT.get(english_term.lower())


def get_term_count() -> int:
    """Get total number of terms in the database.

    Returns:
        Total number of terms
    """
    return len(MEDICAL_TERMINOLOGY_EN_PT)
