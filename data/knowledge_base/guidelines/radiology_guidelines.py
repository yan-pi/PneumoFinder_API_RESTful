"""Radiology guidelines for pneumonia diagnosis.

Evidence-based guidelines from:
- American College of Radiology (ACR)
- Fleischner Society
- Radiological Society of North America (RSNA)
"""

PNEUMONIA_GUIDELINES = [
    {
        "title": "Lobar Pneumonia - Classic Presentation",
        "text": (
            "Lobar pneumonia presents as homogeneous consolidation involving an entire "
            "lobe, with sharp demarcation at fissures. Air bronchograms are typically "
            "present. Most commonly affects lower lobes. Caused predominantly by "
            "Streptococcus pneumoniae."
        ),
        "source": "ACR Appropriateness Criteria",
        "category": "pathology",
    },
    {
        "title": "Bronchopneumonia - Patchy Pattern",
        "text": (
            "Bronchopneumonia shows multifocal patchy consolidations or ground-glass "
            "opacities, often bilateral and perihilar. Affects bronchi and adjacent "
            "alveoli. Common in elderly and immunocompromised patients."
        ),
        "source": "RSNA Radiology Reports",
        "category": "pathology",
    },
    {
        "title": "Interstitial Pneumonia - Reticular Pattern",
        "text": (
            "Interstitial pneumonia presents with reticular or ground-glass opacities, "
            "predominantly in lower lobes. Linear opacities and septal thickening may "
            "be present. Often viral etiology (influenza, COVID-19, RSV)."
        ),
        "source": "Fleischner Society",
        "category": "pathology",
    },
    {
        "title": "Normal Chest X-ray Criteria",
        "text": (
            "Normal chest X-ray: clear bilateral lung fields, sharp costophrenic angles, "
            "normal cardiac silhouette (CTR <0.5), visible pulmonary vessels, intact "
            "hemidiaphragms, no pleural effusion, no mediastinal widening."
        ),
        "source": "ACR Teaching File",
        "category": "normal",
    },
    {
        "title": "Left vs Right Lung Identification",
        "text": (
            "On PA chest X-ray, the RIGHT side appears on the LEFT of the image (patient "
            "facing you). Right lung has three lobes (upper, middle, lower); left lung "
            "has two lobes (upper, lower) with lingula. Always verify with cardiac apex "
            "(points left) and gastric bubble (left lower)."
        ),
        "source": "RSNA Training Guidelines",
        "category": "anatomy",
    },
    {
        "title": "Air Bronchogram Sign - Consolidation Indicator",
        "text": (
            "Air bronchograms appear as dark, branching air-filled bronchi visible "
            "against opacified lung parenchyma. Indicates alveolar consolidation "
            "(pneumonia, pulmonary edema, ARDS). Absence does not exclude disease."
        ),
        "source": "Fleischner Society Glossary",
        "category": "signs",
    },
    {
        "title": "Silhouette Sign - Localization Tool",
        "text": (
            "Loss of normal border between cardiac/mediastinal structures and adjacent "
            "lung indicates pathology in that specific lobe. Right heart border loss = "
            "right middle lobe. Left heart border loss = lingula. Useful for anatomical "
            "localization."
        ),
        "source": "ACR Case Studies",
        "category": "signs",
    },
    {
        "title": "Pleural Effusion - Blunted Costophrenic Angle",
        "text": (
            "Pleural effusion presents as blunting of costophrenic angle (requires ~200mL "
            "on upright PA view). Meniscus sign may be visible. Can accompany pneumonia "
            "(parapneumonic effusion). Lateral decubitus view confirms free-flowing fluid."
        ),
        "source": "RSNA Radiology Essentials",
        "category": "associated_findings",
    },
]

NORMAL_REPORT_EXAMPLES = [
    {
        "title": "Normal Adult Chest X-ray",
        "text": (
            "Campos pulmonares claros bilateralmente. Silhueta cardíaca de dimensões "
            "normais. Ângulos costofrênicos agudos preservados. Hilos de aspecto usual. "
            "Mediastino centrado. Arcabouço ósseo íntegro. Sem alterações pleuropulmonares "
            "agudas."
        ),
        "language": "pt-BR",
        "category": "normal",
    },
    {
        "title": "Normal Chest - No Acute Findings",
        "text": (
            "Estudo radiográfico do tórax sem evidência de consolidações parenquimatosas, "
            "massas ou nódulos. Trama vascular pulmonar preservada. Ausência de derrame "
            "pleural. Área cardíaca dentro dos limites da normalidade. Sem sinais de "
            "pneumotórax."
        ),
        "language": "pt-BR",
        "category": "normal",
    },
]

PNEUMONIA_REPORT_EXAMPLES = [
    {
        "title": "Lobar Pneumonia - Right Lower Lobe",
        "text": (
            "Consolidação homogênea em lobo inferior direito com broncogramas aéreos "
            "visíveis. Ângulo costofrênico direito preservado. Restante do parênquima "
            "pulmonar sem alterações. Achados compatíveis com pneumonia lobar à direita. "
            "Correlação clínica e laboratorial recomendada."
        ),
        "language": "pt-BR",
        "category": "pneumonia",
    },
    {
        "title": "Bilateral Bronchopneumonia",
        "text": (
            "Opacidades bilaterais peri-hilares com padrão reticulonodular. Infiltrados "
            "em placas nos terços médios de ambos os pulmões. Bases pulmonares com "
            "opacidades confluentes. Silhueta cardíaca normal. Achados sugestivos de "
            "broncopneumonia bilateral. Acompanhamento radiológico sugerido após "
            "tratamento."
        ),
        "language": "pt-BR",
        "category": "pneumonia",
    },
]


def get_all_guidelines() -> list[dict]:
    """Get all radiology guidelines.

    Returns:
        List of guideline dictionaries
    """
    return PNEUMONIA_GUIDELINES


def get_report_examples(category: str | None = None) -> list[dict]:
    """Get example radiology reports.

    Args:
        category: Filter by category ('normal' or 'pneumonia'), None for all

    Returns:
        List of report example dictionaries
    """
    all_reports = NORMAL_REPORT_EXAMPLES + PNEUMONIA_REPORT_EXAMPLES

    if category:
        return [r for r in all_reports if r.get("category") == category]

    return all_reports
