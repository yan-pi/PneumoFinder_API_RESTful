"""Radiology guidelines for pneumonia diagnosis.

Evidence-based guidelines from:
- American College of Radiology (ACR)
- Fleischner Society
- Radiological Society of North America (RSNA)
- Centers for Disease Control and Prevention (CDC)
- British Thoracic Society (BTS)
- Infectious Diseases Society of America (IDSA)
"""

PNEUMONIA_GUIDELINES = [
    # === CLASSIC PNEUMONIA PATTERNS ===
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
    # === PNEUMONIA SUBTYPES ===
    {
        "title": "Aspiration Pneumonia - Dependent Distribution",
        "text": (
            "Aspiration pneumonia typically affects gravity-dependent lung segments: "
            "posterior segments of upper lobes and superior segments of lower lobes in "
            "supine patients. Bilateral lower lobes in upright position. Associated with "
            "dysphagia, altered consciousness, or poor dentition."
        ),
        "source": "IDSA Guidelines 2019",
        "category": "pathology",
    },
    {
        "title": "Atypical Pneumonia - Subtle Findings",
        "text": (
            "Atypical pneumonia (Mycoplasma, Chlamydia, Legionella) may show subtle "
            "interstitial patterns, patchy ground-glass opacities, or appear near-normal. "
            "Often bilateral with lower lobe predominance. Clinical symptoms may exceed "
            "radiographic findings. Follow-up imaging may be necessary."
        ),
        "source": "BTS Guidelines",
        "category": "pathology",
    },
    {
        "title": "Hospital-Acquired Pneumonia (HAP/VAP)",
        "text": (
            "Hospital-acquired pneumonia develops ≥48h after hospital admission. Often "
            "multidrug-resistant organisms (MRSA, Pseudomonas). Radiographic features: "
            "new or progressive infiltrates, consolidation, ground-glass opacities. "
            "Cavitation suggests necrotizing pneumonia or abscess formation."
        ),
        "source": "IDSA/ATS HAP/VAP Guidelines 2016",
        "category": "pathology",
    },
    {
        "title": "Community-Acquired Pneumonia (CAP) - Severity Assessment",
        "text": (
            "CAP radiographic severity correlates with clinical outcomes. Multilobar "
            "involvement, rapid progression (>50% in 48h), and bilateral disease "
            "indicate severe CAP. Pleural effusion presence increases mortality risk. "
            "CURB-65 score guides disposition: 0-1 outpatient, 2 consider admission, ≥3 ICU."
        ),
        "source": "BTS/NICE Guidelines",
        "category": "severity",
    },
    {
        "title": "Viral Pneumonia Patterns",
        "text": (
            "Viral pneumonia typically shows bilateral, diffuse, ground-glass opacities "
            "with interstitial pattern. Lower lobe predominance common. COVID-19 shows "
            "peripheral, posterior distribution. Influenza may progress rapidly to ARDS. "
            "RSV in children shows hyperinflation with perihilar infiltrates."
        ),
        "source": "Fleischner Society COVID Statement",
        "category": "pathology",
    },
    {
        "title": "Pediatric Pneumonia Considerations",
        "text": (
            "Pediatric pneumonia differs from adult patterns. Viral causes predominate "
            "in young children. Round pneumonia (spherical consolidation) is classic "
            "in children under 8. Thymic sail sign may mimic upper lobe pathology. "
            "Hyperinflation with perihilar infiltrates suggests bronchiolitis/RSV."
        ),
        "source": "ACR Pediatric Imaging",
        "category": "pathology",
    },
    # === NORMAL CRITERIA ===
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
        "title": "Normal Variants - Do Not Mistake for Disease",
        "text": (
            "Common normal variants: azygos fissure (1%), pectus excavatum (cardiac "
            "displacement), scoliosis (asymmetric lung density), prominent nipple shadows "
            "(bilateral, symmetric), anterior junction line, calcified costal cartilage. "
            "Comparison with priors helps distinguish variants from new findings."
        ),
        "source": "RSNA Teaching Files",
        "category": "normal",
    },
    # === ANATOMY ===
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
        "title": "Lung Zones and Segmental Anatomy",
        "text": (
            "For practical description, divide each lung into upper, middle, and lower "
            "zones (not lobes). Upper zone: above anterior 2nd rib. Lower zone: below "
            "anterior 4th rib. Middle zone: in between. Report location as 'right lower "
            "zone' rather than 'right lower lobe' unless confident of anatomic localization."
        ),
        "source": "ACR Reporting Guidelines",
        "category": "anatomy",
    },
    {
        "title": "Hilar Anatomy and Pathology",
        "text": (
            "Normal hilum contains pulmonary arteries, veins, bronchi, and lymph nodes. "
            "Left hilum 2cm higher than right. Hilar enlargement may indicate: lymphadenopathy, "
            "pulmonary artery enlargement, or mass. Central opacities with air bronchograms "
            "extending from hilum suggest central pneumonia."
        ),
        "source": "RSNA Thoracic Imaging",
        "category": "anatomy",
    },
    # === RADIOLOGIC SIGNS ===
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
        "title": "Spine Sign - Lower Lobe Consolidation",
        "text": (
            "On lateral chest X-ray, thoracic vertebrae normally become more lucent "
            "inferiorly. Loss of this progressive lucency (spine sign) indicates "
            "lower lobe consolidation or collapse. Useful for detecting retrocardiac "
            "pneumonia obscured on frontal view."
        ),
        "source": "RSNA Teaching Files",
        "category": "signs",
    },
    {
        "title": "Deep Sulcus Sign - Pneumothorax Indicator",
        "text": (
            "In supine patients, pneumothorax collects anteriorly and basally. Deep "
            "sulcus sign: deep, lucent costophrenic angle extending toward ipsilateral "
            "hypochondrium. May be the only sign of supine pneumothorax. Compare with "
            "contralateral side."
        ),
        "source": "ACR Emergency Radiology",
        "category": "signs",
    },
    # === ASSOCIATED FINDINGS ===
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
    {
        "title": "Parapneumonic Effusion vs Empyema",
        "text": (
            "Simple parapneumonic effusion: small, free-flowing, responds to antibiotics. "
            "Complicated effusion: loculated, >1cm on decubitus, pH <7.2, requires drainage. "
            "Empyema: purulent pleural fluid, often with split pleura sign on CT. "
            "Radiographs may show D-shaped loculated collection."
        ),
        "source": "BTS Pleural Disease Guidelines",
        "category": "complications",
    },
    {
        "title": "Lung Abscess - Cavitary Lesion",
        "text": (
            "Lung abscess appears as thick-walled cavity with air-fluid level. Wall "
            ">4mm thick distinguishes from cyst or bulla. Crosses fissures (vs empyema). "
            "Common locations: posterior segments of upper lobes, superior segments of "
            "lower lobes. Associated with aspiration, poor dentition, immunocompromise."
        ),
        "source": "IDSA Guidelines",
        "category": "complications",
    },
    {
        "title": "ARDS - Bilateral Opacities",
        "text": (
            "Acute Respiratory Distress Syndrome presents with bilateral, diffuse "
            "opacities within 7 days of clinical insult. Opacities not fully explained "
            "by effusion, atelectasis, or nodules. Ranges from patchy ground-glass "
            "to dense consolidation. Severity correlates with PaO2/FiO2 ratio."
        ),
        "source": "Berlin Definition/ARDS Network",
        "category": "complications",
    },
    # === DIFFERENTIAL DIAGNOSIS ===
    {
        "title": "Atelectasis vs Pneumonia",
        "text": (
            "Key differences: Atelectasis shows volume loss (elevated hemidiaphragm, "
            "mediastinal shift toward opacity, crowded ribs). Pneumonia typically "
            "maintains or increases volume. Air bronchograms present in both. Clinical "
            "context (fever, WBC) helps differentiate. Prior imaging comparison essential."
        ),
        "source": "ACR Teaching Files",
        "category": "differential",
    },
    {
        "title": "Pulmonary Edema vs Pneumonia",
        "text": (
            "Cardiogenic pulmonary edema: bilateral, symmetric, perihilar (butterfly "
            "pattern), cardiomegaly, vascular redistribution, Kerley B lines, pleural "
            "effusions. Pneumonia: typically asymmetric, lobar or patchy, normal heart "
            "size. Combined heart failure and pneumonia common in elderly."
        ),
        "source": "Fleischner Society",
        "category": "differential",
    },
    {
        "title": "Mass vs Consolidation",
        "text": (
            "Pneumonia: ill-defined borders, air bronchograms, rapid evolution on "
            "serial imaging, responds to antibiotics. Mass lesion: well-defined margins, "
            "may have spiculations, stable or slow growth, no air bronchograms. "
            "Follow-up imaging in 6-8 weeks recommended for any consolidation to ensure resolution."
        ),
        "source": "Fleischner Society Guidelines",
        "category": "differential",
    },
    # === SEVERITY ASSESSMENT ===
    {
        "title": "CURB-65 Severity Score",
        "text": (
            "CURB-65 predicts 30-day mortality in CAP: Confusion, Urea >7mmol/L, "
            "Respiratory rate ≥30, Blood pressure <90/60, Age ≥65. Score 0-1: outpatient "
            "(1.5% mortality). Score 2: consider admission (9.2%). Score 3-5: severe, "
            "consider ICU (22% mortality). Radiographic extent adds prognostic value."
        ),
        "source": "BTS CAP Guidelines",
        "category": "severity",
    },
    {
        "title": "Radiographic Progression Indicators",
        "text": (
            "Concerning progression signs: >50% increase in infiltrate size within 48h, "
            "development of multilobar involvement, new pleural effusion, cavity formation, "
            "new bilateral involvement. Rapid progression despite antibiotics suggests "
            "severe infection, resistant organism, or incorrect diagnosis."
        ),
        "source": "IDSA/ATS Severe CAP Criteria",
        "category": "severity",
    },
    # === TECHNICAL QUALITY ===
    {
        "title": "Chest X-ray Technical Quality Assessment",
        "text": (
            "Adequate PA chest X-ray: full inspiration (≥9 posterior ribs visible), "
            "no rotation (spinous processes equidistant from clavicle heads), proper "
            "penetration (spine barely visible through heart), scapulae projected off "
            "lung fields. Suboptimal technique may obscure pathology or create artifacts."
        ),
        "source": "ACR Practice Parameters",
        "category": "technical",
    },
    {
        "title": "AP vs PA Chest X-ray Differences",
        "text": (
            "AP (anteroposterior) views magnify heart and widen mediastinum. Common "
            "in portable/bedside studies. Do not diagnose cardiomegaly on AP view. "
            "AP technique may obscure subtle lung findings. Always note technique "
            "in report when suboptimal. PA preferred when patient can stand."
        ),
        "source": "RSNA Teaching Guidelines",
        "category": "technical",
    },
    # === FOLLOW-UP RECOMMENDATIONS ===
    {
        "title": "Pneumonia Follow-up Imaging",
        "text": (
            "Radiographic resolution of CAP takes 2-6 weeks. Follow-up X-ray at 6-8 "
            "weeks recommended for: smokers >50 years, persistent symptoms, atypical "
            "presentation, recurrent pneumonia. Earlier imaging if clinical deterioration. "
            "Failure to resolve suggests underlying malignancy, TB, or organizing pneumonia."
        ),
        "source": "BTS Follow-up Guidelines",
        "category": "management",
    },
    {
        "title": "When to Recommend CT",
        "text": (
            "CT indicated for: complicated pneumonia (abscess, empyema suspected), "
            "immunocompromised patients, atypical presentation, failure to respond to "
            "treatment, underlying malignancy concern, recurrent pneumonia same location. "
            "CT better defines extent, detects cavitation, and identifies complications."
        ),
        "source": "ACR Appropriateness Criteria",
        "category": "management",
    },
]

NORMAL_REPORT_EXAMPLES = [
    # === ENGLISH NORMAL REPORTS ===
    {
        "title": "Normal Adult Chest X-ray - English",
        "text": (
            "The lungs are clear bilaterally with no focal consolidation, pleural effusion, "
            "or pneumothorax. The cardiac silhouette is within normal limits. The mediastinum "
            "is unremarkable. No acute cardiopulmonary abnormality."
        ),
        "language": "en",
        "category": "normal",
    },
    {
        "title": "Normal Chest - Standard Report",
        "text": (
            "Heart size is normal. The lungs are clear without focal airspace disease. "
            "No pleural effusion or pneumothorax. The visualized osseous structures are "
            "intact. Normal chest radiograph."
        ),
        "language": "en",
        "category": "normal",
    },
    {
        "title": "Normal Chest - Detailed",
        "text": (
            "The cardiomediastinal silhouette is within normal limits. The lungs are "
            "clear without evidence of consolidation, mass, or nodule. Costophrenic "
            "angles are sharp. The pulmonary vasculature appears normal. No pneumothorax "
            "or pleural effusion identified."
        ),
        "language": "en",
        "category": "normal",
    },
    # === PORTUGUESE NORMAL REPORTS ===
    {
        "title": "Normal Adult Chest X-ray - Portuguese",
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
        "title": "Normal Chest - No Acute Findings - Portuguese",
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
    # === ENGLISH PNEUMONIA REPORTS ===
    {
        "title": "Right Lower Lobe Pneumonia - English",
        "text": (
            "There is consolidation in the right lower lobe with air bronchograms. "
            "The remainder of the lungs is clear. No pleural effusion. Heart size is normal. "
            "Findings are consistent with right lower lobe pneumonia. Clinical correlation "
            "recommended."
        ),
        "language": "en",
        "category": "pneumonia",
    },
    {
        "title": "Bilateral Pneumonia - English",
        "text": (
            "Bilateral patchy airspace opacities are present, predominantly in the lower "
            "lobes with additional involvement of the right middle lobe. Small bilateral "
            "pleural effusions. Findings are consistent with multifocal pneumonia. "
            "Recommend clinical correlation and follow-up imaging after treatment."
        ),
        "language": "en",
        "category": "pneumonia",
    },
    {
        "title": "Left Lower Lobe Consolidation - English",
        "text": (
            "Dense consolidation in the left lower lobe partially obscuring the left "
            "hemidiaphragm. Air bronchograms are present. Small left pleural effusion. "
            "The right lung is clear. Cardiac silhouette is normal. Findings suggest "
            "left lower lobe pneumonia with parapneumonic effusion."
        ),
        "language": "en",
        "category": "pneumonia",
    },
    {
        "title": "Mild Pneumonia - Subtle Findings",
        "text": (
            "There is subtle ground-glass opacity in the right lower lobe. No definite "
            "consolidation. No pleural effusion. Cardiac silhouette is normal. Findings "
            "may represent early pneumonia or atelectasis. Clinical correlation and "
            "short-term follow-up recommended if symptoms persist."
        ),
        "language": "en",
        "category": "pneumonia",
    },
    {
        "title": "Severe Bilateral Pneumonia - English",
        "text": (
            "Extensive bilateral consolidation involving all lobes with diffuse ground-glass "
            "opacities. Bilateral pleural effusions. Air bronchograms throughout. Cardiac "
            "silhouette obscured by confluent opacities. Severe bilateral pneumonia. "
            "Findings concerning for ARDS. Urgent clinical correlation required."
        ),
        "language": "en",
        "category": "pneumonia",
    },
    # === PORTUGUESE PNEUMONIA REPORTS ===
    {
        "title": "Lobar Pneumonia - Right Lower Lobe - Portuguese",
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
        "title": "Bilateral Bronchopneumonia - Portuguese",
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
