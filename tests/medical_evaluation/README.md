# Avaliação Comparativa de Modelos LLaVA para Diagnóstico de Pneumonia

## 🎯 Objetivo

Avaliar empiricamente qual modelo LLaVA (7B, 13B, ou Llama3) gera descrições clínicas mais precisas e úteis para diagnóstico auxiliado por IA de pneumonia em radiografias de tórax.

## 📊 Estrutura do Estudo

### Modelos Testados
1. **LLaVA 7B** - Modelo base (4.7GB)
2. **LLaVA 13B** - Modelo grande (8.0GB)
3. **LLaVA-Llama3** - Modelo moderno baseado em Llama 3 (5.5GB)

### Casos Clínicos
Todos os 3 modelos analisaram os **MESMOS 5 casos**:
- **Caso 1:** Pulmão normal (caso claro)
- **Caso 2:** Pneumonia severa
- **Caso 3:** Pulmão normal (caso desafiador)
- **Caso 4:** Pneumonia moderada
- **Caso 5:** ⚠️ **Falso Negativo da CNN** (pneumonia classificada como normal)

**Nota sobre Caso 5:**  
Este caso foi adicionado especificamente para avaliar a robustez dos modelos LLM quando a CNN comete um erro crítico (falso negativo). O objetivo é verificar se as descrições geradas pelos modelos conseguem identificar padrões suspeitos mesmo quando o diagnóstico automatizado está incorreto, demonstrando o valor da análise multimodal como camada adicional de segurança.

## 📁 Arquivos por Caso

Cada caso contém 7 arquivos numerados:

1. **1_original.jpeg** - Radiografia de tórax original
2. **2_gradcam_heatmap.png** - Mapa de calor Grad-CAM (regiões relevantes para CNN)
3. **3_gradcam_overlay.png** - Sobreposição do mapa de calor no raio-X
4. **4_cnn_diagnosis.json** - Diagnóstico da CNN (classe, confiança, latência)
5. **5_llm_description_en.txt** - Descrição clínica em inglês
6. **6_llm_description_pt.txt** - Descrição clínica em português
7. **7_llm_metadata.json** - Metadados técnicos (latências, contagem de palavras)

## 📋 Como Avaliar

1. **Abra o formulário:** `evaluation_form.md`
2. **Para cada modelo** (llava_7b, llava_13b, llava_llama3):
   - Navegue até a pasta do modelo
   - Para cada caso (01 a 05):
     - Visualize a imagem original (`1_original.jpeg`)
     - Observe o Grad-CAM (`3_gradcam_overlay.png`) - áreas em vermelho são onde a CNN focou
     - Leia o diagnóstico da CNN (`4_cnn_diagnosis.json`)
     - **Leia a descrição em português** (`6_llm_description_pt.txt`)
     - Preencha o formulário de avaliação
3. **Compare os 3 modelos** e indique sua recomendação

## ⏱️ Tempo Estimado
- **18-24 minutos** (avaliação completa - 5 casos × 3 modelos)

## 🔬 Critérios de Avaliação

- **Precisão anatômica:** Identificação correta de estruturas e regiões
- **Terminologia médica:** Uso apropriado de termos técnicos
- **Interpretação radiológica:** Correção dos achados descritos
- **Consistência com CNN:** Alinhamento com o diagnóstico automatizado
- **Utilidade clínica:** Valor agregado para tomada de decisão
- **Problemas:** Alucinações, omissões, contradições

## 📊 Dados Técnicos

Detalhes sobre latências, arquitetura da CNN, e parâmetros do LLM estão disponíveis em `evaluation_metadata.json`.

## 📧 Contato

Para dúvidas ou feedback, entre em contato com a equipe do projeto.

---

**Obrigado pela sua contribuição para este estudo!** 🙏
