# Formulário de Avaliação - Modelos LLaVA

**Avaliador:** _________________________  
**Data:** ___/___/______  
**Especialidade/CRM:** _________________________

---

## Instruções

Você avaliará **3 modelos de IA** que geram descrições clínicas de raios-X de tórax.

- Cada modelo analisou os **MESMOS 5 casos** (2 normais + 3 pneumonia, incluindo 1 falso negativo da CNN)
- Foque na descrição em português: `6_llm_description_pt.txt`
- Compare com: imagem original + overlay Grad-CAM + diagnóstico CNN
- **Tempo estimado:** 18-24 minutos

---

## MODELO 1: LLaVA 7B

### Caso 01 - Normal (Claro)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 02 - Pneumonia (Severa)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 03 - Normal (Desafiador)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 04 - Pneumonia (Moderada)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 05 - Pneumonia (Falso Negativo da CNN) ⚠️

> **ATENÇÃO:** Neste caso, a CNN **ERROU** o diagnóstico (classificou como NORMAL, mas é PNEUMONIA).  
> Avalie se a descrição do LLM consegue identificar achados suspeitos apesar do erro da CNN.

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Robustez do Modelo (específico para este caso):**
- [ ] ✅ Identificou achados suspeitos/anormais apesar da CNN indicar "NORMAL"
- [ ] ⚖️ Descrição neutra/objetiva, sem contradizer nem reforçar o erro
- [ ] ❌ Reforçou o erro da CNN (ex: "pulmões limpos", "sem alterações")

**4. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

## MODELO 2: LLaVA 13B

### Caso 01 - Normal (Claro)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 02 - Pneumonia (Severa)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 03 - Normal (Desafiador)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 04 - Pneumonia (Moderada)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 05 - Pneumonia (Falso Negativo da CNN) ⚠️

> **ATENÇÃO:** Neste caso, a CNN **ERROU** o diagnóstico (classificou como NORMAL, mas é PNEUMONIA).  
> Avalie se a descrição do LLM consegue identificar achados suspeitos apesar do erro da CNN.

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Robustez do Modelo (específico para este caso):**
- [ ] ✅ Identificou achados suspeitos/anormais apesar da CNN indicar "NORMAL"
- [ ] ⚖️ Descrição neutra/objetiva, sem contradizer nem reforçar o erro
- [ ] ❌ Reforçou o erro da CNN (ex: "pulmões limpos", "sem alterações")

**4. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

## MODELO 3: LLaVA-Llama3

### Caso 01 - Normal (Claro)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 02 - Pneumonia (Severa)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 03 - Normal (Desafiador)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 04 - Pneumonia (Moderada)

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

### Caso 05 - Pneumonia (Falso Negativo da CNN) ⚠️

> **ATENÇÃO:** Neste caso, a CNN **ERROU** o diagnóstico (classificou como NORMAL, mas é PNEUMONIA).  
> Avalie se a descrição do LLM consegue identificar achados suspeitos apesar do erro da CNN.

**1. Avaliação Geral (1-5):** ___

**2. Problemas Identificados (marque todos aplicáveis):**
- [ ] Anatomia incorreta ou imprecisa
- [ ] Terminologia médica inapropriada
- [ ] Interpretação radiológica incorreta
- [ ] Inconsistente com diagnóstico CNN
- [ ] Alucinações (informações não visíveis)
- [ ] Omissão de achados importantes
- [ ] Nenhum problema significativo

**3. Robustez do Modelo (específico para este caso):**
- [ ] ✅ Identificou achados suspeitos/anormais apesar da CNN indicar "NORMAL"
- [ ] ⚖️ Descrição neutra/objetiva, sem contradizer nem reforçar o erro
- [ ] ❌ Reforçou o erro da CNN (ex: "pulmões limpos", "sem alterações")

**4. Comentários (opcional):**  
_______________________________________________  
_______________________________________________

---

## COMPARAÇÃO FINAL

### Ranking Geral
Ordene os modelos do **melhor (1) ao pior (3)**:

**Geral (considerando todos os 4 casos):**
1. ____________________
2. ____________________
3. ____________________

---

### Modelo Recomendado

Qual modelo você recomendaria para uso em sistema de apoio à decisão clínica?

- [ ] LLaVA 7B
- [ ] LLaVA 13B
- [ ] LLaVA-Llama3
- [ ] Nenhum (não estão prontos para uso clínico)

**Justificativa (1-2 frases):**  
_______________________________________________  
_______________________________________________  
_______________________________________________

---

### Comentários Adicionais

Observações gerais, sugestões de melhoria, ou outros comentários:

_______________________________________________  
_______________________________________________  
_______________________________________________  
_______________________________________________

---

## Escala de Avaliação Geral (1-5)

- **5 - Excelente:** Descrição precisa, detalhada, clinicamente útil
- **4 - Boa:** Descrição correta e adequada
- **3 - Adequada:** Descrição genérica mas não incorreta
- **2 - Ruim:** Descrição imprecisa ou confusa
- **1 - Inadequada:** Descrição incorreta ou não útil

---

**Obrigado pela sua avaliação!** 🙏
