# 📄 Documentação Técnica para TCC/Monografia

Esta pasta contém documentação técnica detalhada do PneumoFinder, focada na implementação com LLMs e arquitetura multimodal para uso em trabalhos acadêmicos (TCC, monografias, artigos).

## 📚 Estrutura dos Documentos

### 1. **01_INTRODUCAO_MOTIVACAO.md**
- Contexto e problema de pesquisa
- Limitações de CNNs como "caixas-pretas"
- Motivação para uso de LLMs multimodais
- Objetivos e contribuições do trabalho
- Diferencial: explicabilidade + busca semântica

### 2. **02_FUNDAMENTOS_TEORICOS.md**
- Redes Neurais Convolucionais para classificação médica
- Técnicas de explicabilidade (Grad-CAM)
- Modelos de Linguagem Multimodais (LLaVA)
- Arquitetura CLIP + Vicuna
- Embeddings vetoriais e busca semântica
- Bases teóricas de ChromaDB

### 3. **03_ARQUITETURA_SISTEMA.md**
- Visão geral da arquitetura multimodal
- Diagrama de componentes (CNN + Grad-CAM + LLM + Database)
- Fluxo de dados completo
- Decisões de design e trade-offs
- Stack tecnológico justificado
- Arquitetura de deployment (Docker)

### 4. **04_IMPLEMENTACAO_LLM.md** ⭐ **DOCUMENTO PRINCIPAL**
- Configuração do Ollama + LLaVA 7B
- Estrutura do prompt médico (prompt engineering)
- Integração da API multimodal
- Código-fonte comentado linha a linha
- Tratamento de erros e fallbacks
- Parâmetros de inferência (temperatura, tokens)
- Encoding de imagens e contexto visual

### 5. **05_BANCO_DADOS_BUSCA.md**
- Persistência de diagnósticos (SQLite)
- Deduplicação inteligente (SHA-256)
- Armazenamento de embeddings (ChromaDB)
- Busca semântica de casos similares
- Geração de embeddings com sentence-transformers
- Casos de uso: recuperação de diagnósticos históricos

### 6. **06_RESULTADOS_AVALIACAO.md**
- Exemplos reais de descrições geradas (PNEUMONIA vs NORMAL)
- Análise de latência (CNN only vs CNN+LLM)
- Métricas de performance da API
- Avaliação qualitativa das explicações
- Limitações identificadas
- Comparação com abordagens baseline

### 7. **07_CODIGO_FONTE.md**
- Trechos críticos do código-fonte
- Prompt template completo
- Exemplos de requisições/respostas da API
- Configurações de deployment
- Scripts de teste

## 🎯 Como Usar Esta Documentação

### Para Monografia/TCC:
1. Leia **01_INTRODUCAO_MOTIVACAO.md** para contextualizar o problema
2. Use **02_FUNDAMENTOS_TEORICOS.md** para o capítulo de fundamentação teórica
3. Baseie o capítulo de metodologia em **03_ARQUITETURA_SISTEMA.md** e **04_IMPLEMENTACAO_LLM.md**
4. Cite resultados de **06_RESULTADOS_AVALIACAO.md**
5. Inclua código relevante de **07_CODIGO_FONTE.md** em apêndices

### Para Artigo Científico:
- **Abstract:** Resumo de 01 + principais resultados de 06
- **Introduction:** Baseado em 01
- **Related Work:** Complementar 02 com survey da literatura
- **Methodology:** Sintetizar 03 + 04
- **Results:** Expandir 06 com gráficos e tabelas
- **Conclusion:** Sintetizar contribuições e trabalhos futuros

### Para Apresentação/Defesa:
- **Slides de contexto:** 01_INTRODUCAO_MOTIVACAO.md
- **Slides técnicos:** Diagramas de 03_ARQUITETURA_SISTEMA.md
- **Demo ao vivo:** Exemplos de 06_RESULTADOS_AVALIACAO.md
- **Perguntas técnicas:** Detalhes de 04_IMPLEMENTACAO_LLM.md

## 📊 Figuras e Diagramas

Os documentos incluem:
- ✅ Diagramas de arquitetura em ASCII art (fácil conversão para LaTeX/Draw.io)
- ✅ Fluxogramas de pipeline
- ✅ Exemplos de código formatados
- ✅ Tabelas de comparação
- ✅ JSON de requisições/respostas reais

## 🔗 Referências Externas

Documentação complementar no repositório:
- `/README.md` - Visão geral do projeto
- `/docs/ARCHITECTURE.md` - Arquitetura técnica detalhada
- `/docs/DATABASE_INTEGRATION.md` - Detalhes do banco de dados
- `/DOCKER.md` - Instruções de deployment
- `/AGENTS.md` - Guidelines de desenvolvimento

## 📝 Formato e Estilo

- **Idioma:** Português brasileiro (pt-BR)
- **Público:** Graduação em Computação/Engenharia
- **Nível técnico:** Intermediário a avançado
- **Formato:** Markdown para fácil conversão (LaTeX, Word, PDF)
- **Código:** Sintaxe destacada com exemplos comentados

## 🚀 Reprodutibilidade

Todos os códigos e configurações documentados são:
- ✅ Reproduzíveis via Docker (`docker-compose up`)
- ✅ Testados e validados
- ✅ Versionados no Git
- ✅ Open-source (licença MIT)

## 📧 Contato e Contribuições

Para dúvidas sobre o conteúdo técnico:
- Abra uma issue no repositório
- Consulte os autores listados em `pyproject.toml`

---

**Última atualização:** Dezembro 2024  
**Versão do sistema:** 2.0.0  
**Status:** Documentação completa e validada
