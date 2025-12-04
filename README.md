# 🫁 PneumoFinder APIRestful

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)  ![Flask](https://img.shields.io/badge/Flask-API-lightgrey?logo=flask)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange?logo=tensorflow)  ![Twilio](https://img.shields.io/badge/Twilio-WhatsApp-green?logo=twilio)  ![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)  

---

O **PneumoFinder** é uma API RESTful desenvolvida em **Flask** que utiliza **Redes Neurais Convolucionais (CNNs)** e **Modelos de Linguagem Multimodais (LLMs)** para análise de radiografias de pulmão.  
A aplicação é capaz de:

- Verificar se a imagem enviada é de um pulmão.  
- Detectar sinais de **pneumonia** em radiografias.  
- Fornecer diagnósticos completos com nível de confiança.  
- Gerar **descrições clínicas explicativas** usando LLaVA (Large Language and Vision Assistant).  
- Produzir **visualizações Grad-CAM** (heatmaps e overlays) das regiões de atenção do modelo.  
- Integrar-se ao **WhatsApp** via Twilio, permitindo que o usuário envie a radiografia e receba o diagnóstico diretamente no aplicativo de mensagens.  

---

## 📌 Funcionalidades

### Endpoints em Inglês (novos)
- **`POST /diagnose`** → Diagnóstico simples de pneumonia com CNN  
- **`POST /diagnose/explained`** → Diagnóstico completo com Grad-CAM + descrição clínica do LLM  
- **`GET /health`** → Health check da API  

### Endpoints em Português (legados - compatibilidade)
- **`/verificar_pulmao`** → Verifica se a imagem enviada é de um pulmão  
- **`/diagnosticar_pneumonia`** → Detecta pneumonia em uma radiografia de pulmão  
- **`/diagnostico_completo`** → Verificação completa: identifica pulmão e analisa pneumonia  
- **`/diagnosticar_com_descricao`** → Diagnóstico multimodal com CNN + LLM + Grad-CAM  

### Integração WhatsApp
- **`/webhook`** → Endpoint conectado ao **Twilio** para diagnósticos via WhatsApp  

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.11** (gerenciado via [mise](https://mise.jdx.dev/))  
- **Flask** - Framework web para API REST  
- **Flask-CORS** - Habilita requisições entre origens  
- **TensorFlow / Keras** - Modelos CNN (ResNet50) para classificação de imagens  
- **Ollama + LLaVA 7B** - Modelo de linguagem multimodal para descrições clínicas  
- **OpenCV (cv2)** - Geração de visualizações Grad-CAM (heatmaps e overlays)  
- **Twilio API** - Integração com WhatsApp  
- **python-dotenv** - Gerenciamento de variáveis de ambiente  
- **Pillow** - Processamento de imagens  
- **requests** - Requisições HTTP para APIs externas  
- **uv** - Gerenciador de pacotes Python ultrarrápido  

---

## 📂 Estrutura do Projeto

```
PneumoFinder/
│── models/                  # Modelos treinados (.keras)
│   ├── pneumonia_model.keras
│   └── pulmao_model.keras
│
│── src/                     # Código-fonte refatorado (arquitetura funcional)
│   ├── api/
│   │   └── app.py           # Rotas Flask (endpoints REST)
│   ├── core/
│   │   ├── diagnosis.py     # Funções de inferência CNN
│   │   ├── visualization.py # Geração de Grad-CAM
│   │   └── clinical_description.py  # Integração com LLM
│   ├── utils/
│   │   ├── config.py        # Configuração centralizada
│   │   ├── file_utils.py    # Operações de arquivos
│   │   └── image_utils.py   # Processamento de imagens
│   └── bots/
│       └── whatsapp_bot.py  # Integração com WhatsApp
│
│── service/
│   └── chat_bot_service.py  # Ponto de entrada do bot WhatsApp
│
│── prompts/
│   └── medical_analysis.txt # Template de prompt para LLM
│
│── temp/                    # Pasta temporária para uploads e visualizações
│── imgs_pulmoes/            # Imagens recebidas via WhatsApp
│── imgs/                    # Imagens de exemplo/teste
│── explicacoes/             # Exemplos de visualizações Grad-CAM
│── relatorios/              # Relatórios gerados com visualizações
│
│── app.py                   # Ponto de entrada da API (importa src.api.app)
│── testar_modelo.py         # Script para testar modelos localmente
│── test_multimodal.py       # Teste do serviço multimodal
│── test_api_multimodal.py   # Teste do endpoint multimodal
│
│── pyproject.toml           # Dependências do projeto (gerenciado por uv)
│── .mise.toml               # Configuração do mise (Python 3.11)
│── requirements.txt         # Dependências (compatibilidade)
│── .env.example             # Exemplo de variáveis de ambiente
│── .gitignore               # Arquivos ignorados pelo git
│── REFACTORING_SUMMARY.md   # Documentação da refatoração
```

---

## ⚙️ Instalação e Configuração

### Pré-requisitos
- [mise](https://mise.jdx.dev/getting-started.html) - Gerenciador de versões de ferramentas
  ```bash
  # macOS/Linux
  curl https://mise.run | sh
  
  # Windows (PowerShell)
  irm https://mise.run | iex
  ```

- [Ollama](https://ollama.ai/) - Para rodar o modelo LLaVA localmente
  ```bash
  # macOS/Linux
  curl -fsSL https://ollama.com/install.sh | sh
  
  # Windows - baixe o instalador em https://ollama.com/download
  
  # Após instalar, baixe o modelo LLaVA:
  ollama pull llava:7b
  ```

### Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/pneumofinder.git
   cd pneumofinder
   ```

2. Configure o ambiente com mise (instala Python 3.11 e cria o virtualenv automaticamente):
   ```bash
   mise install
   ```

3. Instale as dependências com uv:
   ```bash
   mise run install
   # ou: uv sync
   ```

4. Configure as variáveis de ambiente no arquivo `.env`:
   ```env
   TWILIO_ACCOUNT_SID=seu_sid
   TWILIO_AUTH_TOKEN=seu_token
   ```

5. Inicie o servidor Ollama (em outro terminal):
   ```bash
   ollama serve
   # O servidor ficará disponível em http://localhost:11434
   ```

6. Execute a API:
   ```bash
   mise run run
   # ou: uv run python app.py
   ```

7. Para rodar o webhook do WhatsApp:
   ```bash
   mise run bot
   # ou: uv run python service/chat_bot_service.py
   ```

### Comandos Disponíveis

- `mise run install` - Instala dependências
- `mise run run` - Executa a API Flask
- `mise run bot` - Executa o bot do WhatsApp
- `mise run test-model` - Testa o modelo com imagens de exemplo
- `mise run lint` - Verifica código com ruff
- `mise run format` - Formata código com ruff

### Migração do ambiente antigo (opcional)

Se você estava usando `venv` tradicional:
```bash
# Remove o ambiente antigo
rm -rf venv/

# mise + uv cuidam do resto
mise install
mise run install
```

---

## ✅ Exemplos de Uso

### 1. Diagnóstico Tradicional (cURL)

```bash
curl -X POST http://localhost:5001/diagnostico_completo \
  -F "imagem=@radiografia_teste.jpg"
```

Resposta esperada:
```json
{
  "classe_pulmao": "PULMÃO",
  "confianca_pulmao": 0.98,
  "classe_pneumonia": "NORMAL",
  "confianca_pneumonia": 0.92
}
```

### 2. Diagnóstico Multimodal com LLM (cURL)

```bash
curl -X POST http://localhost:5001/diagnosticar_com_descricao \
  -F "imagem=@radiografia_teste.jpg"
```

Resposta esperada:
```json
{
  "class": "PNEUMONIA",
  "confidence": 0.8734,
  "description": "The model has identified pneumonia in this chest X-ray with 87% confidence. The areas of concern are visible in the lower right lung field, showing increased opacity consistent with consolidation. The heatmap highlights regions where the neural network detected patterns associated with bacterial pneumonia, particularly in the right lower lobe.",
  "heatmap_url": "/static/temp/radiografia_teste_heatmap.png",
  "overlay_url": "/static/temp/radiografia_teste_overlay.png"
}
```

### 3. Visualizando as Imagens Geradas

Após fazer o diagnóstico multimodal, você pode acessar as visualizações no navegador:
- **Heatmap:** `http://localhost:5001/static/temp/radiografia_teste_heatmap.png`
- **Overlay:** `http://localhost:5001/static/temp/radiografia_teste_overlay.png`

---

## 📲 Imagens no WhatsApp (Chatbot)

📸 **Exemplo de envio de radiografia e resposta do bot:**  

![Chatbot WhatsApp - Exemplo 1](imgs/chatbot_pnumofinder.jpg)  


---

## 📌 Observações

- O modelo espera imagens no formato e tamanho específico (224x224).
- As imagens são normalizadas antes de serem enviadas ao modelo.
- O projeto está em ambiente local para testes. Para produção, considere segurança, performance e escalabilidade.

---

## 🧠 Sobre os Modelos de IA

### CNN (Rede Neural Convolucional)
O modelo base é uma CNN baseada em **ResNet50** treinada com imagens reais de radiografias de pulmão com e sem pneumonia. 
- Formato: `.keras`
- Entrada: Imagens 224x224 pixels normalizadas
- Saída: Classificação binária (NORMAL/PNEUMONIA) com nível de confiança
- Acurácia atual: **~80%** (em processo de aprimoramento)
- Limiar de decisão: 0.5

### Grad-CAM (Visualização de Atenção)
Técnica de **explainability** que gera mapas de calor (heatmaps) mostrando quais regiões da radiografia influenciaram a decisão da CNN:
- **Heatmap:** Mapa de calor puro com gradiente de cores
- **Overlay:** Heatmap sobreposto à imagem original para contexto anatômico

### LLaVA 7B (Large Language and Vision Assistant)
Modelo multimodal de **7 bilhões de parâmetros** que combina visão computacional com linguagem natural:
- **Função:** Gera descrições clínicas explicativas em linguagem médica profissional
- **Arquitetura:** CLIP (visão) + Vicuna 7B (linguagem)
- **Execução:** Local via Ollama (sem envio de dados para APIs externas)
- **Entrada:** Imagem original + predição CNN + visualizações Grad-CAM
- **Saída:** Narrativa clínica com localização anatômica e interpretação dos achados

### Pipeline Multimodal
```
Radiografia → CNN (ResNet50) → Predição (87% PNEUMONIA)
                    ↓
                Grad-CAM → Heatmap + Overlay
                    ↓
            LLaVA 7B (via Ollama) → Descrição clínica explicativa
                    ↓
        JSON com diagnóstico + visualizações + explicação
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou enviar pull requests com melhorias, correções ou novas funcionalidades.

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---


## 👨‍💻 Autor

Feito por **[Cauã Farias]**  
[LinkedIn](https://www.linkedin.com/in/cau%C3%A3-farias-739013288/) • [GitHub](https://github.com/CauZy-Goes)
