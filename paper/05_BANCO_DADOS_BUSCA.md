# 05 - Banco de Dados e Busca Semântica

## Sumário
- [Visão Geral](#visão-geral)
- [Arquitetura de Dados Híbrida](#arquitetura-de-dados-híbrida)
- [SQLite: Banco Relacional](#sqlite-banco-relacional)
- [ChromaDB: Banco Vetorial](#chromadb-banco-vetorial)
- [Deduplicação com SHA-256](#deduplicação-com-sha-256)
- [Busca Semântica](#busca-semântica)
- [Repositórios e CRUD](#repositórios-e-crud)
- [Performance e Otimizações](#performance-e-otimizações)

---

## Visão Geral

O PneumoFinder utiliza uma **arquitetura de dados híbrida** combinando:

1. **SQLite** - Banco relacional para dados estruturados (diagnósticos, metadados, BLOBs)
2. **ChromaDB** - Banco vetorial para busca semântica de descrições clínicas

Essa abordagem permite:
- ✅ Consultas relacionais rápidas (SQL)
- ✅ Busca semântica em linguagem natural
- ✅ Deduplicação automática de imagens (evita reprocessamento)
- ✅ Persistência de visualizações (heatmaps/overlays)

```
┌────────────────────────────────────────────────────────┐
│                    Camada API                          │
│               (src/api/app.py)                         │
└──────────────────┬─────────────────────────────────────┘
                   │
       ┌───────────▼────────────┐
       │  Camada Repositório    │
       │ (src/db/repositories)  │
       └────────┬───────┬───────┘
                │       │
    ┌───────────▼─┐   ┌─▼────────────────┐
    │   SQLite    │   │   ChromaDB       │
    │  Relacional │   │   Vetorial       │
    │             │   │                  │
    │ - Diagnoses │   │ - Embeddings     │
    │ - BLOBs     │   │ - HNSW Index     │
    │ - Metadata  │   │ - Cosine Sim.    │
    └─────────────┘   └──────────────────┘
```

---

## Arquitetura de Dados Híbrida

### Por que Dois Bancos de Dados?

| Aspecto | SQLite | ChromaDB |
|---------|--------|----------|
| **Tipo** | Relacional (tabelas + SQL) | Vetorial (embeddings + KNN) |
| **Consultas** | Exatas (WHERE, JOIN) | Aproximadas (similaridade) |
| **Índices** | B-tree (coluna única) | HNSW (vetores 384-dim) |
| **Caso de uso** | Buscar diagnóstico por ID | Buscar casos similares por texto |
| **Performance** | O(log n) | O(log n) com HNSW |
| **Tamanho por registro** | ~1KB (estruturado) | ~1.5KB (vetor + metadata) |

### Exemplo Prático

**SQL tradicional:**
```sql
SELECT * FROM diagnoses WHERE diagnosis = 'PNEUMONIA' AND confidence > 0.8;
```
→ Retorna diagnósticos **exatos** com filtros rígidos.

**Busca vetorial:**
```python
vector_store.search_similar("infiltrados nos campos pulmonares inferiores")
```
→ Retorna diagnósticos **semanticamente similares** mesmo sem palavras exatas.

---

## SQLite: Banco Relacional

### Escolha do SQLite

SQLite foi escolhido por:

1. **Simplicidade**: Arquivo único (`pneumofinder.db`), sem servidor externo
2. **Portabilidade**: Facilmente backupeável (simples cópia de arquivo)
3. **ACID**: Transações seguras com rollback
4. **Zero Configuração**: Embutido no Python (módulo `sqlite3`)
5. **Suficiente**: Para aplicações com < 100k requisições/dia

### Schema do Banco de Dados

```sql
-- Tabela principal: diagnósticos
CREATE TABLE diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT NOT NULL UNIQUE,           -- SHA-256 (deduplicação)
    diagnosis TEXT NOT NULL,                   -- "PNEUMONIA" ou "NORMAL"
    confidence REAL NOT NULL,                  -- 0.0 a 1.0
    model_version TEXT NOT NULL,               -- "resnet50_v1" (rastreabilidade)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,                              -- Identificador do usuário (opcional)
    metadata TEXT                              -- JSON com informações extras
);

-- Tabela de visualizações: heatmaps e overlays
CREATE TABLE visualizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id INTEGER NOT NULL,
    heatmap_blob BLOB,                         -- PNG binário do heatmap
    overlay_blob BLOB,                         -- PNG binário do overlay
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
);

-- Tabela de descrições clínicas: texto do LLM
CREATE TABLE clinical_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id INTEGER NOT NULL,
    description_text TEXT NOT NULL,            -- Descrição gerada pelo LLM
    llm_model TEXT NOT NULL,                   -- "llava:7b" (rastreabilidade)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
);

-- Índices para otimização
CREATE INDEX idx_diagnoses_created_at ON diagnoses(created_at DESC);
CREATE INDEX idx_diagnoses_hash ON diagnoses(image_hash);
```

**Código:** `src/db/database.py:27-73`

### Relacionamentos entre Tabelas

```
diagnoses (1) ─────< (N) visualizations
    │
    └─────────────< (N) clinical_descriptions
```

- **1:N com `visualizations`**: Um diagnóstico pode ter múltiplas versões de visualização (futuro suporte a múltiplos modelos)
- **1:N com `clinical_descriptions`**: Um diagnóstico pode ter múltiplas descrições de diferentes LLMs

### Gerenciamento de Conexões

```python
# src/db/database.py
from contextlib import contextmanager

@contextmanager
def get_connection():
    """Context manager para conexões seguras com commit/rollback automático."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # Permite acesso por nome: row['diagnosis']
    try:
        yield conn
        conn.commit()  # Auto-commit em caso de sucesso
    except Exception:
        conn.rollback()  # Auto-rollback em caso de erro
        raise
    finally:
        conn.close()
```

**Código:** `src/db/database.py:81-93`

**Uso:**
```python
with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM diagnoses WHERE id = ?", (42,))
    result = cursor.fetchone()
# Conexão fechada automaticamente, commit/rollback já executado
```

---

## ChromaDB: Banco Vetorial

### O que é um Banco Vetorial?

Bancos vetoriais armazenam **embeddings** (vetores numéricos de alta dimensão) que representam o **significado semântico** de textos. Diferente de bancos tradicionais que fazem matching exato, bancos vetoriais usam **distância vetorial** para encontrar conteúdos similares.

```
Texto: "Infiltrados nos campos pulmonares inferiores"
       ↓ Sentence-Transformers (all-MiniLM-L6-v2)
Embedding: [0.12, -0.45, 0.78, ..., 0.03]  (384 dimensões)
       ↓ Armazenado no ChromaDB
Índice HNSW: Estrutura de grafo para busca aproximada
```

### Por que ChromaDB?

1. **Open Source**: Sem vendor lock-in
2. **Python-native**: API idiomática e fácil integração
3. **Embeddings automáticos**: Suporta modelos do Sentence-Transformers
4. **HNSW index**: Busca rápida O(log n) mesmo com milhões de vetores
5. **Persistência**: Dados salvos em disco (não apenas in-memory)
6. **Dockerizável**: Fácil deployment em produção

### Configuração do ChromaDB

```python
# src/db/vector_store.py
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self):
        # Cliente persistente (salva em database/vectors/)
        self.client = chromadb.PersistentClient(
            path=config.vector_db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Cria coleção com distância de cosseno
        self.collection = self.client.get_or_create_collection(
            name="clinical_descriptions",
            metadata={"hnsw:space": "cosine"}  # Similaridade de cosseno
        )
```

**Código:** `src/db/vector_store.py:15-26`

### Distância de Cosseno

A similaridade de cosseno mede o **ângulo** entre dois vetores:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

Valores:
  1.0  → Vetores idênticos (mesmo significado)
  0.5  → Moderadamente similares
  0.0  → Ortogonais (significados independentes)
 -1.0  → Opostos (significados contrários)
```

**Exemplo prático:**
```python
# "Pneumonia bilateral severa" vs "Infiltrados bilaterais extensos"
similarity = 0.87  # Alta similaridade semântica

# "Pneumonia" vs "Fratura óssea"
similarity = 0.12  # Baixa similaridade (domínios diferentes)
```

### Modelo de Embeddings

ChromaDB usa **all-MiniLM-L6-v2** (Sentence-Transformers) por padrão:

| Característica | Valor |
|----------------|-------|
| **Dimensões** | 384 |
| **Parâmetros** | 22M |
| **Velocidade** | ~1ms por sentença (CPU) |
| **Qualidade** | 80% da BERT-base com 5x menor |
| **Treinamento** | 1B+ pares de sentenças (Wikipedia, StackExchange) |

---

## Deduplicação com SHA-256

### Motivação

Processar a mesma radiografia múltiplas vezes desperdiça:
- ⏱️ **Tempo**: 16 segundos por diagnóstico completo
- 💰 **Recursos**: CPU (CNN + Grad-CAM) e GPU (LLM)
- 💾 **Espaço**: BLOBs duplicados no banco (heatmaps/overlays)

### Algoritmo SHA-256

SHA-256 (Secure Hash Algorithm 256-bit) gera um **fingerprint único** para cada arquivo:

```python
import hashlib

def compute_image_hash(image_bytes: bytes) -> str:
    """
    Gera hash SHA-256 de uma imagem.
    Mesmo arquivo sempre resulta no mesmo hash (determinístico).
    """
    return hashlib.sha256(image_bytes).hexdigest()

# Exemplo:
# Arquivo 1: chest_xray.jpg (50KB)
# Hash: "a3f2b1c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2"
#
# Mesmo arquivo reenviado → Mesmo hash (deduplicação)
# Arquivo diferente → Hash completamente diferente
```

**Código:** `src/db/repositories.py:17-19`

### Fluxo de Deduplicação

```
1. Cliente envia imagem
      ↓
2. API calcula SHA-256
      ↓
3. Consulta banco: SELECT id FROM diagnoses WHERE image_hash = ?
      ├─→ Encontrado: retorna diagnosis_id existente (cache hit)
      │                └─→ Tempo: ~10ms (apenas I/O)
      │
      └─→ Não encontrado: processa imagem normalmente
                          └─→ Tempo: ~16s (CNN + Grad-CAM + LLM)
      ↓
4. Salva novo diagnóstico com image_hash UNIQUE
```

**Código:** `src/db/repositories.py:42-53`

### Código de Deduplicação

```python
# src/api/app.py
with open(temp_path, "rb") as f:
    image_hash = hashlib.sha256(f.read()).hexdigest()

# src/db/repositories.py
def save_diagnosis(image_hash, diagnosis, confidence, ...):
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Verifica se já existe
        cursor.execute(
            "SELECT id FROM diagnoses WHERE image_hash = ?",
            (image_hash,)
        )
        existing = cursor.fetchone()
        
        if existing:
            return existing["id"]  # Cache hit: retorna ID existente
        
        # Não existe: insere novo registro
        cursor.execute("""
            INSERT INTO diagnoses (image_hash, diagnosis, confidence, ...)
            VALUES (?, ?, ?, ...)
        """, (image_hash, diagnosis, confidence, ...))
        
        return cursor.lastrowid  # Retorna ID do novo registro
```

**Código:** `src/db/repositories.py:22-71`

### Taxa de Deduplicação (Estimativa)

Em ambiente de teste com dataset de 100 imagens:

| Métrica | Valor |
|---------|-------|
| **Total de requisições** | 500 |
| **Imagens únicas** | 100 |
| **Cache hits** | 400 (80%) |
| **Tempo economizado** | 400 × 16s = **1h 46min** |

---

## Busca Semântica

### Como Funciona

```
1. Usuário envia query: "infiltrados bilaterais nos campos inferiores"
      ↓
2. ChromaDB gera embedding da query (384 dimensões)
      ↓
3. Busca no índice HNSW os vetores mais próximos (top_k)
      ↓
4. Retorna IDs + distâncias dos diagnósticos similares
      ↓
5. Para cada ID, busca dados completos no SQLite
      ↓
6. Retorna lista ranqueada com:
   - diagnosis, confidence (do SQLite)
   - description (do ChromaDB)
   - similarity_score (convertido de distância para similaridade)
```

### Código de Busca Semântica

```python
# src/db/vector_store.py
class VectorStore:
    def search_similar(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Busca descrições clínicas similares à query.
        
        Args:
            query: Texto de busca em linguagem natural
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados com diagnosis_id, description, distance
        """
        results = self.collection.query(
            query_texts=[query],  # ChromaDB gera embedding automaticamente
            n_results=top_k
        )
        
        # Formata resultados
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "diagnosis_id": int(results["ids"][0][i]),
                "description": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return formatted_results
```

**Código:** `src/db/vector_store.py:43-75`

### Integração com Repositório

```python
# src/db/repositories.py
def search_similar_cases(query: str, top_k: int = 5) -> list[dict]:
    """Combina busca vetorial (ChromaDB) com dados estruturados (SQLite)."""
    
    # 1. Busca vetorial
    vector_store = get_vector_store()
    similar_descriptions = vector_store.search_similar(query, top_k=top_k)
    
    # 2. Enriquece com dados do SQLite
    results = []
    with get_connection() as conn:
        cursor = conn.cursor()
        for desc in similar_descriptions:
            cursor.execute(
                "SELECT * FROM diagnoses WHERE id = ?",
                (desc["diagnosis_id"],)
            )
            row = cursor.fetchone()
            if row:
                results.append({
                    **dict(row),  # diagnosis, confidence, created_at, etc.
                    "description": desc["description"],
                    "similarity_score": 1 - desc["distance"]  # Converte distância em similaridade
                })
    
    return results
```

**Código:** `src/db/repositories.py:182-215`

### Exemplo de Busca

**Query:** `"consolidações nos lobos inferiores"`

**Resultados:**

```json
{
  "query": "consolidações nos lobos inferiores",
  "top_k": 3,
  "results": [
    {
      "diagnosis_id": 42,
      "diagnosis": "PNEUMONIA",
      "confidence": 0.94,
      "similarity_score": 0.87,
      "description": "Radiografia revela consolidações bilaterais nos campos pulmonares inferiores...",
      "created_at": "2024-01-15T10:30:00"
    },
    {
      "diagnosis_id": 38,
      "diagnosis": "PNEUMONIA",
      "confidence": 0.89,
      "similarity_score": 0.82,
      "description": "Opacidades consolidativas predominantes nas bases pulmonares...",
      "created_at": "2024-01-14T08:15:22"
    },
    {
      "diagnosis_id": 50,
      "diagnosis": "PNEUMONIA",
      "confidence": 0.91,
      "similarity_score": 0.79,
      "description": "Infiltrados parenquimatosos bilaterais com distribuição basal...",
      "created_at": "2024-01-16T14:22:11"
    }
  ]
}
```

---

## Repositórios e CRUD

### Padrão Repository

O padrão Repository abstrai a persistência e fornece API limpa para a camada de negócio:

```
src/api/app.py (Flask routes)
       ↓
src/db/repositories.py (Funções CRUD)
       ↓
src/db/database.py (Conexões SQLite)
src/db/vector_store.py (ChromaDB)
```

### Funções Disponíveis

#### `save_diagnosis()`

```python
def save_diagnosis(
    image_hash: str,
    diagnosis: str,
    confidence: float,
    user_id: str | None = None,
    metadata: dict | None = None
) -> int:
    """
    Salva diagnóstico no SQLite com deduplicação automática.
    
    Returns:
        diagnosis_id (int): ID do registro (novo ou existente)
    """
```

**Código:** `src/db/repositories.py:22-71`

---

#### `save_visualization()`

```python
def save_visualization(
    diagnosis_id: int,
    heatmap_bytes: bytes,
    overlay_bytes: bytes
):
    """
    Salva visualizações Grad-CAM como BLOBs no SQLite.
    
    Args:
        diagnosis_id: Foreign key para tabela diagnoses
        heatmap_bytes: PNG binário do heatmap
        overlay_bytes: PNG binário do overlay
    """
```

**Código:** `src/db/repositories.py:74-91`

---

#### `save_clinical_description()`

```python
def save_clinical_description(
    diagnosis_id: int,
    description: str,
    diagnosis: str,
    confidence: float
):
    """
    Salva descrição clínica no SQLite E ChromaDB simultaneamente.
    
    - SQLite: Armazena texto completo
    - ChromaDB: Armazena embedding para busca semântica
    """
```

**Código:** `src/db/repositories.py:94-126`

---

#### `get_diagnosis()`

```python
def get_diagnosis(diagnosis_id: int) -> dict | None:
    """Recupera diagnóstico por ID."""
    # Returns: {"id": 42, "diagnosis": "PNEUMONIA", "confidence": 0.87, ...}
```

**Código:** `src/db/repositories.py:129-138`

---

#### `get_recent_diagnoses()`

```python
def get_recent_diagnoses(limit: int = 10) -> list[dict]:
    """Lista diagnósticos mais recentes (ORDER BY created_at DESC)."""
```

**Código:** `src/db/repositories.py:171-179`

---

#### `search_similar_cases()`

```python
def search_similar_cases(query: str, top_k: int = 5) -> list[dict]:
    """
    Busca semântica combinando ChromaDB + SQLite.
    
    Args:
        query: Texto em linguagem natural
        top_k: Número de resultados
        
    Returns:
        Lista de diagnósticos similares com similarity_score
    """
```

**Código:** `src/db/repositories.py:182-215`

---

## Performance e Otimizações

### Índices SQL

```sql
-- Índice para busca por hash (deduplicação)
CREATE INDEX idx_diagnoses_hash ON diagnoses(image_hash);

-- Índice para listagem recente
CREATE INDEX idx_diagnoses_created_at ON diagnoses(created_at DESC);
```

**Código:** `src/db/database.py:65-73`

**Impacto:**
- ✅ Deduplicação: ~10ms (com índice) vs ~500ms (sem índice, full scan)
- ✅ Listagem recente: ~5ms (com índice) vs ~100ms (sem índice)

### HNSW Index (ChromaDB)

**Hierarchical Navigable Small World** (HNSW) é um algoritmo de busca aproximada em grafos:

```
Complexidade:
- Inserção: O(log n)
- Busca: O(log n)
- Memória: O(n × dimensões)

Parâmetros ChromaDB:
- ef_construction: 200 (qualidade do grafo)
- M: 16 (conexões por nó)
```

**Comparação com busca linear:**

| Registros | Linear (ms) | HNSW (ms) | Speedup |
|-----------|-------------|-----------|---------|
| 1.000 | 50 | 2 | 25x |
| 10.000 | 500 | 5 | 100x |
| 100.000 | 5.000 | 10 | 500x |

### Tamanho de Armazenamento

**SQLite:**
```
Diagnóstico sem visualização: ~500 bytes
  - Metadados: ~300 bytes
  - JSON: ~200 bytes

Diagnóstico com visualização: ~200KB
  - Metadados: ~500 bytes
  - Heatmap PNG: ~100KB
  - Overlay PNG: ~100KB
```

**ChromaDB:**
```
Embedding + metadata: ~1.5KB por descrição
  - Vetor 384-dim (float32): 384 × 4 = 1,536 bytes
  - Metadata JSON: ~100 bytes
```

### Estratégias de Otimização

1. **Compressão de BLOBs**:
```python
import zlib

# Comprimir antes de salvar
compressed = zlib.compress(heatmap_bytes, level=9)

# Descomprimir ao recuperar
original = zlib.decompress(compressed)
```

2. **Paginação de Resultados**:
```python
def get_recent_diagnoses(limit: int = 10, offset: int = 0):
    cursor.execute(
        "SELECT * FROM diagnoses ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset)
    )
```

3. **Cache em Memória** (Redis futuro):
```python
# Pseudocódigo
if redis.exists(f"diagnosis:{diagnosis_id}"):
    return redis.get(f"diagnosis:{diagnosis_id}")
else:
    data = get_diagnosis(diagnosis_id)
    redis.setex(f"diagnosis:{diagnosis_id}", 3600, data)  # TTL 1h
    return data
```

---

## Backup e Recuperação

### Backup do SQLite

```bash
# Backup simples (cópia de arquivo)
cp database/pneumofinder.db backup-$(date +%Y%m%d).db

# Backup com dump SQL
sqlite3 database/pneumofinder.db .dump > backup.sql

# Restaurar de dump
sqlite3 database/pneumofinder.db < backup.sql
```

### Backup do ChromaDB

```bash
# ChromaDB persiste em diretório
tar -czf chroma-backup-$(date +%Y%m%d).tar.gz database/vectors/

# Restaurar
tar -xzf chroma-backup-20240115.tar.gz -C database/
```

### Backup em Docker

```bash
# Backup do volume Docker (SQLite)
docker run --rm \
  -v pneumofinder_api-database:/data \
  -v $(pwd):/backup \
  alpine tar -czf /backup/db-backup.tar.gz -C /data .

# Backup do volume ChromaDB
docker run --rm \
  -v pneumofinder_chroma-data:/data \
  -v $(pwd):/backup \
  alpine tar -czf /backup/chroma-backup.tar.gz -C /data .
```

---

## Monitoramento e Métricas

### Métricas Úteis

```python
# Total de diagnósticos no sistema
SELECT COUNT(*) FROM diagnoses;

# Taxa de deduplicação (últimos 7 dias)
SELECT 
  COUNT(DISTINCT image_hash) as unique_images,
  COUNT(*) as total_requests,
  ROUND((1.0 - COUNT(DISTINCT image_hash) * 1.0 / COUNT(*)) * 100, 2) as dedup_rate_pct
FROM diagnoses
WHERE created_at > datetime('now', '-7 days');

# Distribuição de diagnósticos
SELECT diagnosis, COUNT(*) as count, AVG(confidence) as avg_conf
FROM diagnoses
GROUP BY diagnosis;

# Tamanho do banco ChromaDB
vector_store.get_collection_count()  # Python
```

---

## Referências

- SQLite Documentation: https://www.sqlite.org/docs.html
- ChromaDB Documentation: https://docs.trychroma.com/
- HNSW Algorithm Paper: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
- Sentence-Transformers: https://www.sbert.net/

---

**Próximo documento:** `06_RESULTADOS_AVALIACAO.md` (Resultados experimentais, métricas e exemplos reais)
