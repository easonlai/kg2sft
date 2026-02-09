# kg2sft: Knowledge Graph to Supervised Fine-Tuning Data Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kg2sft.streamlit.app/)

> **Transform your Knowledge Graphs into high-quality training data for fine-tuning Small Language Models (SLMs) using LLM synthesis via Azure OpenAI.**

## 🎯 Overview

kg2sft automatically converts Knowledge Graphs (GraphML format) into supervised fine-tuning datasets by extracting multi-hop relationship paths and using Large Language Models to synthesize natural question-answer pairs. The tool is production-ready with robust retry logic, cost tracking, quality validation, and progress checkpointing.

**Core Philosophy:** Quality over quantity - every training example is validated to ensure it improves, not degrades, model performance.

## 📑 Table of Contents

- [Key Features](#key-features)
- [Quick Start](#-quick-start)
- [Web UI (Streamlit)](#-web-ui-streamlit)
- [Detailed Usage](#-detailed-usage)
- [Understanding Output: Why Fewer Examples](#%EF%B8%8F-understanding-output-why-you-may-get-fewer-examples-than-requested)
- [Sample Knowledge Graphs](#-sample-knowledge-graphs)
- [Input Format](#-input-format-graphml)
- [Output Format](#-output-format)
- [Architecture](#%EF%B8%8F-architecture)
- [Configuration](#%EF%B8%8F-configuration)
- [Cost Estimation](#-cost-estimation)
- [Quality Validation](#-quality-validation)
- [Troubleshooting](#-troubleshooting)
- [Examples](#-examples)
- [Advanced Usage](#%EF%B8%8F-advanced-usage)

### Key Features

- 🔄 **Multi-hop Path Extraction**: Intelligent sampling strategies (frequency-weighted, random)
- 🤖 **LLM-Powered Synthesis**: Generates natural Q&A pairs using Azure OpenAI GPT-4.1-mini
- ✅ **Quality Validation**: Multi-criteria scoring system (0-1 scale) with configurable thresholds
- 💰 **Cost Tracking**: Real-time token usage and cost calculation
- 🔁 **Retry Logic**: Exponential backoff for rate limits, automatic error recovery
- 📊 **Progress Tracking**: tqdm integration with periodic checkpointing
- 🎯 **Domain-Specific**: Built-in templates for beauty/skincare and generic domains
- 🔍 **Path Deduplication**: Jaccard similarity-based filtering for diverse training data
- 📈 **Comprehensive Reporting**: Detailed cost, quality, and graph statistics
- 🖥️ **Web UI (Optional)**: Streamlit-based UI with interactive graph visualization
- 🌐 **Cloud Hosted**: Try it instantly at [kg2sft.streamlit.app](https://kg2sft.streamlit.app/) - no installation required

## 🚀 Quick Start

> 💡 **Quick Option:** Try the [hosted web UI](https://kg2sft.streamlit.app/) instantly without any installation!

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access with GPT-4.1-mini deployment
- GraphML knowledge graph file

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kg2sft.git
cd kg2sft

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install networkx python-dotenv openai tqdm

# Or install from requirements.txt
pip install -r requirements.txt

# Optional: Install UI dependencies for Streamlit web interface
pip install streamlit pyvis
```

### Configuration

Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_MODEL=gpt-4.1-mini
```

### Basic Usage

```bash
# View help and all available options
python kg2sft.py --help

# Run with technology knowledge graph sample (generic domain)
python kg2sft.py --graph technology_knowledge.graphml --count 50

# Run with beauty products graph sample (beauty domain)
python kg2sft.py --graph beauty_products.graphml --count 50 --domain beauty

# Generate 100 examples from your own graph
python kg2sft.py --graph my_graph.graphml --count 100
```

### What You'll Get

After successful execution, kg2sft creates:

- **`output_training.jsonl`** - OpenAI-compatible fine-tuning format (ready to use)
- **`output_training.json`** - Human-readable format with quality scores (for review)
- **Console report** - Comprehensive statistics including:
  - Generation metrics (requested, generated, rejected, acceptance rate)
  - Cost breakdown (tokens, USD)
  - Quality scores (average, min, max)
  - Graph statistics (nodes, edges, paths)

**Expected results:** When requesting 50 examples, you typically get 20-40 high-quality pairs (40-60% acceptance rate) depending on your graph structure and quality threshold.

---

## 🖥️ Web UI (Streamlit)

For users who prefer a graphical interface, kg2sft provides an optional **Streamlit-based web UI** (`kg2sftui.py`) with interactive graph visualization.

### 🌐 Try It Online (No Installation Required)

**Live Demo:** [https://kg2sft.streamlit.app/](https://kg2sft.streamlit.app/)

You can use kg2sft directly in your browser without any local installation. Simply visit the link above, configure your Azure OpenAI credentials, upload your GraphML file, and start generating training data.

### UI Features

- 📁 **Drag-and-drop file upload** for GraphML files
- 🔗 **Interactive graph network visualization** using PyVis
  - Hover over nodes/edges for details
  - Pan, zoom, and navigate the graph
  - Color-coded nodes with size based on connection degree
- ⚙️ **All settings configurable in sidebar**
  - Azure OpenAI credentials (API key, endpoint, deployment)
  - Generation parameters (count, domain, temperature)
  - Advanced settings (quality threshold, dedup threshold, sampling strategy)
- 📊 **Real-time results display**
  - Generation metrics and quality scores
  - Cost breakdown
  - Preview of generated Q&A pairs
- 📥 **One-click download** of JSONL and JSON output files

### Running the Web UI

```bash
# Install UI dependencies (if not already installed)
pip install streamlit pyvis

# Launch the web interface
streamlit run kg2sftui.py
```

This will open a browser window (typically at `http://localhost:8501`) with the kg2sft web interface.

### UI Screenshots & Workflow

1. **Configure Azure OpenAI** in the sidebar (expand "🔑 Azure OpenAI Settings")
2. **Upload a GraphML file** or use the sample file buttons
3. **View the interactive graph** visualization
4. **Adjust generation settings** as needed
5. **Click "🎯 Generate"** to start the process
6. **Review results** and download the training data

### When to Use UI vs CLI

| Use Case | Recommended |
|----------|-------------|
| Quick try without installation | 🌐 **[Cloud UI](https://kg2sft.streamlit.app/)** |
| Quick exploration of a new knowledge graph | 🖥️ **Web UI** |
| Visual inspection of graph structure | 🖥️ **Web UI** |
| One-time generation with manual review | 🖥️ **Web UI** |
| Automated pipelines or scripts | ⌨️ **CLI (`kg2sft.py`)** |
| Batch processing multiple graphs | ⌨️ **CLI (`kg2sft.py`)** |
| CI/CD integration | ⌨️ **CLI (`kg2sft.py`)** |

---

## 📖 Detailed Usage

### Command-Line Arguments

```bash
python kg2sft.py [OPTIONS]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--graph` | str | - | ✅ | Path to input GraphML file |
| `--output` | str | `output_training` | ❌ | Output filename prefix |
| `--count` | int | `10` | ❌ | Number of training examples to generate |
| `--domain` | str | `generic` | ❌ | Domain template (`generic`, `beauty`) |
| `--temperature` | float | `0.7` | ❌ | LLM sampling temperature (0.0-1.0) |
| `--quality-threshold` | float | `0.7` | ❌ | Minimum quality score to accept (0.0-1.0) |
| `--max-depth` | int | `999` | ❌ | Maximum path length (999 = explore naturally until no unvisited neighbors) |
| `--dedup-threshold` | float | `0.95` | ❌ | Path similarity threshold for deduplication |
| `--sampling` | str | `frequency_weighted` | ❌ | Sampling strategy (`frequency_weighted`, `random`) |

### Examples

**Technology knowledge graph (generic domain):**
```bash
python kg2sft.py --graph technology_knowledge.graphml --count 50
```

**Beauty products graph (beauty domain):**
```bash
python kg2sft.py --graph beauty_products.graphml --count 50 --domain beauty
```

**Production use with custom settings:**
```bash
python kg2sft.py \
  --graph knowledge_graph.graphml \
  --output training_data \
  --count 1000 \
  --domain beauty \
  --temperature 0.8 \
  --quality-threshold 0.7 \
  --dedup-threshold 0.90
```

**Quick test with 5 examples:**
```bash
python kg2sft.py --graph technology_knowledge.graphml --count 5
```

---

### ⚠️ Understanding Output: Why You May Get Fewer Examples Than Requested

**TL;DR:** `--count 50` is a **target**, not a guarantee. The final count depends on two quality filters.

#### The Two-Stage Filtering Process

When you request 50 examples, kg2sft applies **two filtering mechanisms** that prioritize quality over quantity:

```
Request: --count 50
    ↓
Stage 1: Path Extraction with Deduplication
    Extract paths from graph
    Filter out duplicates (Jaccard similarity ≥ 0.95)
    → May reduce to ~35-40 unique paths
    ↓
Stage 2: Quality Validation
    Generate QA pairs from paths
    Filter out low-quality examples (score < 0.7)
    → Final output: ~19-30 high-quality examples
```

#### Stage 1: Path Deduplication Filter

**Purpose:** Ensure training data diversity by removing similar paths

**Configuration:** `--dedup-threshold` (default: `0.95`)

**How it works:**
- Calculates Jaccard similarity between paths
- If similarity ≥ 95%, the path is **skipped as a duplicate**
- Example: `Python → Django → Web Dev` vs `Python → Flask → Web Dev` might be 80% similar (kept)
- Example: `Python → Django → Web Dev` vs `Python → Django → API Dev` might be 96% similar (duplicate, skipped)

**Impact:** Smaller or highly connected graphs generate more similar paths, reducing unique paths available.

#### Stage 2: Quality Validation Filter

**Purpose:** Reject poor-quality QA pairs that would hurt model performance

**Philosophy:** **Quality over quantity** - Better 20 excellent training examples than 50 mediocre ones that could degrade model performance.

**Configuration:** `--quality-threshold` (default: `0.7`)

##### Hard Requirements (Immediate Rejection with score 0.0)

Before scoring, these requirements must pass:
1. **Non-empty data:** Question and answer fields must exist and have content
2. **Minimum question length:** Question must be ≥10 characters (prevents incomplete questions)
3. **Generic answer detection:** Rejects useless answers: "yes", "no", "i don't know", "not sure", "maybe"

##### Scoring Criteria (0.0-1.0 scale)

If hard requirements pass, calculate weighted score:

**1. Length Appropriateness (40% weight):**
- Perfect range (20-500 words): **+0.4**
- Too short (<20 words, but ≥10 words): **+0.4 × (actual/20)** (proportional partial credit)
- Too long (>500 words): **+0.35** (slight penalty, but long detailed answers are valuable for training!)

**2. Question Format (30% weight):**
- Has "?" character: **+0.3** (full credit)
- Starts with question word (what/how/why/etc.) but no "?": **+0.2** (partial credit)
- Neither: **+0.0**

**3. Answer Substance (30% weight):**
- ≥50 chars + at least 1 sentence (., !, ?): **+0.3** (substantial with structure)
- 30-49 chars: **+0.2** (moderate substance)
- 20-29 chars: **+0.1** (minimal substance)
- <20 chars: **+0.0** (too brief)

##### Final Decision
```python
if quality_score >= 0.7:  # quality_threshold
    ✅ Accept example (suitable for training)
else:
    ❌ Reject example (would hurt model quality)
```

**Key insight:** The 0.7 threshold balances quality with reasonable acceptance rates (typically 40-60%).

**Examples:**
```python
# ✅ Perfect (score: 1.0 - ACCEPTED)
{"question": "What frameworks does Python support for web development?", 
 "answer": "Python supports Django for full-stack applications and FastAPI... (40 words, 80 chars, 2 sentences)"}
# Breakdown: 0.4 (perfect length) + 0.3 (has ?) + 0.3 (50+ chars + sentences) = 1.0

# ✅ Good (score: 0.9 - ACCEPTED)
{"question": "How does Python support machine learning", 
 "answer": "Python supports machine learning through TensorFlow... (35 words, 70 chars)"}
# Breakdown: 0.4 (good length) + 0.2 (question word, no ?) + 0.3 (substantial) = 0.9

# ❌ Rejected: Below threshold (score: 0.65 < 0.7)
{"question": "What is Python", "answer": "Python is a language for coding (6 words, 35 chars)"}
# Breakdown: 0.4 × (6/20) = 0.12 (too short) + 0.2 (no ?) + 0.2 (30-49 chars) = 0.52

# ❌ Rejected: Generic answer (score: 0.0 - HARD REJECTION)
{"question": "Is Python good?", "answer": "Yes"}
# Generic answer detected → Immediate rejection, no scoring

# ✅ Long detailed answer (score: 0.95 - ACCEPTED)
{"question": "Explain Python's role in machine learning?",
 "answer": "Python dominates machine learning due to... (150 words, comprehensive)"}
# Breakdown: 0.35 (long but valuable!) + 0.3 (has ?) + 0.3 (excellent substance) = 0.95
# Note: Long answers valued - only slight penalty from 0.4 to 0.35
```

**Quality Philosophy in Action:**
- Generic "Yes/No" answers: Hard rejected (prevent garbage data)
- Short incomplete answers: Low scores (prevent poor training examples)
- Long detailed answers: High scores (even >500 words - detail is valuable!)
- Result: Small, high-quality dataset > large, noisy dataset

#### How to Get More Examples

**Option 1:** Lower quality threshold (trade some quality for quantity)
```bash
python kg2sft.py --graph my_graph.graphml --count 50 --quality-threshold 0.6
```

**Option 2:** Lower deduplication threshold (allow more similar paths)
```bash
python kg2sft.py --graph my_graph.graphml --count 50 --dedup-threshold 0.85
```

**Option 3:** Request more to compensate for filtering
```bash
python kg2sft.py --graph my_graph.graphml --count 100  # Expect ~40-60 after filtering
```

**Option 4:** Use a larger, more diverse knowledge graph
- More nodes → more unique path combinations
- More edge types → more diverse relationships
- Better connectivity → richer multi-hop paths

#### Monitoring Acceptance Rate

Check the console output during generation:
```
✅ Extracted 42 paths (skipped 8 duplicates)
🤖 Generating training examples...
✅ Saved 19 examples to output_training.jsonl
   Generated: 19, Rejected: 23, Acceptance Rate: 45.2%
```

**Healthy acceptance rates:**
- 40-60%: Excellent (good graph quality, appropriate thresholds)
- 25-40%: Good (quality-focused filtering working as intended)
- <25%: May need threshold adjustment or graph improvements

**Remember:** Low acceptance rates aren't necessarily bad - they mean the quality filter is working! A dataset of 20 excellent examples trains better models than 50 mediocre ones.

## 📊 Sample Knowledge Graphs

The repository includes two comprehensive sample graphs to showcase different use cases:

### 1. **technology_knowledge.graphml** (Generic Domain)
A sophisticated technology knowledge graph with:
- **40 nodes**: 5 programming languages, 7 frameworks, 7 use cases, 8 features, 4 organizations
- **Multi-layered relationships**: Language→Framework, Language→Feature, Language→Use Case, Framework→Feature, Organization→Technology
- **Example paths**: 
  - `Python → FastAPI → API Development`
  - `TypeScript → EXTENDS → JavaScript → React`
  - `Rust → HAS_FEATURE → Memory Safety`
- **Use case**: General-purpose knowledge representation, technology documentation, educational content

### 2. **beauty_products.graphml** (Beauty Domain)
A comprehensive beauty products knowledge graph with:
- **48 nodes**: 5 brands, 20 products, 15 ingredients, 13 benefits
- **Complex relationships**: Brand→Product, Product→Ingredient, Ingredient→Benefit, Ingredient↔Ingredient (synergies)
- **Example paths**:
  - `SK-II → Facial Treatment Essence → Pitera → Radiance Boost`
  - `La Mer → Crème de la Mer → Miracle Broth → Anti-Aging`
  - `Vitamin C → SYNERGIZES_WITH → Ferulic Acid → Antioxidant Protection`
- **Use case**: E-commerce product Q&A, beauty chatbots, skincare recommendation systems

Both samples demonstrate sophisticated multi-hop paths suitable for generating diverse, high-quality training data.

## 📥 Input Format: GraphML

kg2sft accepts knowledge graphs in GraphML format (XML-based). Here's the expected structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlformat/graphml.xsd">
  <graph id="G" edgedefault="directed">
    
    <!-- Nodes with attributes -->
    <node id="product1">
      <data key="name">Product Name</data>
      <data key="type">Product</data>
    </node>
    
    <node id="ingredient1">
      <data key="name">Ingredient Name</data>
      <data key="type">Ingredient</data>
    </node>
    
    <!-- Edges with relationships -->
    <edge source="product1" target="ingredient1">
      <data key="relationship">CONTAINS</data>
    </edge>
    
  </graph>
</graphml>
```

### Required Node Attributes
- `name` or `label`: Human-readable label (used in prompts)
- Additional attributes optional

### Required Edge Attributes
- `relationship` or `rel`: Relationship type (e.g., CONTAINS, PROVIDES)

## 📤 Output Format

kg2sft generates two files:

### 1. JSONL Format (OpenAI Fine-Tuning)
`output_training.jsonl` - Ready for OpenAI/Azure OpenAI fine-tuning:

```jsonl
{"messages": [{"role": "user", "content": "What does this product contain?"}, {"role": "assistant", "content": "This product contains..."}]}
{"messages": [{"role": "user", "content": "What are the benefits?"}, {"role": "assistant", "content": "The benefits include..."}]}
```

### 2. JSON Format (Human Review)
`output_training.json` - Pretty-printed with quality scores:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What does this product contain?"},
      {"role": "assistant", "content": "This product contains..."}
    ],
    "quality_score": 0.95
  }
]
```

### Checkpoint File
`checkpoint_training.jsonl` - Auto-saved during generation (every 10% progress), automatically deleted on successful completion.

---

## 🏗️ Architecture

Understanding the internal pipeline helps optimize your training data generation.

### Pipeline Workflow

```
┌─────────────────┐
│  GraphML File   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load Graph     │  GraphMLLoader → NetworkX DiGraph
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract Paths  │  PathExtractor: Multi-hop random walks
│                 │  - Frequency-weighted or random sampling
│                 │  - Cycle detection
│                 │  - Deduplication (Jaccard similarity)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │  PromptTemplate: Domain-specific prompts
│  Prompts        │  - Beauty domain
│                 │  - Generic domain
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Call LLM       │  AzureOpenAIClient: GPT-4.1-mini
│  (Azure OpenAI) │  - Retry logic (exponential backoff)
│                 │  - Token tracking
│                 │  - JSON response cleaning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validate       │  Validator: Multi-criteria scoring
│  Quality        │  - Length appropriateness (40%)
│                 │  - Question format (30%)
│                 │  - Answer substance (30%)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Dataset   │  TrainingDataset: JSONL + JSON
│  + Checkpoint   │  - Auto-checkpoint every 10%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │  Cost, quality, graph statistics
│  Report         │
└─────────────────┘
```

### Core Components

#### 1. **Graph Representation** (`Graph`, `GraphMLLoader`)
- Unified NetworkX DiGraph wrapper
- Dual storage: NetworkX graph + attribute dicts
- Efficient neighbor queries with BFS

#### 2. **Path Extraction** (`PathExtractor`)
- **Sampling Strategies (with replacement):**
  - `frequency_weighted`: Favors high-degree nodes (hubs), weighted by node degree
  - `random`: Uniform distribution across all nodes
  - Both allow sampling same node multiple times for diverse random walks
- **Random Walk**: Recursive depth-first traversal
- **Cycle Detection**: Tracks visited nodes (prevents infinite loops)
- **Max Depth**: Default 999 (effectively unlimited) - paths explore naturally until no unvisited neighbors
- **Deduplication**: Jaccard similarity ≥ threshold → skip

#### 3. **Prompt Generation** (`PromptTemplate`)
- Domain-specific templates
- JSON output format specification
- Grounding instructions (path-based answers only)

#### 4. **LLM Integration** (`AzureOpenAIClient`)
- **Retry Logic:**
  - RateLimitError: Exponential backoff (1s, 2s, 4s, 8s...)
  - OpenAIError: Fixed delay retry
  - Other errors: No retry
- **Cost Tracking:** Input/output tokens, USD calculation
- **Response Cleaning:** Strips markdown code blocks

#### 5. **Quality Validation** (`Validator`)
- **Hard Requirements:**
  - Non-empty question/answer
  - Question ≥10 characters
  - Rejects generic answers ("yes", "no", "i don't know")
- **Scoring Criteria (0.0-1.0):**
  - Length (40%): Word count in range → full credit; too short → proportional; too long → 0.35 (slight penalty)
  - Format (30%): Question mark → 0.3; question word only → 0.2
  - Substance (30%): 50+ chars + sentences → 0.3; 30+ → 0.2; 20+ → 0.1
- **Smart Partial Credit:** Proportional scoring based on quality, not binary pass/fail

#### 6. **Dataset Management** (`TrainingDataset`)
- OpenAI fine-tuning format
- Quality score tracking (internal)
- Statistics generation

---

## ⚙️ Configuration

Customize kg2sft behavior through configuration classes or CLI arguments.

### LLM Configuration

```python
LLMConfig(
    provider="azure_openai",
    model_name="gpt-4.1-mini",
    api_version="2024-12-01-preview",  # Current API version
    max_retries=3,                     # Retry attempts
    retry_delay=1.0,                   # Initial delay (seconds)
    input_price_per_1k=0.0004,         # USD per 1K input tokens ($0.40/1M)
    output_price_per_1k=0.0016         # USD per 1K output tokens ($1.60/1M)
)
```

### Generation Configuration

```python
GenerationConfig(
    count=100,                         # Target examples
    temperature=0.7,                   # LLM creativity (0-1)
    top_p=0.95,                        # Nucleus sampling
    max_tokens=500                     # Max response length
)
```

### Extraction Configuration

```python
ExtractionConfig(
    max_hop_depth=999,                 # Unlimited (cycle detection prevents infinite loops)
    sampling_strategy="frequency_weighted",  # or "random"
    dedup_threshold=0.95               # Similarity cutoff (0-1)
)
```

### Validation Configuration

```python
ValidationConfig(
    quality_threshold=0.7,             # Minimum score (0-1) - balanced quality/quantity
    min_length=20,                     # Min answer words
    max_length=500                     # Max answer words
)
```

**Note:** The default `quality_threshold=0.7` provides a good balance between quality and acceptance rate (typically 40-60%). Increase for stricter quality (e.g., 0.85), decrease for higher quantity (e.g., 0.6).

---

## 💰 Cost Estimation

Plan your budget by understanding token usage and associated costs.

### GPT-4.1-mini Pricing (as of 2026)
- Input: $0.40 per 1M tokens ($0.0004 per 1K tokens)
- Output: $1.60 per 1M tokens ($0.0016 per 1K tokens)

### Typical Costs (based on testing)
- **10 examples:** ~$0.02-0.05
- **100 examples:** ~$0.25-0.50
- **1,000 examples:** ~$2.50-5.00

### Cost Factors
- Path complexity (longer paths → longer prompts)
- Temperature (higher → more tokens)
- Max tokens setting
- Rejection rate (retries cost tokens)

### Cost Report Example
```
💰 Cost Report:
   Input Tokens: 12,543
   Output Tokens: 8,721
   Input Cost: $0.0050
   Output Cost: $0.0140
   Total Cost: $0.0190
   Cost per Example: $0.000190
   API Calls: 100
```

---

## 📈 Quality Validation

**Core Philosophy:** Quality over quantity - ensuring every training example improves model performance.

### Hard Requirements (Immediate Rejection)

| Requirement | Reason |
|-------------|--------|
| Non-empty question & answer | Basic data integrity |
| Question ≥10 characters | Prevents incomplete questions |
| Not generic answer | Rejects "yes", "no", "i don't know", "not sure", "maybe" |

**Note:** Answer length is evaluated in the scoring phase, not as a hard requirement.

### Quality Scoring (0.0-1.0 scale)

| Criterion | Weight | Scoring Logic |
|-----------|--------|---------------|
| **Length** | 40% | **Perfect (min ≤ words ≤ max):** 0.4<br>**Too short (10-20 words):** 0.4 × (words/min)<br>**Too long:** 0.35 (long answers still valuable!) |
| **Format** | 30% | **Has "?":** 0.3<br>**Starts with question word:** 0.2<br>**Neither:** 0.0 |
| **Substance** | 30% | **50+ chars + sentences:** 0.3<br>**30-49 chars:** 0.2<br>**20-29 chars:** 0.1<br>**<20 chars:** 0.0 |

### Smart Partial Credit
- **Too short answers:** Proportional credit (e.g., 15 words when min is 20 → 0.4 × 0.75 = 0.3)
- **Too long answers:** Still get high credit (0.35 vs 0.4) - detailed answers are good for training!
- **Question format:** Partial credit for question words even without "?"
- **Substance:** Graduated scoring based on both length and sentence structure

### Acceptance Logic
```
if quality_score >= quality_threshold:
    accept_example()
else:
    reject_example()
```

### Example Scores

```python
# ✅ Perfect example (Score: 1.0)
{"question": "What benefits does X provide?", 
 "answer": "X provides multiple benefits including... (40 words, well-structured)"}
# Breakdown: 0.4 (perfect length) + 0.3 (has ?) + 0.3 (50+ chars + sentences) = 1.0

# ✅ Good example with question word (Score: 0.9)
{"question": "How does Python support web development", 
 "answer": "Python supports web development through frameworks... (35 words)"}
# Breakdown: 0.4 (good length) + 0.2 (question word, no ?) + 0.3 (substantial) = 0.9

# ✅ Good quality (Score: 0.9)
{"question": "What is Django?", 
 "answer": "Django is a Python web framework for building web applications with rapid development... (15 words, 85 chars)"}
# Breakdown: 0.4 (perfect length: 15 words) + 0.3 (has ?) + 0.3 (50+ chars + structure) = 1.0

# ❌ Rejected: Too short (Score: ~0.45)
{"question": "What is Python?", "answer": "A programming language."}
# Breakdown: 0.15 (too short) + 0.3 (has ?) + 0.0 (too few chars) = 0.45

# ❌ Rejected: Generic answer (Score: 0.0)
{"question": "Is Python good?", "answer": "Yes"}
# Hard rejection for generic answer

# ✅ Long detailed answer (Score: 0.95)
{"question": "Explain Python's role in machine learning?",
 "answer": "Python dominates machine learning due to... (150 words, comprehensive)"}
# Breakdown: 0.35 (too long but valuable) + 0.3 (has ?) + 0.3 (excellent substance) = 0.95
# Note: Long answers only get slight penalty, still high quality!
```

### Key Takeaways

✅ **Quality over quantity** - 20 excellent examples > 50 mediocre ones  
✅ **Smart partial credit** - Near-misses still contribute, not binary pass/fail  
✅ **Long answers valued** - Detailed responses (>500 words) score 0.95 vs 1.0  
✅ **Hard rejections prevent garbage** - Generic answers immediately filtered  
✅ **Typical acceptance: 40-60%** - Low rates mean quality filter is working  

---

## 🔧 Troubleshooting

Common issues and their solutions to keep your pipeline running smoothly.

### Common Issues

#### 1. **Missing API Key**
```
❌ AZURE_OPENAI_API_KEY not set
```
**Solution:** Create `.env` file with required credentials

#### 2. **Rate Limit Errors**
```
⚠️ Rate limit hit, retrying in 1s (attempt 1/3)
```
**Solution:** Automatically handled with exponential backoff. If persistent:
- Reduce generation rate
- Increase `retry_delay` in LLMConfig
- Check Azure OpenAI quota

#### 3. **Low Acceptance Rate**
```
Acceptance Rate: 25.0%
```
**Solution:** 
- Lower `quality_threshold` (e.g., 0.6 instead of 0.7)
- Adjust `min_length`/`max_length` in ValidationConfig
- Check graph path quality

#### 4. **No Paths Extracted**
```
⚠️ No nodes in graph!
```
**Solution:**
- Verify GraphML file format
- Ensure graph has edges (not just nodes)
- Check file path is correct

#### 5. **JSON Parse Errors**
```
⚠️ JSON decode error: Expecting value...
```
**Solution:** Automatically handled, but if frequent:
- Lower `temperature` (more structured output)
- Check prompt template clarity
- Review LLM response in logs

### Debug Tips

1. **Enable DEBUG logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check checkpoint file:**
```bash
cat checkpoint_training.jsonl
```

3. **Validate GraphML:**
```python
import networkx as nx
graph = nx.read_graphml("your_graph.graphml")
print(f"Nodes: {len(graph.nodes())}, Edges: {len(graph.edges())}")
```

---

## 📚 Examples

Real-world usage patterns and configurations for different scenarios.

### Example 1: Beauty Products Knowledge Graph

```bash
python kg2sft.py \
  --graph beauty_products.graphml \
  --output beauty_training \
  --count 500 \
  --domain beauty \
  --temperature 0.75 \
  --quality-threshold 0.80
```

**Sample Output:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What ingredient in Shiseido Cleanser helps with acne treatment?"
    },
    {
      "role": "assistant",
      "content": "Shiseido Cleanser contains Salicylic Acid, which is known to help with acne treatment by exfoliating the skin and unclogging pores."
    }
  ],
  "quality_score": 0.95
}
```

### Example 2: Generic Knowledge Graph

```bash
python kg2sft.py \
  --graph general_knowledge.graphml \
  --output general_training \
  --count 1000 \
  --sampling random
```

---

## 🛠️ Advanced Usage

Programmatic API access and customization for power users.

### Programmatic API

```python
from kg2sft import (
    KGToTrainingData,
    GenerationConfig,
    ExtractionConfig,
    ValidationConfig
)

# Initialize with custom configuration
trainer = KGToTrainingData(
    graph_path="my_graph.graphml",
    generation_config=GenerationConfig(
        count=500,
        temperature=0.8
    ),
    extraction_config=ExtractionConfig(
        max_hop_depth=5,
        sampling_strategy="random",
        dedup_threshold=0.90
    ),
    validation_config=ValidationConfig(
        quality_threshold=0.80
    ),
    domain="generic"
)

# Generate dataset
dataset = trainer.generate()

# Access results
print(f"Generated: {len(dataset)} examples")
print(f"Quality: {dataset.get_quality_report()}")

# Save custom format
dataset.save_jsonl("custom_output.jsonl")
trainer.print_report()
```

### Custom Domain Templates

Add new domain in `PromptTemplate.create_prompt_from_path()`:

```python
elif domain == "medical":
    template = f"""You are a medical knowledge assistant...
    
{path_str}

Generate a medically accurate Q&A pair..."""
```

## 📦 Project Structure

```
kg2sft/
├── kg2sft.py                     # Main application (1110 lines)
├── technology_knowledge.graphml  # Sample: Generic domain
├── beauty_products.graphml       # Sample: Beauty domain
├── .env                          # Configuration (create this, not in repo)
├── .gitignore                    # Git ignore rules
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies (optional)
├── output_training.jsonl         # Generated: JSONL format
├── output_training.json          # Generated: JSON format
└── checkpoint_training.jsonl     # Temporary checkpoint (auto-deleted)
```

## 🔄 Workflow Integration

### Use Generated Data for Fine-Tuning

**OpenAI/Azure OpenAI:**
```bash
# Upload training file
az openai fineTune create \
  --training-file output_training.jsonl \
  --model gpt-4.1-mini

# Or use OpenAI CLI
openai fine_tunes.create \
  --training_file output_training.jsonl \
  --model gpt-4.1-mini
```

**Hugging Face:**
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="output_training.jsonl")
# Continue with Transformers Trainer...
```

## 📊 Performance Optimization

### Speed Improvements
1. **Parallel Processing:** Future enhancement for batch API
2. **Caching:** Reuse extracted paths across runs
3. **Streaming:** Process large graphs in chunks

### Quality Improvements
1. **Lower rejection rate:** Adjust `quality_threshold`
2. **Better path diversity:** Tune `dedup_threshold`
3. **Domain tuning:** Create custom prompt templates

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional domain templates (medical, legal, scientific)
- [ ] Support for more graph formats (Neo4j, RDF, etc.)
- [ ] Batch API integration
- [ ] Resume from checkpoint functionality
- [ ] Web UI for configuration
- [ ] Multi-language support
- [ ] Unit tests and CI/CD

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with [NetworkX](https://networkx.org/) for graph operations
- Powered by [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- Inspired by the need for high-quality SLM training data

## 📮 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/easonlai/kg2sft/issues)
- **Discussions:** [GitHub Discussions](https://github.com/easonlai/kg2sft/discussions)

---

**Made with ❤️ for the AI community** | Version 1.0.0 | February 2026 | Author: Eason Lai
