"""
================================================================================
kg2sft: Knowledge Graph to SFT Training Data Generator
================================================================================

Purpose:
    Convert any Knowledge Graph (GraphML format) into high-quality training data
    for Small Language Model (SLM) fine-tuning using LLM synthesis.

Author: Eason Lai
Date: February 5, 2026
Version: 1.0.0
Status: Production-ready

How it works:
    1. Load knowledge graph (GraphML format)
    2. Extract multi-hop relationship paths
    3. Generate LLM prompts from paths
    4. Call Azure OpenAI API to generate QA pairs
    5. Validate generated pairs (quality scoring)
    6. Save as JSONL (Supervised Fine-Tuning format)
    7. Print cost and quality reports

Dependencies:
    pip install networkx python-dotenv openai tqdm

Usage:
    # Quick start with sample files
    python kg2sft.py --graph technology_knowledge.graphml --count 50
    python kg2sft.py --graph beauty_products.graphml --count 50 --domain beauty_product
    python kg2sft.py --graph makeup_knowledge_graph.graphml --count 50 --domain beauty_makeup
    
    # Auto-tuning mode (adjust params automatically to reach target count)
    python kg2sft.py --graph makeup_knowledge_graph.graphml --count 500 --domain beauty_makeup --auto
    
    # Or with your own graph
    python kg2sft.py --graph my_graph.graphml --count 100

Domain Templates:
    - generic:        General-purpose knowledge graph Q&A generation
    - beauty_product: Beauty products domain (brands, products, ingredients, skincare)
    - beauty_makeup:  Makeup consultation domain (skin types, undertones, finishes,
                     coverage, color theory, application techniques)

Configuration:
    Create .env file with:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_DEPLOYMENT
    - AZURE_OPENAI_MODEL

Output:
    - output_training.jsonl (SFT-ready format for fine-tuning)
    - output_training.json (Human-readable format)
    - Console report (cost, quality, statistics)

================================================================================
"""

import json
import os
import re
import random
import time
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from dotenv import load_dotenv
import networkx as nx
from openai import AzureOpenAI, OpenAIError, RateLimitError

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: CONFIGURATION CLASSES
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for Large Language Model Provider (Azure OpenAI)"""
    provider: str = "azure_openai"
    model_name: str = "gpt-4.1-mini"
    api_key: str = ""
    api_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = "2024-12-01-preview"
    max_retries: int = 3
    retry_delay: float = 1.0
    input_price_per_1k: float = 0.0004
    output_price_per_1k: float = 0.0016
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables (.env file)"""
        return cls(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
            model_name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
        )


@dataclass
class GenerationConfig:
    """Configuration for training data generation process.
    
    Attributes:
        count: Target number of training examples to generate
        batch_size: Reserved for future batch API implementation
        temperature: LLM sampling temperature (0.0=deterministic, 1.0=creative)
        top_p: Nucleus sampling parameter for generation diversity
        max_tokens: Maximum tokens per generated response
    """
    count: int = 100
    batch_size: int = 10  # Reserved for future batch processing
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 500


@dataclass
class ExtractionConfig:
    """Configuration for graph path extraction and sampling.
    
    Attributes:
        max_hop_depth: Maximum path length (number of edges to traverse)
        sampling_strategy: Path sampling approach ('frequency_weighted' or 'random')
        dedup_threshold: Similarity threshold for path deduplication (0-1, higher=stricter)
    """
    max_hop_depth: int = 999  # Effectively unlimited (cycle detection prevents infinite loops)
    sampling_strategy: str = "frequency_weighted"  # Options: 'frequency_weighted', 'random'
    dedup_threshold: float = 0.95  # Used for path deduplication


@dataclass
class ValidationConfig:
    """Configuration for quality validation of generated QA pairs.
    
    Quality Scoring (0.0-1.0):
        - Length (40%): Word count in range → full credit; too short → proportional; too long → 0.35
        - Format (30%): Has "?" → 0.3; starts with question word → 0.2
        - Substance (30%): 50+ chars + sentences → 0.3; 30+ → 0.2; 20+ → 0.1
    
    Hard Rejections:
        - Empty question/answer
        - Question <10 characters
        - Generic answers: "yes", "no", "i don't know", "maybe"
    
    Attributes:
        quality_threshold: Minimum quality score (0-1) to accept examples
        check_grammar: Reserved for future grammar checking feature
        min_length: Minimum answer length in words
        max_length: Maximum answer length in words
    """
    quality_threshold: float = 0.7
    check_grammar: bool = True  # Reserved for future grammar validation
    min_length: int = 20
    max_length: int = 500


# ============================================================================
# SECTION 2: GRAPH LOADING & REPRESENTATION
# ============================================================================

class Graph:
    """Unified graph representation using NetworkX"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_attributes = {}
        self.edge_attributes = {}
    
    def add_node(self, node_id: str, **attributes):
        """Add node to graph"""
        self.graph.add_node(node_id)
        self.node_attributes[node_id] = attributes
    
    def add_edge(self, source: str, target: str, **attributes):
        """Add directed edge to graph"""
        self.graph.add_edge(source, target)
        self.edge_attributes[(source, target)] = attributes
    
    def get_nodes(self) -> List[str]:
        """Get list of all node IDs"""
        return list(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Get list of all edges as (source, target) tuples"""
        return list(self.graph.edges())
    
    def get_node_label(self, node_id: str) -> str:
        """Get human-readable label for node.
        
        Checks common GraphML attribute names in priority order:
        name, label, title, display_name, text, value.
        Falls back to first available string attribute, then node_id.
        """
        attrs = self.node_attributes.get(node_id, {})
        # Check common label attribute names in priority order
        for key in ("name", "label", "title", "display_name", "text", "value"):
            if key in attrs and attrs[key]:
                return attrs[key]
        # Fallback: use first non-empty string attribute as label
        for key, val in attrs.items():
            if isinstance(val, str) and val and key not in ("id", "type", "category", "color", "shape", "tier"):
                return val
        return node_id
    
    def get_edge_relationship(self, source: str, target: str) -> str:
        """Get relationship type for edge.
        
        Checks common GraphML attribute names in priority order:
        label, relationship_type, relationship, rel, type, edge_type,
        connection_type, relation, predicate.
        Falls back to first available string attribute, then 'RELATED_TO'.
        """
        attrs = self.edge_attributes.get((source, target), {})
        # Check common edge label attribute names in priority order
        for key in ("label", "relationship_type", "relationship", "rel",
                    "type", "edge_type", "connection_type", "relation", "predicate"):
            if key in attrs and attrs[key]:
                return attrs[key]
        # Fallback: use first non-empty string attribute as relationship
        for key, val in attrs.items():
            if isinstance(val, str) and val and key not in ("id", "color", "weight", "reasoning"):
                return val
        return "RELATED_TO"
    
    def get_node_description(self, node_id: str) -> str:
        """Get description/detail for node (from 'description' or 'desc' attribute)"""
        attrs = self.node_attributes.get(node_id, {})
        return attrs.get("description", attrs.get("desc", ""))
    
    def get_edge_reasoning(self, source: str, target: str) -> str:
        """Get reasoning/rationale for edge (from 'reasoning' attribute)"""
        attrs = self.edge_attributes.get((source, target), {})
        return attrs.get("reasoning", "")
    
    def get_neighbors(self, node: str, depth: int = 1) -> Dict:
        """Get neighbors at specific depth"""
        neighbors = {}
        try:
            paths = nx.single_source_shortest_path_length(
                self.graph, node, cutoff=depth
            )
            for d in range(1, depth + 1):
                neighbors[f"hop_{d}"] = [
                    p for p in paths.keys() if paths[p] == d
                ]
        except (nx.NetworkXError, KeyError) as e:
            logger.warning(f"Error getting neighbors for node {node}: {e}")
        return neighbors


class GraphMLLoader:
    """Load graphs from GraphML (XML-based graph format)"""
    
    @staticmethod
    def load(filepath: str) -> Graph:
        """Load GraphML file and return unified Graph"""
        print(f"📂 Loading GraphML: {filepath}")
        
        nx_graph = nx.read_graphml(filepath)
        graph = Graph()
        
        for node_id, attrs in nx_graph.nodes(data=True):
            graph.add_node(node_id, **attrs)
        
        for source, target, attrs in nx_graph.edges(data=True):
            graph.add_edge(source, target, **attrs)
        
        print(f"✅ Loaded {len(graph.get_nodes())} nodes, "
              f"{len(graph.get_edges())} edges")
        
        return graph


# ============================================================================
# SECTION 3: PATH EXTRACTION & SAMPLING
# ============================================================================

class PathExtractor:
    """Extract multi-hop paths from knowledge graph with deduplication.
    
    Uses configurable sampling strategies and path similarity checking
    to ensure diverse training data generation.
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.extracted_paths = []
    
    def _calculate_path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """Calculate similarity between two paths using Jaccard index.
        
        Args:
            path1: First path as list of node IDs
            path2: Second path as list of node IDs
            
        Returns:
            Similarity score (0-1, where 1 = identical)
        """
        if not path1 or not path2:
            return 0.0
        
        set1 = set(path1)
        set2 = set(path2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_duplicate(self, new_path: List[str], existing_paths: List[Dict]) -> bool:
        """Check if path is too similar to existing paths.
        
        Args:
            new_path: New path to check
            existing_paths: List of already extracted path dicts
            
        Returns:
            True if path exceeds similarity threshold with any existing path
        """
        for existing in existing_paths:
            similarity = self._calculate_path_similarity(new_path, existing["nodes"])
            if similarity >= self.config.dedup_threshold:
                return True
        return False
    
    def extract_paths(self, graph: Graph, max_paths: int = 100) -> List[Dict]:
        """Extract paths from graph using configured sampling strategy"""
        print(f"\n📊 Extracting paths (strategy: {self.config.sampling_strategy})...")
        
        nodes = graph.get_nodes()
        if not nodes:
            print("⚠️  No nodes in graph!")
            return []
        
        # Calculate node degrees
        node_degrees = {}
        for source, target in graph.get_edges():
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        # Sample starting nodes
        if self.config.sampling_strategy == "frequency_weighted":
            total_degree = sum(node_degrees.values())
            if total_degree == 0:
                weights = [1/len(nodes) for _ in nodes]
            else:
                weights = [
                    node_degrees.get(n, 0) / total_degree 
                    for n in nodes
                ]
            # Allow sampling with replacement to get multiple paths from same nodes
            sampled_nodes = random.choices(
                nodes,
                weights=weights,
                k=max_paths
            )
        else:  # random
            # Allow sampling with replacement for consistent behavior
            sampled_nodes = random.choices(nodes, k=max_paths)
        
        # Extract paths with deduplication
        paths = []
        duplicates_skipped = 0
        
        for start_node in sampled_nodes:
            path = self._build_path(graph, start_node, depth=0, visited=None)
            
            if path and len(path) > 1:
                # Check for duplicates if threshold is set
                if self._is_duplicate(path, paths):
                    duplicates_skipped += 1
                    continue
                
                # Build path string with edge labels between nodes
                parts = [graph.get_node_label(path[0])]
                for j in range(1, len(path)):
                    edge_label = graph.get_edge_relationship(path[j-1], path[j])
                    parts.append(f"-[{edge_label}]->")
                    parts.append(graph.get_node_label(path[j]))
                path_str = " ".join(parts)
                
                # Build rich context with descriptions and reasoning
                rich_lines = []
                for j, node_id in enumerate(path):
                    label = graph.get_node_label(node_id)
                    desc = graph.get_node_description(node_id)
                    rich_lines.append(f"  [{label}]" + (f" - {desc}" if desc else ""))
                    if j < len(path) - 1:
                        edge_label = graph.get_edge_relationship(path[j], path[j+1])
                        reasoning = graph.get_edge_reasoning(path[j], path[j+1])
                        edge_line = f"    --({edge_label})-->"
                        if reasoning:
                            edge_line += f" (Reasoning: {reasoning})"
                        rich_lines.append(edge_line)
                rich_context = "\n".join(rich_lines)
                
                paths.append({
                    "start": start_node,
                    "nodes": path,
                    "path_str": path_str,
                    "rich_context": rich_context,
                    "length": len(path)
                })
            
            if len(paths) >= max_paths:
                break
        
        print(f"✅ Extracted {len(paths)} paths (skipped {duplicates_skipped} duplicates)")
        self.extracted_paths = paths
        return paths
    
    def _build_path(self, graph: Graph, node: str, depth: int, visited: Optional[Set[str]] = None) -> List[str]:
        """Build a single path using random walk with cycle detection"""
        if visited is None:
            visited = set()
        
        if depth >= self.config.max_hop_depth or node in visited:
            return [node]
        
        visited.add(node)
        neighbors = list(graph.graph.successors(node))
        
        # Filter out already visited neighbors to avoid cycles
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        if not unvisited_neighbors:
            return [node]
        
        next_node = random.choice(unvisited_neighbors)
        return [node] + self._build_path(graph, next_node, depth + 1, visited)


# ============================================================================
# SECTION 4: PROMPT TEMPLATE SYSTEM
# ============================================================================

class PromptTemplate:
    """Generate LLM prompts from graph paths"""
    
    @staticmethod
    def create_prompt_from_path(path_str: str, domain: str = "generic", rich_context: str = "") -> str:
        """Create LLM prompt from graph path.
        
        Args:
            path_str: Path string with edge labels (e.g., "A -[requires]-> B")
            domain: Domain template key:
                - 'generic': General-purpose knowledge graph Q&A
                - 'beauty_product': Beauty products (brands, products, ingredients, skincare benefits)
                - 'beauty_makeup': Makeup consultation (skin types, undertones, finishes,
                  coverage levels, color theory, application procedures)
            rich_context: Optional detailed context with node descriptions and edge reasoning
        """
        
        # === Domain: beauty_makeup ===
        # For makeup consultation knowledge graphs (e.g., makeup_knowledge_graph.graphml)
        # Covers: skin depth, undertones, skin types, concerns, finishes, coverage,
        #         formulations, color theory, color correction, application procedures
        if domain == "beauty_makeup":
            context_block = rich_context if rich_context else path_str
            template = f"""You are a professional makeup artist and beauty consultant data synthesis assistant.

Your task is to generate a high-quality question-answer pair based on the following makeup consultation knowledge graph path.

Relationship path:
{path_str}

Detailed context (node descriptions and reasoning):
{context_block}

IMPORTANT: Pay close attention to the edge labels:
- "requires", "prefers", "works_with", "recommends" = POSITIVE recommendations
- "avoid" = NEGATIVE contraindication (do NOT recommend this combination)
- "input_to", "determines" = decision flow connections
- "requires_procedure" = application technique needed

Guidelines:
1. The question should be natural and from a customer seeking makeup advice
2. Questions should ask about skin type matches, finish recommendations, what to avoid, application techniques, or color matching
3. The answer MUST respect the edge direction: if the edge says "avoid", the answer should warn AGAINST that combination
4. Include the expert REASONING (the "why") in the answer, not just the recommendation
5. The answer should sound like professional makeup artist advice
6. Be specific - mention actual skin types, finishes, formulations by name

Generate your response as valid JSON with exactly this format:
{{
    "question": "A natural customer question about makeup selection or technique",
    "answer": "Expert makeup artist advice with reasoning, respecting positive/negative relationships",
    "difficulty": "easy|medium|hard"
}}"""
        
        # === Domain: beauty_product ===
        # For beauty product knowledge graphs (e.g., beauty_products.graphml)
        # Covers: brands, products, ingredients, benefits, skincare routines
        elif domain == "beauty_product":
            context_section = ""
            if rich_context:
                context_section = f"""\n\nDetailed context (node descriptions and reasoning):\n{rich_context}"""
            template = f"""You are a beauty expert data synthesis assistant specializing in skincare, cosmetics, and beauty products.

Your task is to generate a high-quality question-answer pair based on the following beauty product knowledge graph relationship:

{path_str}{context_section}

Guidelines:
1. The question should be natural and from a consumer's perspective
2. Questions might ask about ingredients, benefits, products, or combinations
3. The answer should be accurate and based ONLY on the relationship path shown
4. Answers should be professional but conversational
5. Avoid making claims not directly supported by the path

Generate your response as valid JSON with exactly this format:
{{
    "question": "A natural, consumer-focused question about this relationship",
    "answer": "An accurate answer based on the relationship path",
    "difficulty": "easy|medium|hard"
}}"""
        
        # === Domain: generic (default) ===
        # For any general-purpose knowledge graph (e.g., technology_knowledge.graphml)
        else:
            context_section = ""
            if rich_context:
                context_section = f"""\n\nDetailed context (node descriptions and reasoning):\n{rich_context}"""
            template = f"""You are a data synthesis assistant helping to generate training data from knowledge graphs.

Your task is to generate a high-quality question-answer pair based on the following relationship path in a knowledge graph:

{path_str}{context_section}

Guidelines:
1. The question should be natural and test understanding of the relationship
2. The answer should be factually accurate based ONLY on the path shown
3. The Q&A pair should be suitable for training a language model
4. Questions should vary in style (direct, comparative, explanatory, etc.)
5. Answers should be clear and complete

Generate your response as valid JSON with exactly this format:
{{
    "question": "A clear question testing understanding of this relationship",
    "answer": "An accurate answer based on the relationship path",
    "difficulty": "easy|medium|hard"
}}"""
        
        return template


# ============================================================================
# SECTION 5: LLM INTEGRATION & API CALLS
# ============================================================================

class AzureOpenAIClient:
    """Azure OpenAI LLM client for generating training data"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.api_endpoint,
            max_retries=0  # We handle retries manually
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
    
    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Remove markdown code blocks and repair common JSON issues.
        
        Handles:
            - Markdown code block wrappers
            - Unescaped newlines inside JSON string values
            - Unescaped control characters
        """
        # Remove markdown code blocks
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Try parsing as-is first (fast path)
        try:
            json.loads(text)
            return text
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Repair: fix unescaped newlines/tabs inside JSON string values
        # Strategy: find content between quotes and escape literal newlines
        def escape_string_content(match):
            """Escape newlines and tabs inside JSON string values"""
            content = match.group(1)
            content = content.replace('\\n', '\x00NEWLINE\x00')  # Preserve already-escaped
            content = content.replace('\\t', '\x00TAB\x00')
            content = content.replace('\n', '\\n')
            content = content.replace('\r', '\\r')
            content = content.replace('\t', '\\t')
            content = content.replace('\x00NEWLINE\x00', '\\n')  # Restore
            content = content.replace('\x00TAB\x00', '\\t')
            return f'"{content}"'
        
        # Match JSON string values (handles escaped quotes inside)
        text = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_string_content, text, flags=re.DOTALL)
        
        return text
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        """Generate text using Azure OpenAI API with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.call_count += 1
                
                content = response.choices[0].message.content
                return self._clean_json_response(content) if content else None
            
            except RateLimitError as e:
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{self.config.max_retries})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit error after {self.config.max_retries} attempts: {e}")
                    return None
            
            except OpenAIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error in API call: {type(e).__name__}: {e}")
                return None
        
        return None
    
    def get_cost_report(self) -> Dict:
        """Calculate and return cost report using configurable pricing"""
        input_cost = (self.total_input_tokens / 1000) * self.config.input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * self.config.output_price_per_1k
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(total_cost, 4),
            "api_calls": self.call_count,
            "avg_cost_per_example": round(
                total_cost / self.call_count, 6
            ) if self.call_count > 0 else 0
        }


# ============================================================================
# SECTION 6: QUALITY VALIDATION
# ============================================================================

class Validator:
    """Quality validation for generated QA pairs with intelligent scoring.
    
    Implements multi-criteria validation with smart partial credit:
        - Hard requirements (immediate rejection)
        - Graduated scoring (0.0-1.0 scale)
        - Generic answer detection
        - Sentence structure analysis
    
    Scoring Components:
        - Length (40%): Proportional credit based on word count
        - Format (30%): Question mark or question word detection
        - Substance (30%): Character count + sentence structure
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, qa_pair: Dict) -> Tuple[bool, float]:
        """Validate QA pair and return quality score.
        
        Scoring Breakdown (0.0-1.0 scale):
            - Length appropriateness: 40%
            - Question format: 30%
            - Answer substance: 30%
        
        Returns:
            Tuple of (is_valid, quality_score)
        """
        
        if not qa_pair or "question" not in qa_pair or "answer" not in qa_pair:
            return False, 0.0
        
        question = str(qa_pair.get("question", "")).strip()
        answer = str(qa_pair.get("answer", "")).strip()
        
        # Hard requirements (reject immediately if failed)
        if not question or not answer:
            return False, 0.0
        
        if len(question) < 10:  # Question too short (likely incomplete)
            return False, 0.0
        
        # Check for generic/useless answers
        generic_answers = {"yes", "no", "i don't know", "not sure", "maybe"}
        if answer.lower().strip(".,!? ") in generic_answers:
            return False, 0.0
        
        # Calculate quality score (0.0-1.0)
        score = 0.0
        
        # 1. Length appropriateness (40% of score)
        answer_words = len(answer.split())
        
        if self.config.min_length <= answer_words <= self.config.max_length:
            # Perfect length
            score += 0.4
        elif answer_words < self.config.min_length:
            # Too short - proportional partial credit
            if answer_words >= 10:  # At least some content
                ratio = answer_words / self.config.min_length
                score += 0.4 * ratio
            # else: 0 points (too short, already < min)
        else:
            # Too long - still valuable, slight penalty
            # Long detailed answers are often good for training!
            score += 0.35  # Slight penalty but still high credit
        
        # 2. Question format validation (30% of score)
        if "?" in question:
            score += 0.3
        elif any(question.lower().startswith(word) for word in 
                 ["what", "how", "why", "when", "where", "who", "which", "can", "is", "does", "do"]):
            # Question word present but missing '?' - partial credit
            score += 0.2
        
        # 3. Answer substance (30% of score)
        # Check for meaningful content, not just character count
        answer_chars = len(answer)
        answer_sentences = answer.count('.') + answer.count('!') + answer.count('?')
        
        if answer_chars >= 50 and answer_sentences >= 1:
            # Substantial answer with sentence structure
            score += 0.3
        elif answer_chars >= 30:
            # Moderate substance
            score += 0.2
        elif answer_chars >= 20:
            # Minimal substance
            score += 0.1
        # else: 0 points for very short answers
        
        # Final validation
        is_valid = score >= self.config.quality_threshold
        
        return is_valid, min(1.0, score)


# ============================================================================
# SECTION 7: TRAINING DATASET MANAGEMENT
# ============================================================================

@dataclass
class TrainingDataset:
    """Container for generated training examples"""
    examples: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_example(self, question: str, answer: str, quality_score: float = 1.0):
        """Add a validated example to dataset"""
        self.examples.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ],
            "quality_score": quality_score
        })
    
    def save_jsonl(self, filepath: str):
        """Save dataset as JSONL (SFT format)"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for ex in self.examples:
                # Remove quality_score from output (internal metric)
                output_ex = {k: v for k, v in ex.items() if k != 'quality_score'}
                f.write(json.dumps(output_ex) + '\n')
        logger.info(f"Saved {len(self.examples)} examples to {filepath}")
        print(f"\n✅ Saved {len(self.examples)} examples to {filepath}")
    
    def save_json(self, filepath: str):
        """Save dataset as pretty-printed JSON for human review.
        
        Unlike JSONL format, this creates a single JSON array with
        indentation for easy reading. Useful for manual inspection
        and quality assessment. Includes quality_score for analysis.
        
        Args:
            filepath: Output file path (.json extension recommended)
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.examples)} examples to {filepath}")
    
    def get_quality_report(self) -> Dict:
        """Generate quality metrics report"""
        if not self.examples:
            return {"avg_quality": 0, "total_examples": 0}
        
        scores = [ex.get("quality_score", 1.0) for ex in self.examples]
        
        return {
            "total_examples": len(self.examples),
            "avg_quality": round(sum(scores) / len(scores), 3) if scores else 0,
            "min_quality": round(min(scores), 3) if scores else 0,
            "max_quality": round(max(scores), 3) if scores else 1.0
        }
    
    def __len__(self):
        """Return number of examples"""
        return len(self.examples)
    
    @staticmethod
    def cleanup_checkpoint(checkpoint_path: str):
        """Remove checkpoint file after successful completion.
        
        Args:
            checkpoint_path: Path to checkpoint file to remove
        """
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info(f"Cleaned up checkpoint: {checkpoint_path}")
                print(f"🧹 Cleaned up checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")


# ============================================================================
# SECTION 8: MAIN ORCHESTRATOR
# ============================================================================

class KGToTrainingData:
    """Main orchestrator for Knowledge Graph to Training Data pipeline.
    
    Coordinates the end-to-end process of converting knowledge graphs into
    training data for supervised fine-tuning of language models.
    
    Key Features:
        - Automatic path extraction with deduplication
        - Retry logic for API failures
        - Real-time progress tracking
        - Periodic checkpointing
        - Comprehensive cost and quality reporting
    
    Attributes:
        graph_path: Path to input GraphML file
        domain: Domain template for prompt generation
        llm_config: Azure OpenAI configuration
        generation_config: Generation parameters
        extraction_config: Path extraction settings
        validation_config: Quality validation criteria
        graph: Loaded Graph instance
        llm_client: Azure OpenAI client
        path_extractor: Path extraction engine
        validator: Quality validator
        dataset: Generated training dataset
    """
    
    def __init__(
        self,
        graph_path: str,
        llm_config: Optional[LLMConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        extraction_config: Optional[ExtractionConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        domain: str = "generic"
    ):
        self.graph_path = graph_path
        self.domain = domain
        
        self.llm_config = llm_config or LLMConfig.from_env()
        self.generation_config = generation_config or GenerationConfig()
        self.extraction_config = extraction_config or ExtractionConfig()
        self.validation_config = validation_config or ValidationConfig()
        
        self.graph: Optional[Graph] = None
        self.llm_client: Optional[AzureOpenAIClient] = None
        self.path_extractor: Optional[PathExtractor] = None
        self.validator: Optional[Validator] = None
        self.dataset = TrainingDataset()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        loader = GraphMLLoader()
        self.graph = loader.load(self.graph_path)
        
        if not self.llm_config.api_key:
            raise ValueError(
                "❌ AZURE_OPENAI_API_KEY not set. "
                "Set it in .env file or environment variables"
            )
        
        self.llm_client = AzureOpenAIClient(self.llm_config)
        self.path_extractor = PathExtractor(self.extraction_config)
        self.validator = Validator(self.validation_config)
    
    def generate(self, count: Optional[int] = None) -> TrainingDataset:
        """Generate training data from knowledge graph"""
        count = count or self.generation_config.count
        
        print(f"\n🚀 Starting generation of {count} examples...")
        print(f"   Domain: {self.domain}")
        print(f"   Temperature: {self.generation_config.temperature}")
        
        paths = self.path_extractor.extract_paths(self.graph, max_paths=count)
        if not paths:
            print("❌ No paths extracted from graph!")
            return self.dataset
        
        print(f"\n🤖 Generating training examples...")
        generated = 0
        rejected = 0
        
        # Create progress iterator
        if HAS_TQDM:
            path_iterator = tqdm(enumerate(paths), total=len(paths), desc="Generating")
        else:
            path_iterator = enumerate(paths)
        
        # Save checkpoint every 10% of progress (minimum every 10 examples)
        checkpoint_interval = max(10, count // 10)
        checkpoint_path = "checkpoint_training.jsonl"
        
        for i, path_data in path_iterator:
            if len(self.dataset) >= count:
                break
            
            prompt = PromptTemplate.create_prompt_from_path(
                path_data["path_str"],
                domain=self.domain,
                rich_context=path_data.get("rich_context", "")
            )
            
            response_text = self.llm_client.generate(
                prompt,
                temperature=self.generation_config.temperature,
                max_tokens=self.generation_config.max_tokens
            )
            
            if response_text:
                try:
                    qa_pair = json.loads(response_text)
                    is_valid, quality_score = self.validator.validate(qa_pair)
                    
                    if is_valid:
                        self.dataset.add_example(
                            qa_pair["question"],
                            qa_pair["answer"],
                            quality_score
                        )
                        generated += 1
                        logger.debug(f"Generated example {generated}/{count}")
                    else:
                        rejected += 1
                        logger.debug(f"Rejected example (quality score: {quality_score:.2f})")
                
                except json.JSONDecodeError as e:
                    rejected += 1
                    logger.warning(f"JSON decode error: {e}. Response: {response_text[:100]}...")
            else:
                rejected += 1
                logger.warning("No response from LLM")
            
            # Checkpoint progress
            if (generated + rejected) % checkpoint_interval == 0 and len(self.dataset) > 0:
                checkpoint_path = "checkpoint_training.jsonl"
                self.dataset.save_jsonl(checkpoint_path)
                logger.info(f"Checkpoint saved: {generated} generated, {rejected} rejected")
            
            # Progress update for non-tqdm
            if not HAS_TQDM:
                progress_interval = max(1, len(paths) // 10)
                if (i + 1) % progress_interval == 0:
                    print(f"   Progress: {i + 1}/{len(paths)} paths processed, "
                          f"{generated} generated, {rejected} rejected")
        
        print(f"\n✅ Generation complete!")
        print(f"   Generated: {generated} examples")
        print(f"   Rejected: {rejected} examples")
        
        # Cleanup checkpoint file on successful completion
        if os.path.exists(checkpoint_path):
            self.dataset.cleanup_checkpoint(checkpoint_path)
        
        cost_report = self.llm_client.get_cost_report()
        quality_report = self.dataset.get_quality_report()
        
        self.dataset.metadata = {
            "generation_config": {
                "count_requested": count,
                "count_generated": len(self.dataset),
                "count_rejected": rejected,
                "acceptance_rate": round(
                    generated / (generated + rejected), 3
                ) if (generated + rejected) > 0 else 0
            },
            "cost": cost_report,
            "quality": quality_report,
            "graph": {
                "nodes": len(self.graph.get_nodes()),
                "edges": len(self.graph.get_edges()),
                "paths_extracted": len(paths)
            }
        }
        
        return self.dataset
    
    def print_report(self):
        """Print comprehensive generation report"""
        if not self.dataset.metadata:
            print("⚠️  No generation report available")
            return
        
        print("\n" + "="*60)
        print("📊 GENERATION REPORT")
        print("="*60)
        
        gen_cfg = self.dataset.metadata.get("generation_config", {})
        print(f"\n📈 Generation Statistics:")
        print(f"   Requested: {gen_cfg.get('count_requested', 0)} examples")
        print(f"   Generated: {gen_cfg.get('count_generated', 0)} examples")
        print(f"   Rejected: {gen_cfg.get('count_rejected', 0)} examples")
        print(f"   Acceptance Rate: {gen_cfg.get('acceptance_rate', 0)*100:.1f}%")
        
        cost = self.dataset.metadata.get("cost", {})
        print(f"\n💰 Cost Report:")
        print(f"   Input Tokens: {cost.get('input_tokens', 0):,}")
        print(f"   Output Tokens: {cost.get('output_tokens', 0):,}")
        print(f"   Input Cost: ${cost.get('input_cost', 0):.4f}")
        print(f"   Output Cost: ${cost.get('output_cost', 0):.4f}")
        print(f"   Total Cost: ${cost.get('total_cost', 0):.4f}")
        print(f"   Cost per Example: ${cost.get('avg_cost_per_example', 0):.6f}")
        print(f"   API Calls: {cost.get('api_calls', 0)}")
        
        quality = self.dataset.metadata.get("quality", {})
        print(f"\n⭐ Quality Report:")
        print(f"   Total Examples: {quality.get('total_examples', 0)}")
        print(f"   Average Quality: {quality.get('avg_quality', 0):.3f}/1.0")
        print(f"   Min Quality: {quality.get('min_quality', 0):.3f}")
        print(f"   Max Quality: {quality.get('max_quality', 0):.3f}")
        
        graph_info = self.dataset.metadata.get("graph", {})
        print(f"\n📊 Graph Statistics:")
        print(f"   Nodes: {graph_info.get('nodes', 0)}")
        print(f"   Edges: {graph_info.get('edges', 0)}")
        print(f"   Paths Extracted: {graph_info.get('paths_extracted', 0)}")
        
        print("\n" + "="*60)


# ============================================================================
# SECTION 9: AUTO-TUNING MODE
# ============================================================================

@dataclass
class AutoTuneResult:
    """Result of an auto-tuning run, tracking parameters and quality impact."""
    iteration: int = 0
    quality_threshold: float = 0.7
    dedup_threshold: float = 0.95
    temperature: float = 0.7
    sampling_strategy: str = "frequency_weighted"
    generated_count: int = 0
    rejected_count: int = 0
    avg_quality: float = 0.0
    quality_warnings: List[str] = field(default_factory=list)


class AutoTuner:
    """Automatic parameter tuner that adjusts generation settings to hit a
    target sample count, progressively relaxing quality constraints.

    Strategy (each iteration relaxes further):
        Iter 0 – Default parameters (best quality)
        Iter 1 – Lower dedup threshold (0.95 → 0.70), allow more similar paths
        Iter 2 – Lower quality threshold (0.7 → 0.5), accept more borderline examples
        Iter 3 – Raise temperature (0.7 → 0.9), increase generation diversity
        Iter 4 – Switch to random sampling, lower dedup further (0.50)
        Iter 5 – Aggressive: quality 0.3, dedup 0.30, temperature 1.0

    The tuner stops as soon as the target count is reached.
    """

    # Each tier defines the parameter overrides for that relaxation level
    TUNING_TIERS = [
        {   # Tier 0: defaults (highest quality)
            "quality_threshold": 0.7,
            "dedup_threshold": 0.95,
            "temperature": 0.7,
            "sampling_strategy": "frequency_weighted",
            "description": "Default parameters (best quality)",
        },
        {   # Tier 1: relax dedup
            "quality_threshold": 0.7,
            "dedup_threshold": 0.70,
            "temperature": 0.7,
            "sampling_strategy": "frequency_weighted",
            "description": "Relaxed dedup threshold",
        },
        {   # Tier 2: relax quality
            "quality_threshold": 0.5,
            "dedup_threshold": 0.70,
            "temperature": 0.7,
            "sampling_strategy": "frequency_weighted",
            "description": "Lower quality threshold",
        },
        {   # Tier 3: raise temperature
            "quality_threshold": 0.5,
            "dedup_threshold": 0.60,
            "temperature": 0.9,
            "sampling_strategy": "frequency_weighted",
            "description": "Higher temperature for diversity",
        },
        {   # Tier 4: random sampling
            "quality_threshold": 0.5,
            "dedup_threshold": 0.50,
            "temperature": 0.9,
            "sampling_strategy": "random",
            "description": "Random sampling with relaxed dedup",
        },
        {   # Tier 5: aggressive
            "quality_threshold": 0.3,
            "dedup_threshold": 0.30,
            "temperature": 1.0,
            "sampling_strategy": "random",
            "description": "Aggressive (minimum quality, maximum yield)",
        },
    ]

    def __init__(self, graph_path: str, domain: str, target_count: int,
                 output_prefix: str = "output_training",
                 max_depth: int = 999):
        self.graph_path = graph_path
        self.domain = domain
        self.target_count = target_count
        self.output_prefix = output_prefix
        self.max_depth = max_depth
        self.tier_results: List[AutoTuneResult] = []

    # ------------------------------------------------------------------
    def _analyse_graph(self) -> Dict:
        """Quick analysis of the graph to estimate generation capacity."""
        loader = GraphMLLoader()
        graph = loader.load(self.graph_path)
        nodes = len(graph.get_nodes())
        edges = len(graph.get_edges())
        # Rough heuristic: each edge can produce ~1-2 unique paths
        estimated_capacity = max(edges, nodes) * 2
        return {
            "nodes": nodes,
            "edges": edges,
            "estimated_capacity": estimated_capacity,
        }

    # ------------------------------------------------------------------
    def _run_iteration(self, tier_index: int,
                       remaining: int,
                       existing_dataset: Optional[TrainingDataset] = None
                       ) -> Tuple[TrainingDataset, AutoTuneResult]:
        """Run one generation iteration with the given tier parameters."""
        tier = self.TUNING_TIERS[tier_index]

        # Over-request paths to compensate for rejections
        overshoot = int(remaining * 1.5) + 10

        trainer = KGToTrainingData(
            graph_path=self.graph_path,
            generation_config=GenerationConfig(
                count=overshoot,
                temperature=tier["temperature"],
                max_tokens=500,
            ),
            extraction_config=ExtractionConfig(
                max_hop_depth=self.max_depth,
                sampling_strategy=tier["sampling_strategy"],
                dedup_threshold=tier["dedup_threshold"],
            ),
            validation_config=ValidationConfig(
                quality_threshold=tier["quality_threshold"],
                min_length=20 if tier["quality_threshold"] >= 0.5 else 10,
                max_length=500,
            ),
            domain=self.domain,
        )

        dataset = trainer.generate(count=remaining)

        # Build result
        quality_report = dataset.get_quality_report()
        gen_cfg = dataset.metadata.get("generation_config", {})

        result = AutoTuneResult(
            iteration=tier_index,
            quality_threshold=tier["quality_threshold"],
            dedup_threshold=tier["dedup_threshold"],
            temperature=tier["temperature"],
            sampling_strategy=tier["sampling_strategy"],
            generated_count=gen_cfg.get("count_generated", 0),
            rejected_count=gen_cfg.get("count_rejected", 0),
            avg_quality=quality_report.get("avg_quality", 0.0),
        )

        # Record quality warnings
        warnings = []
        if tier["quality_threshold"] < 0.7:
            warnings.append(
                f"Quality threshold lowered to {tier['quality_threshold']:.1f} "
                f"(default: 0.7)"
            )
        if tier["dedup_threshold"] < 0.95:
            warnings.append(
                f"Dedup threshold lowered to {tier['dedup_threshold']:.2f} "
                f"(default: 0.95) — some examples may be more similar"
            )
        if tier["temperature"] > 0.7:
            warnings.append(
                f"Temperature raised to {tier['temperature']:.1f} "
                f"(default: 0.7) — outputs may be less deterministic"
            )
        if tier["sampling_strategy"] == "random":
            warnings.append(
                "Switched to random sampling — path selection less optimised"
            )
        result.quality_warnings = warnings

        return dataset, result

    # ------------------------------------------------------------------
    def run(self) -> TrainingDataset:
        """Execute the auto-tuning loop.

        Returns the merged TrainingDataset once target count is reached
        or all tiers are exhausted.
        """
        print("\n" + "=" * 60)
        print("🤖 AUTO-TUNING MODE")
        print("=" * 60)

        # Analyse graph
        graph_info = self._analyse_graph()
        print(f"\n📊 Graph Analysis:")
        print(f"   Nodes: {graph_info['nodes']}")
        print(f"   Edges: {graph_info['edges']}")
        print(f"   Estimated capacity: ~{graph_info['estimated_capacity']} examples")
        print(f"   Target count: {self.target_count} examples")

        if self.target_count > graph_info["estimated_capacity"]:
            print(f"\n⚠️  Target ({self.target_count}) exceeds estimated graph "
                  f"capacity (~{graph_info['estimated_capacity']}). "
                  f"Quality trade-offs will be applied.")

        merged_dataset = TrainingDataset()
        total_api_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_rejected = 0

        for tier_idx, tier in enumerate(self.TUNING_TIERS):
            remaining = self.target_count - len(merged_dataset)
            if remaining <= 0:
                break

            print(f"\n{'─'*50}")
            print(f"🔄 Iteration {tier_idx}: {tier['description']}")
            print(f"   Remaining: {remaining} examples needed")
            print(f"   Params: quality≥{tier['quality_threshold']}, "
                  f"dedup={tier['dedup_threshold']}, "
                  f"temp={tier['temperature']}, "
                  f"sampling={tier['sampling_strategy']}")

            dataset, result = self._run_iteration(tier_idx, remaining)
            self.tier_results.append(result)

            # Merge new examples into combined dataset
            for ex in dataset.examples:
                if len(merged_dataset) >= self.target_count:
                    break
                merged_dataset.examples.append(ex)

            # Accumulate cost tracking from metadata
            cost = dataset.metadata.get("cost", {})
            total_api_calls += cost.get("api_calls", 0)
            total_input_tokens += cost.get("input_tokens", 0)
            total_output_tokens += cost.get("output_tokens", 0)
            total_rejected += result.rejected_count

            print(f"   ✅ Got {result.generated_count} examples "
                  f"(total: {len(merged_dataset)}/{self.target_count})")

            if len(merged_dataset) >= self.target_count:
                print(f"\n🎯 Target reached at iteration {tier_idx}!")
                break
        else:
            print(f"\n⚠️  Exhausted all tuning tiers. "
                  f"Generated {len(merged_dataset)}/{self.target_count} examples.")

        # Build merged metadata
        llm_cfg = LLMConfig.from_env()
        input_cost = (total_input_tokens / 1000) * llm_cfg.input_price_per_1k
        output_cost = (total_output_tokens / 1000) * llm_cfg.output_price_per_1k
        total_cost = input_cost + output_cost

        merged_dataset.metadata = {
            "generation_config": {
                "mode": "auto",
                "count_requested": self.target_count,
                "count_generated": len(merged_dataset),
                "count_rejected": total_rejected,
                "acceptance_rate": round(
                    len(merged_dataset) / (len(merged_dataset) + total_rejected), 3
                ) if (len(merged_dataset) + total_rejected) > 0 else 0,
                "tuning_iterations": len(self.tier_results),
            },
            "cost": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(total_cost, 4),
                "api_calls": total_api_calls,
                "avg_cost_per_example": round(
                    total_cost / len(merged_dataset), 6
                ) if len(merged_dataset) > 0 else 0,
            },
            "quality": merged_dataset.get_quality_report(),
            "graph": graph_info,
        }

        return merged_dataset

    # ------------------------------------------------------------------
    def print_auto_report(self, dataset: TrainingDataset):
        """Print comprehensive auto-tuning report with quality warnings."""
        if not dataset.metadata:
            print("⚠️  No generation report available")
            return

        print("\n" + "=" * 60)
        print("📊 AUTO-TUNING GENERATION REPORT")
        print("=" * 60)

        gen_cfg = dataset.metadata.get("generation_config", {})
        print(f"\n📈 Generation Statistics:")
        print(f"   Mode: AUTO-TUNING")
        print(f"   Requested: {gen_cfg.get('count_requested', 0)} examples")
        print(f"   Generated: {gen_cfg.get('count_generated', 0)} examples")
        print(f"   Rejected: {gen_cfg.get('count_rejected', 0)} examples")
        print(f"   Acceptance Rate: {gen_cfg.get('acceptance_rate', 0)*100:.1f}%")
        print(f"   Tuning Iterations: {gen_cfg.get('tuning_iterations', 0)}")

        # Per-iteration breakdown
        if self.tier_results:
            print(f"\n🔧 Iteration Breakdown:")
            for r in self.tier_results:
                tier_desc = self.TUNING_TIERS[r.iteration]["description"]
                print(f"   Iter {r.iteration} ({tier_desc}):")
                print(f"      Generated: {r.generated_count}, "
                      f"Rejected: {r.rejected_count}, "
                      f"Avg Quality: {r.avg_quality:.3f}")

        cost = dataset.metadata.get("cost", {})
        print(f"\n💰 Cost Report:")
        print(f"   Input Tokens: {cost.get('input_tokens', 0):,}")
        print(f"   Output Tokens: {cost.get('output_tokens', 0):,}")
        print(f"   Input Cost: ${cost.get('input_cost', 0):.4f}")
        print(f"   Output Cost: ${cost.get('output_cost', 0):.4f}")
        print(f"   Total Cost: ${cost.get('total_cost', 0):.4f}")
        print(f"   Cost per Example: ${cost.get('avg_cost_per_example', 0):.6f}")
        print(f"   API Calls: {cost.get('api_calls', 0)}")

        quality = dataset.metadata.get("quality", {})
        print(f"\n⭐ Quality Report:")
        print(f"   Total Examples: {quality.get('total_examples', 0)}")
        print(f"   Average Quality: {quality.get('avg_quality', 0):.3f}/1.0")
        print(f"   Min Quality: {quality.get('min_quality', 0):.3f}")
        print(f"   Max Quality: {quality.get('max_quality', 0):.3f}")

        graph_info = dataset.metadata.get("graph", {})
        print(f"\n📊 Graph Statistics:")
        print(f"   Nodes: {graph_info.get('nodes', 0)}")
        print(f"   Edges: {graph_info.get('edges', 0)}")
        est = graph_info.get('estimated_capacity', 'N/A')
        print(f"   Estimated Capacity: ~{est}")

        # === QUALITY IMPACT WARNING SECTION ===
        all_warnings = []
        max_tier_used = -1
        for r in self.tier_results:
            if r.generated_count > 0:
                max_tier_used = max(max_tier_used, r.iteration)
                all_warnings.extend(r.quality_warnings)

        # Deduplicate warnings
        seen = set()
        unique_warnings = []
        for w in all_warnings:
            if w not in seen:
                seen.add(w)
                unique_warnings.append(w)

        print(f"\n{'='*60}")
        print(f"⚠️  QUALITY IMPACT ASSESSMENT (Auto-Tuning Mode)")
        print(f"{'='*60}")

        if max_tier_used <= 0:
            print(f"   ✅ All examples generated with DEFAULT parameters.")
            print(f"   Quality level: HIGH — no trade-offs were needed.")
        elif max_tier_used <= 2:
            print(f"   🟡 Moderate quality trade-offs were applied.")
            print(f"   Quality level: MODERATE — some parameters were relaxed.")
            print(f"   Highest tuning tier used: {max_tier_used} "
                  f"({self.TUNING_TIERS[max_tier_used]['description']})")
        else:
            print(f"   🔴 Significant quality trade-offs were applied.")
            print(f"   Quality level: REDUCED — aggressive parameters were used")
            print(f"   to meet the target count.")
            print(f"   Highest tuning tier used: {max_tier_used} "
                  f"({self.TUNING_TIERS[max_tier_used]['description']})")

        if unique_warnings:
            print(f"\n   Trade-offs applied:")
            for w in unique_warnings:
                print(f"   • {w}")

        print(f"\n   💡 Recommendations:")
        if max_tier_used <= 0:
            print(f"   • Data is high quality and ready for fine-tuning.")
        else:
            print(f"   • Review generated examples for accuracy before fine-tuning.")
            if max_tier_used >= 3:
                print(f"   • Consider manually filtering low-quality examples.")
                print(f"   • A smaller graph may not support {gen_cfg.get('count_requested', 0)} "
                      f"high-quality examples.")
                print(f"   • Consider enriching your knowledge graph with more "
                      f"nodes/edges for better results.")
            if quality.get('avg_quality', 1.0) < 0.6:
                print(f"   • Average quality ({quality.get('avg_quality', 0):.3f}) is below 0.6 —"
                      f" manual review strongly recommended.")

        print(f"\n{'='*60}")


# ============================================================================
# SECTION 10: MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point with CLI argument support.
    
    Command-line Arguments:
        --graph: Path to input GraphML file (default: beauty_products.graphml)
        --output: Output filename prefix (default: output_training)
        --count: Number of examples to generate (default: 10)
        --domain: Domain template ('generic', 'beauty_product', or 'beauty_makeup', default: generic)
        --temperature: LLM temperature 0-1 (default: 0.7)
        --quality-threshold: Minimum quality score 0-1 (default: 0.7)
        --max-depth: Maximum path depth (default: 999, paths explore naturally)
        --dedup-threshold: Path similarity threshold 0-1 (default: 0.95)
        --sampling: Sampling strategy ('frequency_weighted' or 'random')
        --auto: Enable auto-tuning mode — automatically adjusts parameters to hit
                the target --count. May trade off quality for quantity. A quality
                impact report is printed after generation.
    
    Examples:
        # Generic domain (technology knowledge graph)
        python kg2sft.py --graph technology_knowledge.graphml --count 50
        
        # Beauty product domain (beauty products knowledge graph)
        python kg2sft.py --graph beauty_products.graphml --count 50 --domain beauty_product
        
        # Beauty Makeup domain (makeup consultation knowledge graph)
        python kg2sft.py --graph makeup_knowledge_graph.graphml --count 50 --domain beauty_makeup
        
        # Auto-tuning mode (automatically tune params to reach target count)
        python kg2sft.py --graph makeup_knowledge_graph.graphml --count 500 --domain beauty_makeup --auto
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate training data from knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to input GraphML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_training",
        help="Output filename prefix (default: output_training)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of examples to generate (default: 10)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="generic",
        choices=["generic", "beauty_product", "beauty_makeup"],
        help="Domain template: 'generic' for general-purpose KG, 'beauty_product' for beauty products (brands/ingredients/skincare), 'beauty_makeup' for makeup consultation (skin types/finishes/techniques) (default: generic)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature 0-1 (default: 0.7)"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.7,
        help="Minimum quality score 0-1 to accept examples (default: 0.7)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=999,
        help="Maximum path depth - paths explore naturally until no unvisited neighbors (default: 999)"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Path similarity threshold 0-1 (default: 0.95)"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="frequency_weighted",
        choices=["frequency_weighted", "random"],
        help="Sampling strategy (default: frequency_weighted)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Enable auto-tuning mode: automatically adjusts quality threshold, "
             "dedup threshold, temperature, and sampling strategy to reach the "
             "target --count. May sacrifice quality for quantity. A quality "
             "impact report is shown after generation."
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("kg2sft: Knowledge Graph to Training Data Generator")
    print("=" * 60)
    
    graph_path = args.graph
    output_jsonl = f"{args.output}.jsonl"
    output_json = f"{args.output}.json"
    
    # Check if graph file exists
    if not os.path.exists(graph_path):
        print(f"\n❌ Error: Graph file not found: {graph_path}")
        print(f"   Please use one of the sample files or provide your own GraphML file:")
        print(f"   - technology_knowledge.graphml (generic domain)")
        print(f"   - beauty_products.graphml (beauty_product domain - products, brands, ingredients)")
        print(f"   - makeup_knowledge_graph.graphml (beauty_makeup domain - skin types, finishes, techniques)")
        print(f"\n   Examples:")
        print(f"   python kg2sft.py --graph technology_knowledge.graphml --count 50")
        print(f"   python kg2sft.py --graph beauty_products.graphml --count 50 --domain beauty_product")
        print(f"   python kg2sft.py --graph makeup_knowledge_graph.graphml --count 50 --domain beauty_makeup")
        return
    
    # Map domain names to human-readable descriptions for logging
    domain_descriptions = {
        "generic": "generic (general-purpose KG)",
        "beauty_product": "beauty_product (products, brands, ingredients, skincare)",
        "beauty_makeup": "beauty_makeup (makeup consultation: skin types, finishes, techniques)"
    }
    
    print(f"\n🔧 Initializing kg2sft...")
    print(f"   Graph: {graph_path}")
    print(f"   Output: {args.output}")
    print(f"   Count: {args.count}")
    print(f"   Domain: {domain_descriptions.get(args.domain, args.domain)}")
    print(f"   Mode: {'AUTO-TUNING' if args.auto else 'MANUAL'}")
    
    # === AUTO-TUNING MODE ===
    if args.auto:
        print("\n⚡ Auto-tuning mode enabled.")
        print("   Parameters will be automatically adjusted to reach the target count.")
        print("   Quality may be traded off — see the report after generation.")
        
        tuner = AutoTuner(
            graph_path=graph_path,
            domain=args.domain,
            target_count=args.count,
            output_prefix=args.output,
            max_depth=args.max_depth,
        )
        
        dataset = tuner.run()
        tuner.print_auto_report(dataset)
        
        print("\n💾 Saving results...")
        dataset.save_jsonl(output_jsonl)
        dataset.save_json(output_json)
        
        print("\n✨ Done! Check output files:")
        print(f"   - {output_jsonl} (SFT format - ready for fine-tuning)")
        print(f"   - {output_json} (Human-readable format)")
        
        print("\n📖 Next steps:")
        print("   1. Review the QUALITY IMPACT ASSESSMENT above")
        print("   2. Review output_training.jsonl for accuracy")
        print("   3. Use for SLM fine-tuning (Hugging Face, LiteLLM, etc.)")
        return
    
    # === MANUAL MODE (default) ===
    trainer = KGToTrainingData(
        graph_path=graph_path,
        
        generation_config=GenerationConfig(
            count=args.count,
            temperature=args.temperature,
            max_tokens=500
        ),
        
        extraction_config=ExtractionConfig(
            max_hop_depth=args.max_depth,
            sampling_strategy=args.sampling,
            dedup_threshold=args.dedup_threshold
        ),
        
        validation_config=ValidationConfig(
            quality_threshold=args.quality_threshold,
            min_length=20,
            max_length=500
        ),
        
        domain=args.domain
    )
    
    print("\n" + "="*60)
    dataset = trainer.generate()
    
    trainer.print_report()
    
    print("\n💾 Saving results...")
    dataset.save_jsonl(output_jsonl)
    dataset.save_json(output_json)
    
    print("\n✨ Done! Check output files:")
    print(f"   - {output_jsonl} (SFT format - ready for fine-tuning)")
    print(f"   - {output_json} (Human-readable format)")
    
    print("\n📖 Next steps:")
    print("   1. Review output_training.jsonl")
    print("   2. Use for SLM fine-tuning (Hugging Face, LiteLLM, etc.)")
    print("   3. Adjust settings for your use case")


if __name__ == "__main__":
    main()
