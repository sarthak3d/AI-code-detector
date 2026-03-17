"""
Generate Feature Dataset for ML Meta-Classifier
================================================
This script generates method×model score features using detect_full.py
for training ML models to improve AI code detection.

Supports 15+ LLMs including:
- OpenAI: gpt-oss-20b
- Mistral: Codestral-22B
- DeepSeek: DeepSeek-Coder-V2-Lite
- Qwen: Qwen2.5-Coder-32B (4-bit)
- Google: gemma-2-27b-it, codegemma-7b
- Meta: Llama-3.3-70B (4-bit), CodeLlama-34b (8-bit)
- Salesforce: codegen models
- BigCode: starcoder models

Usage:
    python generate_features.py --samples 500 --models codegen-350m-multi
    python generate_features.py --samples 500 --models all-small  # All small models
    python generate_features.py --samples 500 --models all-medium  # Medium models
    python generate_features.py --full  # Full 10,000 samples with default model
    
System: 2x Tesla V100-SXM2-32GB (64GB total VRAM)
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torch
import gc

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Extended Model Registry (15+ Models)
EXTENDED_MODELS = {
    # SMALL MODELS (< 2GB VRAM) - Run on single GPU easily
    "codegen-350m-multi": {
        "name": "Salesforce/codegen-350M-multi",
        "description": "CodeGen 350M multi-language",
        "vram": "~700MB",
        "quantization": None,
        "category": "small",
        "has_safetensors": False  # Uses pytorch_model.bin
    },
    
    # MEDIUM MODELS (2-8GB VRAM)
    "codegen-2b-multi": {
        "name": "Salesforce/codegen-2B-multi",
        "description": "CodeGen 2B multi-language",
        "vram": "~4GB",
        "quantization": None,
        "category": "medium"
    },
    "starcoder-1b": {
        "name": "bigcode/starcoderbase-1b",
        "description": "StarCoder 1B base",
        "vram": "~2GB",
        "quantization": None,
        "category": "medium"
    },
    "starcoder2-3b": {
        "name": "bigcode/starcoder2-3b",
        "description": "StarCoder2 3B",
        "vram": "~6GB",
        "quantization": None,
        "category": "medium"
    },
    "deepseek-1.3b": {
        "name": "deepseek-ai/deepseek-coder-1.3b-base",
        "description": "DeepSeek Coder 1.3B",
        "vram": "~3GB",
        "quantization": None,
        "category": "medium"
    },
    "codegemma-2b": {
        "name": "google/codegemma-2b",
        "description": "Google CodeGemma 2B",
        "vram": "~4GB",
        "quantization": None,
        "category": "medium"
    },
    
    # LARGE MODELS (8-16GB VRAM)
    "codegen-6b-multi": {
        "name": "Salesforce/codegen-6B-multi",
        "description": "CodeGen 6B multi-language",
        "vram": "~12GB",
        "quantization": None,
        "category": "large"
    },
    "starcoder-7b": {
        "name": "bigcode/starcoder2-7b",
        "description": "StarCoder2 7B",
        "vram": "~14GB",
        "quantization": None,
        "category": "large"
    },
    "codegemma-7b": {
        "name": "google/codegemma-7b",
        "description": "Google CodeGemma 7B",
        "vram": "~14GB",
        "quantization": None,
        "category": "large"
    },
    "codellama-7b": {
        "name": "codellama/CodeLlama-7b-hf",
        "description": "Meta CodeLlama 7B",
        "vram": "~14GB",
        "quantization": None,
        "category": "large"
    },
    "deepseek-6.7b": {
        "name": "deepseek-ai/deepseek-coder-6.7b-base",
        "description": "DeepSeek Coder 6.7B",
        "vram": "~14GB",
        "quantization": None,
        "category": "large"
    },
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "description": "OpenAI GPT-OSS 20B (MoE, 3.6B active)",
        "vram": "~16GB",
        "quantization": None,
        "category": "large"
    },
    
    # XLARGE MODELS (16-32GB VRAM) - Need 8-bit quantization
    "codellama-13b": {
        "name": "codellama/CodeLlama-13b-hf",
        "description": "Meta CodeLlama 13B",
        "vram": "~26GB (full) / ~14GB (8-bit)",
        "quantization": "8bit",
        "category": "xlarge"
    },
    "starcoder-15b": {
        "name": "bigcode/starcoder2-15b",
        "description": "StarCoder2 15B",
        "vram": "~30GB (full) / ~16GB (8-bit)",
        "quantization": "8bit",
        "category": "xlarge"
    },
    "deepseek-v2-lite": {
        "name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "description": "DeepSeek Coder V2 Lite 16B",
        "vram": "~32GB (full) / ~16GB (8-bit)",
        "quantization": "8bit",
        "category": "xlarge"
    },
    "codestral-22b": {
        "name": "mistralai/Codestral-22B-v0.1",
        "description": "Mistral Codestral 22B",
        "vram": "~44GB (full) / ~22GB (8-bit)",
        "quantization": "8bit",
        "category": "xlarge"
    },
    "gemma-2-27b": {
        "name": "google/gemma-2-27b-it",
        "description": "Google Gemma 2 27B Instruct",
        "vram": "~54GB (full) / ~28GB (8-bit)",
        "quantization": "8bit",
        "category": "xlarge"
    },
    
    # XXLARGE MODELS (>32GB VRAM) - Need 4-bit quantization for V100
    "codellama-34b": {
        "name": "codellama/CodeLlama-34b-Instruct-hf",
        "description": "Meta CodeLlama 34B Instruct",
        "vram": "~68GB (full) / ~18GB (4-bit)",
        "quantization": "4bit",
        "category": "xxlarge"
    },
    "qwen2.5-coder-32b": {
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "description": "Qwen 2.5 Coder 32B Instruct",
        "vram": "~64GB (full) / ~18GB (4-bit)",
        "quantization": "4bit",
        "category": "xxlarge"
    },
    "llama-3.3-70b": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "Meta Llama 3.3 70B Instruct",
        "vram": "~140GB (full) / ~38GB (4-bit)",
        "quantization": "4bit",
        "category": "xxlarge"
    },
}

# Model categories for easy selection
MODEL_CATEGORIES = {
    "all": list(EXTENDED_MODELS.keys()),  # All models
    "all-small": [k for k, v in EXTENDED_MODELS.items() if v["category"] == "small"],
    "all-medium": [k for k, v in EXTENDED_MODELS.items() if v["category"] == "medium"],
    "all-large": [k for k, v in EXTENDED_MODELS.items() if v["category"] == "large"],
    "all-xlarge": [k for k, v in EXTENDED_MODELS.items() if v["category"] == "xlarge"],
    "all-xxlarge": [k for k, v in EXTENDED_MODELS.items() if v["category"] == "xxlarge"],
    "recommended": ["codegen-350m-multi", "codegen-2b-multi", "starcoder-1b", "codegemma-7b", "gpt-oss-20b"],
    "fast": ["codegen-350m-multi", "starcoder2-3b", "deepseek-1.3b"],
}

# Configuration
DATASET_AI_PATH = Path("dataset_ai")
DATASET_HUMAN_PATH = Path("dataset_human")
OUTPUT_DIR = Path("features")

LANGUAGES = ['python', 'javascript', 'java', 'c', 'cpp', 'go', 'rust', 'ruby', 'php', 'c_sharp']
METHODS = ['npr', 'lrr', 'logrank', 'entropy', 'likelihood', 'detectgpt', 't5npr', 'idnpr']

# Language encoding for ML
LANGUAGE_ENCODING = {lang: idx for idx, lang in enumerate(LANGUAGES)}

# Model Loading with Quantization
def load_model_with_quantization(model_name: str, quantization: str = None, device: str = "cuda", use_safetensors: bool = True):
    """Load model with optional quantization for large models.
    
    Uses safetensors by default to avoid CVE-2025-32434 vulnerability.
    Falls back to pytorch if safetensors not available.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"   Loading: {model_name}")
    if quantization:
        print(f"   Quantization: {quantization}")
    
    # Load tokenizer (no device_map for tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    # Try safetensors first to avoid CVE-2025-32434
    if use_safetensors:
        load_kwargs["use_safetensors"] = True
    
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        load_kwargs["quantization_config"] = bnb_config
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = torch.bfloat16
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        if "safetensors" in str(e).lower() or "use_safetensors" in str(e).lower():
            # Fallback: try without safetensors requirement
            print(f"   Safetensors not available, trying with pytorch format...")
            load_kwargs.pop("use_safetensors", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            raise
    
    model.eval()
    
    return model, tokenizer

# Custom Detector for Extended Models
class ExtendedDetector:
    """Detector that supports quantized models."""
    
    def __init__(self, model_alias: str, device: str = "cuda", n_perturbations: int = 5):
        self.model_alias = model_alias
        self.device = device
        self.n_perturbations = n_perturbations
        
        model_info = EXTENDED_MODELS.get(model_alias)
        if not model_info:
            raise ValueError(f"Unknown model: {model_alias}")
        
        self.model_name = model_info["name"]
        self.quantization = model_info.get("quantization")
        
        # Load model
        self.model, self.tokenizer = load_model_with_quantization(
            self.model_name, self.quantization, device
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_log_rank(self, text: str, log: bool = True) -> float:
        """Compute average log-rank of tokens."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get token predictions
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]
            
            # Compute ranks
            ranks = []
            for i in range(shift_labels.shape[1]):
                token_id = shift_labels[0, i].item()
                token_logits = shift_logits[0, i]
                sorted_indices = torch.argsort(token_logits, descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
            
            avg_rank = np.mean(ranks)
            return np.log(avg_rank) if log else avg_rank
        except Exception as e:
            return float('nan')
    
    def get_log_likelihood(self, text: str) -> float:
        """Compute average log-likelihood."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            
            return -loss  # Negative loss = log-likelihood
        except Exception as e:
            return float('nan')
    
    def get_entropy(self, text: str) -> float:
        """Compute average token entropy."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Compute entropy for each token position
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            return entropy.mean().item()
        except Exception as e:
            return float('nan')
    
    def perturb_simple(self, text: str) -> str:
        """Simple space/newline perturbation."""
        import re
        # Insert random spaces
        lines = text.split('\n')
        perturbed_lines = []
        for line in lines:
            if random.random() < 0.3:
                # Insert space at random position
                if len(line) > 0:
                    pos = random.randint(0, len(line))
                    line = line[:pos] + ' ' + line[pos:]
            perturbed_lines.append(line)
        return '\n'.join(perturbed_lines)
    
    def detect_npr(self, code: str) -> dict:
        """NPR detection method."""
        original_logrank = self.get_log_rank(code, log=True)
        
        if np.isnan(original_logrank):
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}
        
        perturbed_logranks = []
        for _ in range(self.n_perturbations):
            perturbed = self.perturb_simple(code)
            p_logrank = self.get_log_rank(perturbed, log=True)
            if not np.isnan(p_logrank):
                perturbed_logranks.append(p_logrank)
        
        if not perturbed_logranks:
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}
        
        avg_perturbed = np.mean(perturbed_logranks)
        npr_score = avg_perturbed / original_logrank if original_logrank != 0 else 1.0
        
        # Normalize to 0-1 (higher = more AI-like)
        normalized_score = max(0, 1 - (npr_score - 1.0) / 1.0)
        normalized_score = min(1, normalized_score)
        
        prediction = 1 if npr_score < 1.40 else 0
        
        return {"score": normalized_score, "prediction": prediction, "raw": npr_score, "error": False}
    
    def detect_lrr(self, code: str) -> dict:
        """LRR detection method."""
        ll = self.get_log_likelihood(code)
        logrank = self.get_log_rank(code, log=True)
        
        if np.isnan(ll) or np.isnan(logrank) or logrank == 0:
            return {"score": 0.0, "prediction": 0, "error": True}
        
        score = -ll / logrank
        prediction = 1 if score > 3.5 else 0
        
        return {"score": score, "prediction": prediction, "error": False}
    
    def detect_logrank(self, code: str) -> dict:
        """LogRank detection method."""
        logrank = self.get_log_rank(code, log=True)
        
        if np.isnan(logrank):
            return {"score": 0.0, "prediction": 0, "error": True}
        
        score = -logrank
        prediction = 1 if score > -1.5 else 0
        
        return {"score": score, "prediction": prediction, "error": False}
    
    def detect_entropy(self, code: str) -> dict:
        """Entropy detection method."""
        entropy = self.get_entropy(code)
        
        if np.isnan(entropy):
            return {"score": 0.0, "prediction": 0, "error": True}
        
        score = -entropy
        prediction = 1 if score > -3.0 else 0
        
        return {"score": score, "prediction": prediction, "error": False}
    
    def detect_likelihood(self, code: str) -> dict:
        """Likelihood detection method."""
        ll = self.get_log_likelihood(code)
        
        if np.isnan(ll):
            return {"score": 0.0, "prediction": 0, "error": True}
        
        prediction = 1 if ll > -2.0 else 0
        
        return {"score": ll, "prediction": prediction, "error": False}
    
    def detect_detectgpt(self, code: str) -> dict:
        """DetectGPT score (curvature-based detection).
        
        DetectGPT uses the curvature of the log-probability function:
        score = (original_logprob - mean_perturbed_logprob) / std_perturbed_logprob
        
        Higher curvature = more likely AI-generated.
        """
        original_ll = self.get_log_likelihood(code)
        
        if np.isnan(original_ll):
            return {"score": 0.0, "prediction": 0, "curvature": 0.0, "error": True}
        
        perturbed_lls = []
        for _ in range(self.n_perturbations):
            perturbed = self.perturb_simple(code)
            p_ll = self.get_log_likelihood(perturbed)
            if not np.isnan(p_ll):
                perturbed_lls.append(p_ll)
        
        if len(perturbed_lls) < 2:
            return {"score": 0.0, "prediction": 0, "curvature": 0.0, "error": True}
        
        mean_perturbed = np.mean(perturbed_lls)
        std_perturbed = np.std(perturbed_lls)
        
        # Curvature score (DetectGPT formula)
        if std_perturbed > 1e-6:
            curvature = (original_ll - mean_perturbed) / std_perturbed
        else:
            curvature = original_ll - mean_perturbed
        
        # Higher curvature = more likely AI
        prediction = 1 if curvature > 1.5 else 0
        
        return {"score": curvature, "prediction": prediction, "curvature": curvature, "error": False}
    
    def perturb_t5_mask(self, text: str, mask_ratio: float = 0.15) -> str:
        """T5 Mask-Fill perturbation.
        
        Masks random spans and uses T5 to fill them.
        Falls back to simple perturbation if T5 not available.
        """
        try:
            # Try to load T5 model (only once)
            if not hasattr(self, 't5_model'):
                try:
                    from transformers import T5ForConditionalGeneration, T5Tokenizer
                    self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                    self.t5_model = T5ForConditionalGeneration.from_pretrained(
                        "t5-small", 
                        torch_dtype=torch.float16
                    ).to(self.model.device)
                    self.t5_model.eval()
                except Exception:
                    self.t5_model = None
                    self.t5_tokenizer = None
            
            if self.t5_model is None:
                # Fallback to simple perturbation
                return self.perturb_simple(text)
            
            # Tokenize
            tokens = text.split()
            if len(tokens) < 5:
                return self.perturb_simple(text)
            
            # Mask random spans
            n_masks = max(1, int(len(tokens) * mask_ratio))
            mask_positions = random.sample(range(len(tokens)), min(n_masks, len(tokens)))
            
            for pos in mask_positions:
                tokens[pos] = "<extra_id_0>"
            
            masked_text = " ".join(tokens)
            
            # Generate fills
            inputs = self.t5_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.t5_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.t5_model.generate(**inputs, max_length=50)
            
            filled = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Replace masks with filled content
            result = masked_text.replace("<extra_id_0>", filled.split()[0] if filled else "")
            
            return result if result.strip() else self.perturb_simple(text)
            
        except Exception:
            return self.perturb_simple(text)
    
    def detect_t5_npr(self, code: str) -> dict:
        """NPR using T5 Mask-Fill perturbation instead of simple perturbation."""
        original_logrank = self.get_log_rank(code, log=True)
        
        if np.isnan(original_logrank):
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}
        
        perturbed_logranks = []
        for _ in range(self.n_perturbations):
            perturbed = self.perturb_t5_mask(code)
            p_logrank = self.get_log_rank(perturbed, log=True)
            if not np.isnan(p_logrank):
                perturbed_logranks.append(p_logrank)
        
        if not perturbed_logranks:
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}
        
        avg_perturbed = np.mean(perturbed_logranks)
        npr_score = avg_perturbed / original_logrank if original_logrank != 0 else 1.0
        
        normalized_score = max(0, min(1, 1 - (npr_score - 1.0) / 1.0))
        prediction = 1 if npr_score < 1.40 else 0
        
        return {"score": normalized_score, "prediction": prediction, "raw": npr_score, "error": False}
    
    def perturb_identifiers(self, code: str, language: str = 'python') -> str:
        """Identifier masking perturbation using tree-sitter.
        
        Parses code and replaces variable/function names with generic names.
        Falls back to regex-based replacement if tree-sitter not available.
        """
        try:
            # Try tree-sitter first
            try:
                import tree_sitter_python as tspython
                from tree_sitter import Language, Parser
                
                if not hasattr(self, 'ts_parser'):
                    self.ts_parser = Parser()
                    # Only Python supported for now
                    if language == 'python':
                        PY_LANGUAGE = Language(tspython.language())
                        self.ts_parser.language = PY_LANGUAGE
                    else:
                        self.ts_parser = None
                
                if self.ts_parser and language == 'python':
                    tree = self.ts_parser.parse(bytes(code, 'utf8'))
                    
                    # Find all identifiers
                    identifiers = []
                    def visit(node):
                        if node.type == 'identifier':
                            name = code[node.start_byte:node.end_byte]
                            # Skip common keywords and builtins
                            if name not in ['self', 'cls', 'print', 'len', 'range', 'True', 'False', 'None']:
                                identifiers.append((node.start_byte, node.end_byte, name))
                        for child in node.children:
                            visit(child)
                    
                    visit(tree.root_node)
                    
                    # Replace random subset of identifiers
                    if identifiers:
                        n_replace = max(1, len(identifiers) // 3)
                        to_replace = random.sample(identifiers, min(n_replace, len(identifiers)))
                        
                        # Sort by position (reverse) to replace from end
                        to_replace.sort(key=lambda x: x[0], reverse=True)
                        
                        result = code
                        for start, end, name in to_replace:
                            new_name = f"var_{random.randint(0, 999)}"
                            result = result[:start] + new_name + result[end:]
                        
                        return result
                        
            except ImportError:
                pass
            
            # Fallback: regex-based identifier replacement
            import re
            
            # Find potential identifiers (simple pattern)
            if language == 'python':
                pattern = r'\b([a-z_][a-z0-9_]*)\b'
            else:
                pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            
            # Skip common keywords
            keywords = {'if', 'else', 'for', 'while', 'def', 'class', 'return', 'import', 
                       'from', 'as', 'try', 'except', 'finally', 'with', 'in', 'not', 
                       'and', 'or', 'is', 'None', 'True', 'False', 'self', 'print'}
            
            matches = list(re.finditer(pattern, code))
            valid_matches = [m for m in matches if m.group(1) not in keywords]
            
            if valid_matches:
                n_replace = max(1, len(valid_matches) // 4)
                to_replace = random.sample(valid_matches, min(n_replace, len(valid_matches)))
                
                # Sort by position (reverse)
                to_replace.sort(key=lambda x: x.start(), reverse=True)
                
                result = code
                for match in to_replace:
                    new_name = f"var_{random.randint(0, 999)}"
                    result = result[:match.start()] + new_name + result[match.end():]
                
                return result
            
            return code
            
        except Exception:
            return code
    
    def detect_identifier_npr(self, code: str, language: str = 'python') -> dict:
        """NPR using Identifier Masking perturbation."""
        original_logrank = self.get_log_rank(code, log=True)
        
        if np.isnan(original_logrank):
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}
        
        perturbed_logranks = []
        for _ in range(self.n_perturbations):
            perturbed = self.perturb_identifiers(code, language)
            p_logrank = self.get_log_rank(perturbed, log=True)
            if not np.isnan(p_logrank):
                perturbed_logranks.append(p_logrank)
        
        if not perturbed_logranks:
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}
        
        avg_perturbed = np.mean(perturbed_logranks)
        npr_score = avg_perturbed / original_logrank if original_logrank != 0 else 1.0
        
        normalized_score = max(0, min(1, 1 - (npr_score - 1.0) / 1.0))
        prediction = 1 if npr_score < 1.40 else 0
        
        return {"score": normalized_score, "prediction": prediction, "raw": npr_score, "error": False}
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        # Clean up T5 if loaded
        if hasattr(self, 't5_model') and self.t5_model is not None:
            del self.t5_model
            del self.t5_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

# Data Loading Functions
def load_ai_samples(language: str, n_samples: int = 500) -> list:
    """Load AI-generated code samples for a language."""
    samples = []
    file_path = DATASET_AI_PATH / language / "outputs.txt"
    
    if not file_path.exists():
        print(f"Warning: AI dataset not found for {language}: {file_path}")
        return samples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(samples) >= n_samples:
                break
            try:
                data = json.loads(line.strip())
                if 'output' in data and data['output'].strip():
                    samples.append({
                        'code': data['output'],
                        'language': language,
                        'language_encoded': LANGUAGE_ENCODING[language],
                        'label': 1,  # AI = 1
                        'source': 'gpt-oss-20b'
                    })
            except json.JSONDecodeError:
                continue
    
    return samples

def load_human_samples(language: str, n_samples: int = 500) -> list:
    """Load human-written code samples for a language (random sample)."""
    samples = []
    file_path = DATASET_HUMAN_PATH / language / "small_train.jsonl"
    
    if not file_path.exists():
        print(f"Warning: Human dataset not found for {language}: {file_path}")
        return samples
    
    # First, count total lines
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Random sample indices
    if total_lines <= n_samples:
        sample_indices = set(range(total_lines))
    else:
        sample_indices = set(random.sample(range(total_lines), n_samples))
    
    # Load selected samples
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in sample_indices:
                try:
                    data = json.loads(line.strip())
                    code = data.get('code', data.get('content', ''))
                    if code and code.strip():
                        samples.append({
                            'code': code,
                            'language': language,
                            'language_encoded': LANGUAGE_ENCODING[language],
                            'label': 0,  # Human = 0
                            'source': 'human'
                        })
                except json.JSONDecodeError:
                    continue
            
            if len(samples) >= n_samples:
                break
    
    return samples

def load_balanced_dataset(samples_per_language: int = 500, languages: list = None) -> list:
    """Load balanced dataset with equal AI and Human samples per language."""
    if languages is None:
        languages = LANGUAGES
    
    all_samples = []
    
    print(f"\nLoading balanced dataset ({samples_per_language} samples per class per language)...")
    
    for lang in languages:
        print(f"\n{lang}:")
        
        # Load AI samples
        ai_samples = load_ai_samples(lang, samples_per_language)
        print(f"  AI samples:    {len(ai_samples)}")
        
        # Load Human samples (random subset)
        human_samples = load_human_samples(lang, samples_per_language)
        print(f"  Human samples: {len(human_samples)}")
        
        all_samples.extend(ai_samples)
        all_samples.extend(human_samples)
    
    print(f"\n{'=' * 60}")
    print(f"Total samples loaded: {len(all_samples)}")
    print(f"  AI:    {sum(1 for s in all_samples if s['label'] == 1)}")
    print(f"  Human: {sum(1 for s in all_samples if s['label'] == 0)}")
    
    return all_samples

# Feature Generation
def extract_features_extended(detector: ExtendedDetector, code: str, model_alias: str, 
                               language: str = 'python') -> dict:
    """Extract all method scores for a single code sample.
    
    Methods extracted:
    1. NPR (Space/Newline perturbation) - DetectCodeGPT main method
    2. LRR (Log-Likelihood Ratio)
    3. LogRank
    4. Entropy
    5. Likelihood
    6. DetectGPT (Curvature-based)
    7. T5-NPR (T5 Mask-Fill perturbation)
    8. Identifier-NPR (Identifier masking perturbation)
    """
    features = {}
    
    # 1. NPR (Space/Newline perturbation) - Main DetectCodeGPT method
    try:
        npr = detector.detect_npr(code)
        features[f'{model_alias}_npr_score'] = npr['score']
        features[f'{model_alias}_npr_prediction'] = npr['prediction']
        features[f'{model_alias}_npr_raw'] = npr['raw']
    except Exception:
        features[f'{model_alias}_npr_score'] = 0.5
        features[f'{model_alias}_npr_prediction'] = 0
        features[f'{model_alias}_npr_raw'] = 1.0
    
    # 2. LRR (Log-Likelihood Ratio)
    try:
        lrr = detector.detect_lrr(code)
        features[f'{model_alias}_lrr_score'] = lrr['score']
        features[f'{model_alias}_lrr_prediction'] = lrr['prediction']
    except Exception:
        features[f'{model_alias}_lrr_score'] = 0.0
        features[f'{model_alias}_lrr_prediction'] = 0
    
    # 3. LogRank
    try:
        logrank = detector.detect_logrank(code)
        features[f'{model_alias}_logrank_score'] = logrank['score']
        features[f'{model_alias}_logrank_prediction'] = logrank['prediction']
    except Exception:
        features[f'{model_alias}_logrank_score'] = 0.0
        features[f'{model_alias}_logrank_prediction'] = 0
    
    # 4. Entropy
    try:
        entropy = detector.detect_entropy(code)
        features[f'{model_alias}_entropy_score'] = entropy['score']
        features[f'{model_alias}_entropy_prediction'] = entropy['prediction']
    except Exception:
        features[f'{model_alias}_entropy_score'] = 0.0
        features[f'{model_alias}_entropy_prediction'] = 0
    
    # 5. Likelihood
    try:
        ll = detector.detect_likelihood(code)
        features[f'{model_alias}_likelihood_score'] = ll['score']
        features[f'{model_alias}_likelihood_prediction'] = ll['prediction']
    except Exception:
        features[f'{model_alias}_likelihood_score'] = 0.0
        features[f'{model_alias}_likelihood_prediction'] = 0
    
    # Advanced methods
    # 6. DetectGPT (Curvature-based)
    try:
        detectgpt = detector.detect_detectgpt(code)
        features[f'{model_alias}_detectgpt_score'] = detectgpt['score']
        features[f'{model_alias}_detectgpt_prediction'] = detectgpt['prediction']
        features[f'{model_alias}_detectgpt_curvature'] = detectgpt['curvature']
    except Exception:
        features[f'{model_alias}_detectgpt_score'] = 0.0
        features[f'{model_alias}_detectgpt_prediction'] = 0
        features[f'{model_alias}_detectgpt_curvature'] = 0.0
    
    # 7. T5-NPR (T5 Mask-Fill perturbation)
    try:
        t5_npr = detector.detect_t5_npr(code)
        features[f'{model_alias}_t5npr_score'] = t5_npr['score']
        features[f'{model_alias}_t5npr_prediction'] = t5_npr['prediction']
        features[f'{model_alias}_t5npr_raw'] = t5_npr['raw']
    except Exception:
        features[f'{model_alias}_t5npr_score'] = 0.5
        features[f'{model_alias}_t5npr_prediction'] = 0
        features[f'{model_alias}_t5npr_raw'] = 1.0
    
    # 8. Identifier-NPR (Identifier masking perturbation)
    try:
        id_npr = detector.detect_identifier_npr(code, language)
        features[f'{model_alias}_idnpr_score'] = id_npr['score']
        features[f'{model_alias}_idnpr_prediction'] = id_npr['prediction']
        features[f'{model_alias}_idnpr_raw'] = id_npr['raw']
    except Exception:
        features[f'{model_alias}_idnpr_score'] = 0.5
        features[f'{model_alias}_idnpr_prediction'] = 0
        features[f'{model_alias}_idnpr_raw'] = 1.0
    
    return features

def generate_feature_dataset(samples: list, model_aliases: list, 
                              n_perturbations: int = 5,
                              device: str = "cuda") -> pd.DataFrame:
    """Generate feature dataset for all samples across all models.
    
    Args:
        samples: List of code samples with metadata
        model_aliases: List of model aliases to use
        n_perturbations: Number of perturbations for NPR methods
        device: Device to run on
    """
    
    # Define all possible methods for placeholder features
    all_methods = ['npr', 'lrr', 'logrank', 'entropy', 'likelihood', 'detectgpt', 't5npr', 'idnpr']
    
    # Initialize feature list with sample metadata
    all_features = []
    for idx, sample in enumerate(samples):
        feature_row = {
            'sample_id': idx,
            'language': sample['language'],
            'language_encoded': sample['language_encoded'],
            'label': sample['label'],
            'source': sample['source']
        }
        all_features.append(feature_row)
    
    # Process each model
    for model_alias in model_aliases:
        print(f"\n{'=' * 60}")
        print(f"Processing with model: {model_alias}")
        model_info = EXTENDED_MODELS.get(model_alias, {})
        print(f"  Name: {model_info.get('name', model_alias)}")
        print(f"  VRAM: {model_info.get('vram', 'Unknown')}")
        print(f"  Quantization: {model_info.get('quantization', 'None')}")
        print(f"{'=' * 60}")
        
        try:
            detector = ExtendedDetector(model_alias, device, n_perturbations)
        except Exception as e:
            print(f"Error loading model {model_alias}: {e}")
            # Add placeholder features
            for idx in range(len(samples)):
                for method in all_methods:
                    all_features[idx][f'{model_alias}_{method}_score'] = 0.5
                    all_features[idx][f'{model_alias}_{method}_prediction'] = 0
            continue
        
        # Process each sample
        for idx, sample in enumerate(tqdm(samples, desc=f"Extracting features ({model_alias})")):
            try:
                model_features = extract_features_extended(
                    detector, 
                    sample['code'], 
                    model_alias,
                    language=sample['language']
                )
                all_features[idx].update(model_features)
            except Exception as e:
                # Add default features on error
                for method in all_methods:
                    all_features[idx][f'{model_alias}_{method}_score'] = 0.5
                    all_features[idx][f'{model_alias}_{method}_prediction'] = 0
        
        # Clean up GPU memory
        detector.cleanup()
    
    return pd.DataFrame(all_features)

# Main
def list_available_models():
    """Print all available models."""
    print("AVAILABLE MODELS")
    
    for category in ["small", "medium", "large", "xlarge", "xxlarge"]:
        print(f"\n{category.upper()} MODELS:")
        for alias, info in EXTENDED_MODELS.items():
            if info["category"] == category:
                quant = f" ({info['quantization']})" if info.get('quantization') else ""
                print(f"  {alias:<25} {info['vram']:<20} {quant}")
    print("MODEL GROUPS:")
    for group, models in MODEL_CATEGORIES.items():
        print(f"  {group}: {', '.join(models)}")

def main():
    parser = argparse.ArgumentParser(description="Generate ML features from extended model set")
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples per class per language (default: 500)')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model aliases or category (default: all)')
    parser.add_argument('--languages', type=str, default=None,
                        help='Comma-separated languages (default: all)')
    parser.add_argument('--perturbations', type=int, default=5,
                        help='Number of perturbations for NPR (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--full', action='store_true',
                        help='Generate full 10,000 sample dataset')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Parse model aliases
    if args.models in MODEL_CATEGORIES:
        model_aliases = MODEL_CATEGORIES[args.models]
    else:
        model_aliases = [m.strip() for m in args.models.split(',')]
    
    # Validate models
    for alias in model_aliases:
        if alias not in EXTENDED_MODELS:
            print(f"Warning: Unknown model '{alias}'. Use --list-models to see available models.")
    
    model_aliases = [a for a in model_aliases if a in EXTENDED_MODELS]
    
    if not model_aliases:
        print("Error: No valid models specified!")
        return
    
    # Parse arguments
    if args.full:
        args.samples = 500
        args.languages = None  # All languages
    
    languages = args.languages.split(',') if args.languages else LANGUAGES
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_models = len(model_aliases)
        n_langs = len(languages)
        total_samples = args.samples * 2 * n_langs
        output_path = OUTPUT_DIR / f"features_{total_samples}samples_{n_models}models_{timestamp}.csv"
    print("FEATURE GENERATION FOR ML META-CLASSIFIER")
    print(f"\nConfiguration:")
    print(f"  Samples per class per language: {args.samples}")
    print(f"  Languages: {languages}")
    print(f"  Models ({len(model_aliases)}): {model_aliases}")
    print(f"  Perturbations: {args.perturbations}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_path}")
    
    # Show VRAM requirements
    print(f"\nVRAM Requirements:")
    total_vram = 0
    for alias in model_aliases:
        info = EXTENDED_MODELS[alias]
        print(f"  {alias}: {info['vram']}")
    
    # Load balanced dataset
    samples = load_balanced_dataset(args.samples, languages)
    
    if not samples:
        print("Error: No samples loaded!")
        return
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Generate features
    print("GENERATING FEATURES")
    
    df = generate_feature_dataset(
        samples, 
        model_aliases, 
        n_perturbations=args.perturbations,
        device=args.device
    )
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Features saved to: {output_path}")
    
    # Print summary
    print("DATASET SUMMARY")
    print(f"Total samples: {len(df)}")
    print(f"Features per sample: {len(df.columns) - 5}")  # Exclude metadata columns
    print(f"AI samples: {(df['label'] == 1).sum()}")
    print(f"Human samples: {(df['label'] == 0).sum()}")
    
    print(f"\nLanguage distribution:")
    for lang in df['language'].unique():
        count = len(df[df['language'] == lang])
        ai = len(df[(df['language'] == lang) & (df['label'] == 1)])
        human = len(df[(df['language'] == lang) & (df['label'] == 0)])
        print(f"  {lang}: {count} (AI: {ai}, Human: {human})")
    
    print(f"\nFeature columns:")
    feature_cols = [c for c in df.columns if c not in ['sample_id', 'language', 'language_encoded', 'label', 'source']]
    for col in feature_cols[:15]:
        print(f"  - {col}")
    if len(feature_cols) > 15:
        print(f"  ... and {len(feature_cols) - 15} more")
    
    print(f"\n✓ Feature generation complete!")
    print(f"\nNote: 'language_encoded' column can be used as a feature for ML training.")
    print("Including language helps the model learn language-specific patterns.")

if __name__ == "__main__":
    main()
