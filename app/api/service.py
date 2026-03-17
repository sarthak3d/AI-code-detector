
from __future__ import annotations

import asyncio
import gc
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
}

LANGUAGES = [
    "python",
    "javascript",
    "java",
    "c",
    "cpp",
    "go",
    "rust",
    "ruby",
    "php",
    "c_sharp",
]
LANGUAGE_ENCODING = {lang: idx for idx, lang in enumerate(LANGUAGES)}

FEATURE_MODELS = {
    "starcoder2-3b": "bigcode/starcoder2-3b",
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
}
MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
LEGACY_MODEL_DIR = Path("dl_models")


def _load_metrics_file() -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    for metrics_path in (MODEL_DIR / "metrics.json", LEGACY_MODEL_DIR / "metrics.json"):
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                return json.load(f), metrics_path
    return None, None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_language(language: Optional[str], filename: Optional[str] = None) -> str:
    if language:
        value = language.strip().lower().replace("-", "_").replace(" ", "_")
        if value in LANGUAGE_ENCODING:
            return value
        if value == "typescript":
            return "javascript"
        if value in {"c#", "csharp"}:
            return "c_sharp"
        if value in {"c++"}:
            return "cpp"

    if filename:
        ext = Path(filename).suffix.lower()
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]

    return "python"


class CodeDetectorMLP(nn.Module):
    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        activation_fn = self.ACTIVATIONS.get(activation.lower(), nn.ReLU)
        self.input_bn = nn.BatchNorm1d(input_size) if use_batch_norm else nn.Identity()

        layers: List[nn.Module] = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            drop_rate = dropout * (1 - 0.2 * i / max(len(hidden_sizes), 1))
            layers.append(nn.Dropout(drop_rate))
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.hidden(x)
        return self.output(x)


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class ANNClassifier:
    def __init__(self, model_path: Optional[str] = None, threshold: Optional[float] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._resolve_model_path(model_path)
        model_data = _torch_load(self.model_path, self.device)

        config = model_data["model_config"]
        self.model = CodeDetectorMLP(
            input_size=config["input_size"],
            hidden_sizes=config["hidden_sizes"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            activation=config["activation"],
            use_batch_norm=config["use_batch_norm"],
        ).to(self.device)
        self.model.load_state_dict(model_data["model_state_dict"])
        self.model.eval()

        self.scaler = model_data["scaler"]
        self.feature_names: List[str] = list(model_data["feature_names"])
        self.threshold = threshold if threshold is not None else self._resolve_threshold(self.model_path)

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Path:
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"MLP model path does not exist: {path}")
            return path

        metrics, metrics_path = _load_metrics_file()
        if metrics:
            reported = metrics.get("model_info", {}).get("model_path")
            if reported:
                path = Path(reported)
                if path.exists():
                    return path
                if metrics_path is not None:
                    from_metrics_dir = metrics_path.parent / path.name
                    if from_metrics_dir.exists():
                        return from_metrics_dir

        for directory in (MODEL_DIR, LEGACY_MODEL_DIR):
            candidates = sorted(directory.glob("mlp_model_*.pkl"))
            if candidates:
                return candidates[-1]
        raise FileNotFoundError("No MLP model found in model/ or dl_models/")

    @staticmethod
    def _resolve_threshold(model_path: Path) -> float:
        env_threshold = os.getenv("MLP_THRESHOLD")
        if env_threshold:
            return float(env_threshold)

        metrics, _ = _load_metrics_file()
        if not metrics:
            return 0.5

        reported_path = metrics.get("model_info", {}).get("model_path")
        reported_threshold = metrics.get("model_info", {}).get("optimal_threshold")

        if reported_threshold is None:
            return 0.5
        if not reported_path:
            return float(reported_threshold)
        if Path(reported_path).name == model_path.name:
            return float(reported_threshold)
        return 0.5

    @staticmethod
    def _default_feature_value(feature_name: str) -> float:
        if feature_name == "language_encoded":
            return 0.0
        if feature_name.endswith("_prediction"):
            return 0.0
        if feature_name.endswith("_raw"):
            return 1.0
        if feature_name.endswith("_score"):
            if feature_name.endswith("_npr_score") or feature_name.endswith("_t5npr_score") or feature_name.endswith("_idnpr_score"):
                return 0.5
            return 0.0
        return 0.0

    def predict_sync(self, features: Dict[str, float], language: str, detectors_used: List[str]) -> Dict[str, Any]:
        vector: List[float] = []
        missing: List[str] = []
        for feature_name in self.feature_names:
            if feature_name == "language_encoded":
                vector.append(float(LANGUAGE_ENCODING.get(language, 0)))
                continue
            if feature_name in features:
                vector.append(float(features[feature_name]))
            else:
                vector.append(self._default_feature_value(feature_name))
                missing.append(feature_name)

        X = np.array([vector], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        ai_probability = float(probs[1])
        human_probability = float(probs[0])
        prediction = int(ai_probability >= self.threshold)
        confidence = max(ai_probability, human_probability)

        if confidence >= 0.8:
            confidence_level = "HIGH"
        elif confidence >= 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "prediction": "AI-GENERATED" if prediction == 1 else "HUMAN-WRITTEN",
            "confidence": confidence_level,
            "confidence_score": confidence,
            "ai_probability": ai_probability,
            "human_probability": human_probability,
            "threshold": self.threshold,
            "language": language,
            "ann_model_path": str(self.model_path),
            "detectors_used": detectors_used,
            "features_extracted": len(features),
            "features_required": len(self.feature_names),
            "missing_feature_count": len(missing),
            "missing_features_preview": missing[:10],
        }

    async def predict(self, features: Dict[str, float], language: str, detectors_used: List[str]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.predict_sync, features, language, detectors_used)


class DetectorModel:
    _t5_model: Optional[T5ForConditionalGeneration] = None
    _t5_tokenizer: Optional[T5Tokenizer] = None
    _t5_lock = asyncio.Lock()

    def __init__(
        self,
        alias: str,
        engine: AsyncLLMEngine,
        tokenizer: AutoTokenizer,
        n_perturbations: int = 5,
        max_length: int = 512,
        prompt_logprob_topk: int = 20,
    ) -> None:
        self.alias = alias
        self.model_name = FEATURE_MODELS[alias]
        self.engine = engine
        self.tokenizer = tokenizer
        self.n_perturbations = n_perturbations
        self.max_length = max_length
        self.prompt_logprob_topk = max(prompt_logprob_topk, 1)
        self.local_files_only = _bool_env("HF_LOCAL_FILES_ONLY", False)
        self._engine_lock = asyncio.Lock()
        self._stats_cache: Dict[Tuple[str, int], Dict[str, float]] = {}

    @classmethod
    async def create(
        cls,
        alias: str,
        n_perturbations: int = 5,
        max_length: int = 512,
        prompt_logprob_topk: int = 20,
    ) -> "DetectorModel":
        if alias not in FEATURE_MODELS:
            raise ValueError(f"Unsupported detector alias: {alias}")

        model_name = FEATURE_MODELS[alias]
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            model_name,
            trust_remote_code=True,
            local_files_only=_bool_env("HF_LOCAL_FILES_ONLY", False),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = os.getenv("VLLM_DTYPE", "float16" if torch.cuda.is_available() else "float32")
        tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", str(max_length)))
        trust_remote_code = _bool_env("VLLM_TRUST_REMOTE_CODE", True)
        download_dir = os.getenv("HF_HOME")

        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            download_dir=download_dir,
            disable_log_requests=True,
            enforce_eager=_bool_env("VLLM_ENFORCE_EAGER", False),
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return cls(alias, engine, tokenizer, n_perturbations, max_length, prompt_logprob_topk)

    @staticmethod
    def _extract_logprob_rank(entry: Any) -> Tuple[Optional[float], Optional[int]]:
        if entry is None:
            return None, None
        if isinstance(entry, (int, float)):
            return float(entry), None
        if isinstance(entry, dict):
            lp = entry.get("logprob")
            rank = entry.get("rank")
            return (float(lp) if lp is not None else None), (int(rank) if rank is not None else None)

        lp = getattr(entry, "logprob", None)
        rank = getattr(entry, "rank", None)
        return (float(lp) if lp is not None else None), (int(rank) if rank is not None else None)

    @staticmethod
    def _approx_entropy(log_probs: List[float]) -> float:
        if not log_probs:
            return float("nan")
        arr = np.array(log_probs, dtype=np.float64)
        m = float(np.max(arr))
        probs = np.exp(arr - m)
        probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    async def _score_prompt(self, text: str, topk: Optional[int] = None) -> Dict[str, float]:
        k = topk or self.prompt_logprob_topk
        cache_key = (text, k)
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=k,
            logprobs=1,
            detokenize=False,
        )

        request_id = f"{self.alias}-{uuid.uuid4().hex}"
        final_output = None
        async with self._engine_lock:
            stream = self.engine.generate(text, sampling_params, request_id=request_id)
            async for output in stream:
                final_output = output

        if final_output is None:
            stats = {"log_likelihood": float("nan"), "log_rank": float("nan"), "entropy": float("nan")}
            self._stats_cache[cache_key] = stats
            return stats

        prompt_token_ids = getattr(final_output, "prompt_token_ids", None) or []
        prompt_logprobs = getattr(final_output, "prompt_logprobs", None) or []
        n = min(len(prompt_token_ids), len(prompt_logprobs))
        if n <= 1:
            stats = {"log_likelihood": float("nan"), "log_rank": float("nan"), "entropy": float("nan")}
            self._stats_cache[cache_key] = stats
            return stats

        actual_logprobs: List[float] = []
        actual_ranks: List[int] = []
        entropies: List[float] = []

        for idx in range(1, n):
            actual_token_id = int(prompt_token_ids[idx])
            token_candidates = prompt_logprobs[idx]
            if not token_candidates:
                continue

            selected = token_candidates.get(actual_token_id)
            if selected is None:
                selected = token_candidates.get(str(actual_token_id))
            lp, rank = self._extract_logprob_rank(selected)
            if lp is not None and np.isfinite(lp):
                actual_logprobs.append(float(lp))
            if rank is not None and rank > 0:
                actual_ranks.append(int(rank))

            candidate_lps: List[float] = []
            for candidate in token_candidates.values():
                cand_lp, _ = self._extract_logprob_rank(candidate)
                if cand_lp is not None and np.isfinite(cand_lp):
                    candidate_lps.append(float(cand_lp))
            if candidate_lps:
                entropies.append(self._approx_entropy(candidate_lps))

        ll = float(np.mean(actual_logprobs)) if actual_logprobs else float("nan")
        avg_rank = float(np.mean(actual_ranks)) if actual_ranks else float("nan")
        log_rank = float(np.log(avg_rank)) if np.isfinite(avg_rank) and avg_rank > 0 else float("nan")
        entropy = float(np.mean(entropies)) if entropies else float("nan")

        stats = {"log_likelihood": ll, "log_rank": log_rank, "entropy": entropy}
        if len(self._stats_cache) > 2000:
            self._stats_cache.clear()
        self._stats_cache[cache_key] = stats
        return stats

    async def get_log_rank(self, text: str, log: bool = True) -> float:
        stats = await self._score_prompt(text)
        value = stats["log_rank"]
        if np.isnan(value):
            return float("nan")
        return value if log else float(np.exp(value))

    async def get_log_likelihood(self, text: str) -> float:
        return (await self._score_prompt(text))["log_likelihood"]

    async def get_entropy(self, text: str) -> float:
        return (await self._score_prompt(text))["entropy"]

    @staticmethod
    def perturb_simple(text: str) -> str:
        lines = text.split("\n")
        perturbed = []
        for line in lines:
            if random.random() < 0.3 and line:
                pos = random.randint(0, len(line))
                line = line[:pos] + " " + line[pos:]
            perturbed.append(line)
        return "\n".join(perturbed)

    @classmethod
    async def _load_t5(cls, local_files_only: bool) -> None:
        if cls._t5_model is not None and cls._t5_tokenizer is not None:
            return
        async with cls._t5_lock:
            if cls._t5_model is not None and cls._t5_tokenizer is not None:
                return
            cls._t5_tokenizer = await asyncio.to_thread(
                T5Tokenizer.from_pretrained,
                "t5-small",
                local_files_only=local_files_only,
            )
            cls._t5_model = await asyncio.to_thread(
                T5ForConditionalGeneration.from_pretrained,
                "t5-small",
                local_files_only=local_files_only,
            )
            cls._t5_model.to("cpu")
            cls._t5_model.eval()

    @staticmethod
    def _perturb_identifiers_sync(code: str, language: str = "python") -> str:
        try:
            import re

            if language == "python":
                pattern = r"\b([a-z_][a-z0-9_]*)\b"
            else:
                pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"

            keywords = {
                "if", "else", "for", "while", "def", "class", "return", "import", "from", "as", "try", "except", "finally",
                "with", "in", "not", "and", "or", "is", "None", "True", "False", "self", "print",
            }

            matches = list(re.finditer(pattern, code))
            valid = [m for m in matches if m.group(1) not in keywords]
            if not valid:
                return code

            n_replace = max(1, len(valid) // 4)
            selected = random.sample(valid, min(n_replace, len(valid)))
            selected.sort(key=lambda x: x.start(), reverse=True)

            result = code
            for m in selected:
                new_name = f"var_{random.randint(0, 999)}"
                result = result[: m.start()] + new_name + result[m.end() :]
            return result
        except Exception:
            return code

    async def perturb_identifiers(self, code: str, language: str = "python") -> str:
        return await asyncio.to_thread(self._perturb_identifiers_sync, code, language)

    def _perturb_t5_mask_sync(self, text: str, mask_ratio: float = 0.15) -> str:
        try:
            if self._t5_model is None or self._t5_tokenizer is None:
                return self.perturb_simple(text)

            tokens = text.split()
            if len(tokens) < 5:
                return self.perturb_simple(text)

            n_masks = max(1, int(len(tokens) * mask_ratio))
            mask_positions = random.sample(range(len(tokens)), min(n_masks, len(tokens)))
            for pos in mask_positions:
                tokens[pos] = "<extra_id_0>"
            masked_text = " ".join(tokens)

            inputs = self._t5_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self._t5_model.generate(**inputs, max_length=50)
            filled = self._t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            replacement = filled.split()[0] if filled else ""
            result = masked_text.replace("<extra_id_0>", replacement)
            return result if result.strip() else self.perturb_simple(text)
        except Exception:
            return self.perturb_simple(text)

    async def perturb_t5_mask(self, text: str, mask_ratio: float = 0.15) -> str:
        await self._load_t5(self.local_files_only)
        return await asyncio.to_thread(self._perturb_t5_mask_sync, text, mask_ratio)

    async def detect_npr(self, code: str) -> Dict[str, Any]:
        original_logrank = await self.get_log_rank(code, log=True)
        if np.isnan(original_logrank):
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}

        perturbed_logranks = []
        for _ in range(self.n_perturbations):
            p_logrank = await self.get_log_rank(self.perturb_simple(code), log=True)
            if not np.isnan(p_logrank):
                perturbed_logranks.append(p_logrank)
        if not perturbed_logranks:
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}

        npr_score = float(np.mean(perturbed_logranks) / original_logrank) if original_logrank != 0 else 1.0
        normalized_score = max(0.0, min(1.0, 1 - (npr_score - 1.0)))
        prediction = 1 if npr_score < 1.40 else 0
        return {"score": normalized_score, "prediction": prediction, "raw": npr_score, "error": False}

    async def detect_lrr(self, code: str) -> Dict[str, Any]:
        ll = await self.get_log_likelihood(code)
        logrank = await self.get_log_rank(code, log=True)
        if np.isnan(ll) or np.isnan(logrank) or logrank == 0:
            return {"score": 0.0, "prediction": 0, "error": True}
        score = float(-ll / logrank)
        prediction = 1 if score > 3.5 else 0
        return {"score": score, "prediction": prediction, "error": False}

    async def detect_logrank(self, code: str) -> Dict[str, Any]:
        logrank = await self.get_log_rank(code, log=True)
        if np.isnan(logrank):
            return {"score": 0.0, "prediction": 0, "error": True}
        score = float(-logrank)
        prediction = 1 if score > -1.5 else 0
        return {"score": score, "prediction": prediction, "error": False}

    async def detect_entropy(self, code: str) -> Dict[str, Any]:
        entropy = await self.get_entropy(code)
        if np.isnan(entropy):
            return {"score": 0.0, "prediction": 0, "error": True}
        score = float(-entropy)
        prediction = 1 if score > -3.0 else 0
        return {"score": score, "prediction": prediction, "error": False}

    async def detect_likelihood(self, code: str) -> Dict[str, Any]:
        ll = await self.get_log_likelihood(code)
        if np.isnan(ll):
            return {"score": 0.0, "prediction": 0, "error": True}
        prediction = 1 if ll > -2.0 else 0
        return {"score": float(ll), "prediction": prediction, "error": False}

    async def detect_detectgpt(self, code: str) -> Dict[str, Any]:
        original_ll = await self.get_log_likelihood(code)
        if np.isnan(original_ll):
            return {"score": 0.0, "prediction": 0, "curvature": 0.0, "error": True}

        perturbed_lls = []
        for _ in range(self.n_perturbations):
            p_ll = await self.get_log_likelihood(self.perturb_simple(code))
            if not np.isnan(p_ll):
                perturbed_lls.append(p_ll)
        if len(perturbed_lls) < 2:
            return {"score": 0.0, "prediction": 0, "curvature": 0.0, "error": True}

        mean_p = float(np.mean(perturbed_lls))
        std_p = float(np.std(perturbed_lls))
        curvature = (original_ll - mean_p) / std_p if std_p > 1e-6 else (original_ll - mean_p)
        prediction = 1 if curvature > 1.5 else 0
        return {"score": float(curvature), "prediction": prediction, "curvature": float(curvature), "error": False}

    async def detect_t5_npr(self, code: str) -> Dict[str, Any]:
        original_logrank = await self.get_log_rank(code, log=True)
        if np.isnan(original_logrank):
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}

        perturbed_logranks = []
        for _ in range(self.n_perturbations):
            perturbed = await self.perturb_t5_mask(code)
            p_logrank = await self.get_log_rank(perturbed, log=True)
            if not np.isnan(p_logrank):
                perturbed_logranks.append(p_logrank)
        if not perturbed_logranks:
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}

        npr_score = float(np.mean(perturbed_logranks) / original_logrank) if original_logrank != 0 else 1.0
        normalized_score = max(0.0, min(1.0, 1 - (npr_score - 1.0)))
        prediction = 1 if npr_score < 1.40 else 0
        return {"score": normalized_score, "prediction": prediction, "raw": npr_score, "error": False}

    async def detect_identifier_npr(self, code: str, language: str = "python") -> Dict[str, Any]:
        original_logrank = await self.get_log_rank(code, log=True)
        if np.isnan(original_logrank):
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}

        perturbed_logranks = []
        for _ in range(self.n_perturbations):
            perturbed = await self.perturb_identifiers(code, language)
            p_logrank = await self.get_log_rank(perturbed, log=True)
            if not np.isnan(p_logrank):
                perturbed_logranks.append(p_logrank)
        if not perturbed_logranks:
            return {"score": 0.5, "prediction": 0, "raw": 1.0, "error": True}

        npr_score = float(np.mean(perturbed_logranks) / original_logrank) if original_logrank != 0 else 1.0
        normalized_score = max(0.0, min(1.0, 1 - (npr_score - 1.0)))
        prediction = 1 if npr_score < 1.40 else 0
        return {"score": normalized_score, "prediction": prediction, "raw": npr_score, "error": False}

    async def extract_features(self, code: str, language: str) -> Dict[str, float]:
        features: Dict[str, float] = {}
        alias = self.alias

        npr = await self.detect_npr(code)
        features[f"{alias}_npr_score"] = float(npr["score"])
        features[f"{alias}_npr_prediction"] = float(npr["prediction"])
        features[f"{alias}_npr_raw"] = float(npr["raw"])

        lrr = await self.detect_lrr(code)
        features[f"{alias}_lrr_score"] = float(lrr["score"])
        features[f"{alias}_lrr_prediction"] = float(lrr["prediction"])

        logrank = await self.detect_logrank(code)
        features[f"{alias}_logrank_score"] = float(logrank["score"])
        features[f"{alias}_logrank_prediction"] = float(logrank["prediction"])

        entropy = await self.detect_entropy(code)
        features[f"{alias}_entropy_score"] = float(entropy["score"])
        features[f"{alias}_entropy_prediction"] = float(entropy["prediction"])

        ll = await self.detect_likelihood(code)
        features[f"{alias}_likelihood_score"] = float(ll["score"])
        features[f"{alias}_likelihood_prediction"] = float(ll["prediction"])

        detectgpt = await self.detect_detectgpt(code)
        features[f"{alias}_detectgpt_score"] = float(detectgpt["score"])
        features[f"{alias}_detectgpt_prediction"] = float(detectgpt["prediction"])
        features[f"{alias}_detectgpt_curvature"] = float(detectgpt["curvature"])

        t5_npr = await self.detect_t5_npr(code)
        features[f"{alias}_t5npr_score"] = float(t5_npr["score"])
        features[f"{alias}_t5npr_prediction"] = float(t5_npr["prediction"])
        features[f"{alias}_t5npr_raw"] = float(t5_npr["raw"])

        id_npr = await self.detect_identifier_npr(code, language)
        features[f"{alias}_idnpr_score"] = float(id_npr["score"])
        features[f"{alias}_idnpr_prediction"] = float(id_npr["prediction"])
        features[f"{alias}_idnpr_raw"] = float(id_npr["raw"])

        return features

    async def close(self) -> None:
        self._stats_cache.clear()
        shutdown = getattr(self.engine, "shutdown_background_loop", None)
        if callable(shutdown):
            maybe_coro = shutdown()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AICodeDetectionService:
    def __init__(
        self,
        detector_aliases: List[str],
        detectors: Dict[str, DetectorModel],
        classifier: ANNClassifier,
        n_perturbations: int,
    ) -> None:
        self.detector_aliases = detector_aliases
        self.detectors = detectors
        self.classifier = classifier
        self.n_perturbations = n_perturbations
        self._lock = asyncio.Lock()

    @classmethod
    async def create(
        cls,
        detector_aliases: Optional[List[str]] = None,
        n_perturbations: int = 5,
        model_path: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> "AICodeDetectionService":
        aliases = detector_aliases or list(FEATURE_MODELS.keys())
        invalid = [a for a in aliases if a not in FEATURE_MODELS]
        if invalid:
            raise ValueError(f"Invalid detector aliases: {invalid}")

        prompt_logprob_topk = int(os.getenv("VLLM_PROMPT_LOGPROBS_TOPK", "20"))
        max_length = int(os.getenv("DETECTOR_MAX_LENGTH", "512"))
        classifier = ANNClassifier(model_path=model_path, threshold=threshold)

        detectors: Dict[str, DetectorModel] = {}
        for alias in aliases:
            detectors[alias] = await DetectorModel.create(
                alias=alias,
                n_perturbations=n_perturbations,
                max_length=max_length,
                prompt_logprob_topk=prompt_logprob_topk,
            )

        return cls(
            detector_aliases=aliases,
            detectors=detectors,
            classifier=classifier,
            n_perturbations=n_perturbations,
        )

    async def evaluate(self, code: str, language: str) -> Dict[str, Any]:
        async with self._lock:
            all_features: Dict[str, float] = {}
            for alias in self.detector_aliases:
                detector_features = await self.detectors[alias].extract_features(
                    code=code,
                    language=language,
                )
                all_features.update(detector_features)

            result = await self.classifier.predict(
                features=all_features,
                language=language,
                detectors_used=self.detector_aliases,
            )
            result["perturbations"] = self.n_perturbations
            result["scoring_backend"] = "vllm_async_engine"
            return result

    async def close(self) -> None:
        for detector in self.detectors.values():
            await detector.close()
