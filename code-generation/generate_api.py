"""
Generate AI Code Dataset from API-based LLM Providers.

Generates AI-written code from private/proprietary LLM APIs using human-written
prompts as input. Produces output in the same JSONL format as generate.py so
the downstream feature-extraction and training pipeline works unchanged.

Supported providers:
  - OpenAI        (GPT-4o, GPT-4, GPT-3.5-turbo, o1, o3-mini, etc.)
  - Anthropic     (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
  - Google        (Gemini 2.0 Flash, Gemini 1.5 Pro, etc.)
  - OpenAI-compat (Groq, Together, DeepSeek API, Mistral API, any /v1/chat/completions endpoint)

Usage:
    # OpenAI GPT-4o
    python generate_api.py --provider openai --model gpt-4o --path ../dataset_human --language python

    # Anthropic Claude 3.5 Sonnet
    python generate_api.py --provider anthropic --model claude-3-5-sonnet-20241022 --path ../dataset_human --language all

    # Google Gemini
    python generate_api.py --provider google --model gemini-2.0-flash --path ../dataset_human --language python

    # Any OpenAI-compatible API (e.g. DeepSeek, Groq, Together, local vLLM)
    python generate_api.py --provider openai-compat --model deepseek-coder --base-url https://api.deepseek.com/v1 --api-key $DEEPSEEK_KEY --path ../dataset_human --language python

    # List supported providers
    python generate_api.py --list-providers
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    "python", "javascript", "java", "c", "cpp",
    "go", "rust", "ruby", "php", "c_sharp",
]

LANGUAGE_SYSTEM_PROMPTS = {
    "python":     "You are an expert Python programmer. Complete the given code. Only output the code body, no explanations.",
    "javascript": "You are an expert JavaScript programmer. Complete the given code. Only output the code body, no explanations.",
    "java":       "You are an expert Java programmer. Complete the given code. Only output the code body, no explanations.",
    "c":          "You are an expert C programmer. Complete the given code. Only output the code body, no explanations.",
    "cpp":        "You are an expert C++ programmer. Complete the given code. Only output the code body, no explanations.",
    "go":         "You are an expert Go programmer. Complete the given code. Only output the code body, no explanations.",
    "rust":       "You are an expert Rust programmer. Complete the given code. Only output the code body, no explanations.",
    "ruby":       "You are an expert Ruby programmer. Complete the given code. Only output the code body, no explanations.",
    "php":        "You are an expert PHP programmer. Complete the given code. Only output the code body, no explanations.",
    "c_sharp":    "You are an expert C# programmer. Complete the given code. Only output the code body, no explanations.",
}

PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "openai": {
        "description": "OpenAI API (GPT-4o, GPT-4, GPT-3.5-turbo, o1, o3-mini, etc.)",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "base_url": None,
    },
    "anthropic": {
        "description": "Anthropic API (Claude 3.5 Sonnet, Claude 3 Opus, etc.)",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-sonnet-20241022",
        "base_url": None,
    },
    "google": {
        "description": "Google Generative AI API (Gemini 2.0 Flash, Gemini 1.5 Pro, etc.)",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
        "base_url": None,
    },
    "openai-compat": {
        "description": "Any OpenAI-compatible API (Groq, Together, DeepSeek, Mistral, local vLLM)",
        "env_key": "OPENAI_COMPAT_API_KEY",
        "default_model": "deepseek-coder",
        "base_url": "https://api.deepseek.com/v1",
    },
}


class APIProvider(ABC):
    """Base class for LLM API providers."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.2,
                 max_tokens: int = 512, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate code completion from a prompt. Returns the generated text."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""


class OpenAIProvider(APIProvider):
    """OpenAI API provider (also works for Azure OpenAI)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def generate(self, prompt: str, system_prompt: str) -> str:
        for attempt in range(5):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Complete this code:\n\n{prompt}"},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    wait = 2 ** attempt + 1
                    logger.warning("Rate limited, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                if attempt < 4:
                    logger.warning("API error (attempt %d/5): %s", attempt + 1, e)
                    time.sleep(1)
                    continue
                logger.error("Failed after 5 attempts: %s", e)
                return ""
        return ""


class OpenAICompatProvider(OpenAIProvider):
    """Any OpenAI-compatible API endpoint (Groq, Together, DeepSeek, Mistral, vLLM)."""

    @property
    def provider_name(self) -> str:
        return f"OpenAI-Compatible ({self.base_url})"


class AnthropicProvider(APIProvider):
    """Anthropic Claude API provider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    def generate(self, prompt: str, system_prompt: str) -> str:
        for attempt in range(5):
            try:
                message = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"Complete this code:\n\n{prompt}"},
                    ],
                    temperature=self.temperature,
                )
                content_blocks = message.content
                return "".join(
                    block.text for block in content_blocks if hasattr(block, "text")
                )
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "overloaded" in error_str:
                    wait = 2 ** attempt + 1
                    logger.warning("Rate limited, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                if attempt < 4:
                    logger.warning("API error (attempt %d/5): %s", attempt + 1, e)
                    time.sleep(1)
                    continue
                logger.error("Failed after 5 attempts: %s", e)
                return ""
        return ""


class GoogleProvider(APIProvider):
    """Google Generative AI API provider for Gemini models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install: pip install google-generativeai"
            )
        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model_instance = genai.GenerativeModel(
            self.model,
            system_instruction=None,
        )

    @property
    def provider_name(self) -> str:
        return "Google"

    def generate(self, prompt: str, system_prompt: str) -> str:
        for attempt in range(5):
            try:
                full_prompt = f"{system_prompt}\n\nComplete this code:\n\n{prompt}"
                response = self._model_instance.generate_content(
                    full_prompt,
                    generation_config=self._genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                return response.text or ""
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "quota" in error_str:
                    wait = 2 ** attempt + 2
                    logger.warning("Rate limited, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                if attempt < 4:
                    logger.warning("API error (attempt %d/5): %s", attempt + 1, e)
                    time.sleep(1)
                    continue
                logger.error("Failed after 5 attempts: %s", e)
                return ""
        return ""


def create_provider(provider_name: str, model: str, api_key: str,
                    temperature: float = 0.2, max_tokens: int = 512,
                    base_url: Optional[str] = None) -> APIProvider:
    """Factory function to create the appropriate API provider."""
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "base_url": base_url,
    }

    if provider_name == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_name == "anthropic":
        return AnthropicProvider(**kwargs)
    elif provider_name == "google":
        return GoogleProvider(**kwargs)
    elif provider_name == "openai-compat":
        if not base_url:
            raise ValueError("--base-url is required for openai-compat provider")
        return OpenAICompatProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported: {list(PROVIDER_REGISTRY.keys())}"
        )


def extract_prompt_solution(code: str, language: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract prompt (function header) and solution (function body) from source code.
    Reuses the same logic as generate.py for consistency."""
    code = code.strip()
    if not code or len(code) < 20:
        return None, None

    lines = code.split("\n")
    if len(lines) < 2:
        return None, None

    if language == "python":
        code_norm = code.replace("'''", '"""')
        parts = code_norm.split('"""')
        if len(parts) >= 3:
            prompt = parts[0] + '"""' + parts[1] + '"""'
            solution = '"""'.join(parts[2:])
            if solution.strip():
                return prompt.strip(), solution.strip()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("async def "):
                for j in range(i, min(i + 5, len(lines))):
                    if lines[j].rstrip().endswith(":"):
                        prompt = "\n".join(lines[:j + 1])
                        solution = "\n".join(lines[j + 1:])
                        if solution.strip():
                            return prompt.strip(), solution.strip()
                        break
        return None, None

    elif language == "ruby":
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("def("):
                prompt = "\n".join(lines[:i + 1])
                solution = "\n".join(lines[i + 1:])
                if solution.strip():
                    return prompt.strip(), solution.strip()
        return None, None

    else:
        for i, line in enumerate(lines):
            if "{" in line:
                prompt = "\n".join(lines[:i + 1])
                solution = "\n".join(lines[i + 1:])
                if len(prompt.strip()) >= 10 and len(solution.strip()) >= 5:
                    return prompt.strip(), solution.strip()

        split_point = max(1, len(lines) // 3)
        prompt = "\n".join(lines[:split_point])
        solution = "\n".join(lines[split_point:])
        if len(prompt.strip()) >= 10 and len(solution.strip()) >= 5:
            return prompt.strip(), solution.strip()

        return None, None


def load_human_prompts(dataset_path: str, language: str, max_num: int = 1000) -> List[Tuple[str, str]]:
    """Load human code prompts from the dataset. Returns list of (prompt, solution) tuples."""
    prompts_solutions: List[Tuple[str, str]] = []

    small_train = Path(dataset_path) / language / "small_train.jsonl"
    train_file = Path(dataset_path) / language / "train.jsonl"

    data_path = small_train if small_train.exists() else train_file
    if not data_path.exists():
        logger.error("Dataset file not found: %s or %s", small_train, train_file)
        return []

    logger.info("Loading prompts from %s", data_path)

    max_prompt_len = 128
    min_prompt_len = 5
    max_solution_len = 256
    min_solution_len = 5

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            code = data.get("original_string", data.get("code", data.get("content", "")))
            if not code:
                continue

            prompt, solution = extract_prompt_solution(code, language)
            if prompt is None or solution is None:
                continue

            if not (min_prompt_len <= len(prompt.split()) <= max_prompt_len):
                continue
            if not (min_solution_len <= len(solution.split()) <= max_solution_len):
                continue

            prompts_solutions.append((prompt, solution))

    logger.info("Loaded %d prompts for %s", len(prompts_solutions), language)

    if len(prompts_solutions) > max_num:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(prompts_solutions), max_num, replace=False)
        prompts_solutions = [prompts_solutions[i] for i in indices]
        logger.info("Sampled down to %d prompts", len(prompts_solutions))

    return prompts_solutions


def generate_for_language(
    provider: APIProvider,
    dataset_path: str,
    language: str,
    max_num: int,
    output_base_dir: str,
    concurrency: int = 1,
    delay: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Generate code for a single language and save in the pipeline-compatible format."""
    logger.info("=" * 60)
    logger.info("Processing language: %s", language)
    logger.info("=" * 60)

    prompts_solutions = load_human_prompts(dataset_path, language, max_num)
    if not prompts_solutions:
        logger.warning("No prompts found for %s, skipping", language)
        return None

    system_prompt = LANGUAGE_SYSTEM_PROMPTS.get(
        language,
        "You are an expert programmer. Complete the given code. Only output the code body, no explanations.",
    )

    prompts = [ps[0] for ps in prompts_solutions]
    solutions = [ps[1] for ps in prompts_solutions]
    outputs: List[str] = [""] * len(prompts)
    failed_count = 0

    def _generate_one(idx: int) -> Tuple[int, str]:
        if delay > 0:
            time.sleep(delay)
        result = provider.generate(prompts[idx], system_prompt)
        return idx, result

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_generate_one, i): i for i in range(len(prompts))}
            for future in tqdm(as_completed(futures), total=len(prompts),
                               desc=f"Generating ({language})", ncols=80):
                idx, result = future.result()
                outputs[idx] = result
                if not result:
                    failed_count += 1
    else:
        for i in tqdm(range(len(prompts)), desc=f"Generating ({language})", ncols=80):
            _, result = _generate_one(i)
            outputs[i] = result
            if not result:
                failed_count += 1

    valid_outputs = [(p, o, s) for p, o, s in zip(prompts, outputs, solutions) if o.strip()]
    logger.info(
        "Generated %d valid outputs for %s (%d failed)",
        len(valid_outputs), language, failed_count,
    )

    if not valid_outputs:
        logger.error("No valid outputs generated for %s", language)
        return None

    output_dir = Path(output_base_dir) / language
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_file = output_dir / "outputs.txt"
    with open(outputs_file, "w", encoding="utf-8") as f:
        for prompt, output, solution in valid_outputs:
            record = {
                "language": language,
                "prompt": prompt,
                "output": output,
                "solution": solution,
            }
            f.write(json.dumps(record) + "\n")
    logger.info("Saved %d samples to %s", len(valid_outputs), outputs_file)

    outputs_v2_file = output_dir / "outputs_v2.txt"
    with open(outputs_v2_file, "w", encoding="utf-8") as f:
        for prompt, output, solution in valid_outputs:
            f.write(f"Prompt: \n{prompt}\n")
            f.write(f"Output: \n{output}\n")
            f.write(f"Solution: \n{solution}\n")

    return {
        "language": language,
        "count": len(valid_outputs),
        "failed": failed_count,
        "output_file": str(outputs_file),
    }


def list_providers():
    """Print all supported providers and their details."""
    print("SUPPORTED API PROVIDERS")
    for name, info in PROVIDER_REGISTRY.items():
        print(f"\n  {name}")
        print(f"    Description:   {info['description']}")
        print(f"    Env variable:  {info['env_key']}")
        print(f"    Default model: {info['default_model']}")
        if info.get("base_url"):
            print(f"    Base URL:      {info['base_url']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate AI code dataset from API-based LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI GPT-4o
  python generate_api.py --provider openai --model gpt-4o --path ../dataset_human --language python

  # Anthropic Claude
  python generate_api.py --provider anthropic --model claude-3-5-sonnet-20241022 --path ../dataset_human --language all

  # Google Gemini
  python generate_api.py --provider google --model gemini-2.0-flash --path ../dataset_human --language python,java

  # OpenAI-compatible API (DeepSeek, Groq, Together, etc.)
  python generate_api.py --provider openai-compat --model deepseek-coder \\
      --base-url https://api.deepseek.com/v1 --api-key $DEEPSEEK_KEY \\
      --path ../dataset_human --language python
""",
    )
    parser.add_argument("--provider", type=str, default="openai",
                        choices=list(PROVIDER_REGISTRY.keys()),
                        help="API provider (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider-specific default)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (overrides env variable)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for OpenAI-compatible APIs")
    parser.add_argument("--path", type=str, default="../dataset_human",
                        help="Path to human code dataset")
    parser.add_argument("--language", "-l", type=str, default="python",
                        help='Language(s): single name, comma-separated, or "all"')
    parser.add_argument("--max-num", type=int, default=1000,
                        help="Maximum samples per language (default: 1000)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Generation temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--output-dir", type=str, default="dataset_ai",
                        help="Base output directory (default: dataset_ai)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent API requests (default: 1)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between API requests in seconds (default: 0.1)")
    parser.add_argument("--list-providers", action="store_true",
                        help="List supported providers and exit")

    args = parser.parse_args()

    if args.list_providers:
        list_providers()
        return

    provider_info = PROVIDER_REGISTRY[args.provider]
    model = args.model or provider_info["default_model"]

    api_key = args.api_key or os.environ.get(provider_info["env_key"], "")
    if not api_key:
        logger.error(
            "API key not provided. Set --%s or export %s",
            "api-key", provider_info["env_key"],
        )
        sys.exit(1)

    base_url = args.base_url or provider_info.get("base_url")

    if args.language.lower() == "all":
        languages = SUPPORTED_LANGUAGES
    else:
        languages = [lang.strip() for lang in args.language.split(",")]
        invalid = [l for l in languages if l not in SUPPORTED_LANGUAGES]
        if invalid:
            logger.warning("Unknown languages: %s. Supported: %s", invalid, SUPPORTED_LANGUAGES)
        languages = [l for l in languages if l in SUPPORTED_LANGUAGES]

    if not languages:
        logger.error("No valid languages specified")
        sys.exit(1)
    print("AI CODE GENERATION VIA API")
    print(f"  Provider:     {args.provider}")
    print(f"  Model:        {model}")
    print(f"  Languages:    {languages}")
    print(f"  Max samples:  {args.max_num}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Output dir:   {args.output_dir}")

    provider = create_provider(
        provider_name=args.provider,
        model=model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=base_url,
    )

    results_summary = []
    for language in languages:
        try:
            result = generate_for_language(
                provider=provider,
                dataset_path=args.path,
                language=language,
                max_num=args.max_num,
                output_base_dir=args.output_dir,
                concurrency=args.concurrency,
                delay=args.delay,
            )
            if result:
                results_summary.append(result)
        except Exception as e:
            logger.error("Error processing %s: %s", language, e)
            continue
    print("GENERATION SUMMARY")

    total_samples = 0
    total_failed = 0
    for result in results_summary:
        print(
            f"  {result['language']:<15} {result['count']:>6} samples "
            f"({result['failed']} failed) -> {result['output_file']}"
        )
        total_samples += result["count"]
        total_failed += result["failed"]
    print(f"  Total: {total_samples} samples across {len(results_summary)} languages ({total_failed} failed)")
    print(f"  Provider: {provider.provider_name} | Model: {model}")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "provider": args.provider,
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "languages": languages,
        "results": results_summary,
        "total_samples": total_samples,
        "total_failed": total_failed,
    }
    manifest_path = Path(args.output_dir) / "generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
