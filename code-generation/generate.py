import time
import openai
import numpy as np
import re
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from loguru import logger
import gzip
import json
import pdb
from tqdm import tqdm
import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_HF_HOOKS"] = "1"
os.environ["ACCELERATE_USE_FSDP"] = "0"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "0"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def extract_prompt_solution(code: str, language: str) -> tuple:
    """
    Extract prompt (function signature/header) and solution (function body) 
    from source code for any supported language.
    
    Strategy: Find where the function body starts and split there.
    - For C-style languages: split at first '{'
    - For Python: split at first ':' after def/class or after docstring
    - For Ruby: split after 'def ...'
    
    Returns:
        (prompt, solution) tuple, or (None, None) if parsing fails
    """
    code = code.strip()
    if not code or len(code) < 20:  # Too short
        return None, None
    
    lines = code.split('\n')
    if len(lines) < 2:  # Need at least 2 lines
        return None, None
    
    # Python: special handling for docstrings
    if language == 'python':
        # Replace single quotes with double quotes for consistency
        code = code.replace("'''", '"""')
        parts = code.split('"""')
        if len(parts) >= 3:
            prompt = parts[0] + '"""' + parts[1] + '"""'
            solution = '"""'.join(parts[2:])  # Join remaining parts
            if solution.strip():
                return prompt.strip(), solution.strip()
        
        # Fallback: Find 'def' or 'class' and split after the colon line
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('def ') or stripped.startswith('class ') or 
                stripped.startswith('async def ')):
                # Find line ending with ':'
                for j in range(i, min(i + 5, len(lines))):
                    if lines[j].rstrip().endswith(':'):
                        prompt = '\n'.join(lines[:j+1])
                        solution = '\n'.join(lines[j+1:])
                        if solution.strip():
                            return prompt.strip(), solution.strip()
                        break
        return None, None
    
    # Ruby: split after 'def ...'
    elif language == 'ruby':
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('def('):
                prompt = '\n'.join(lines[:i+1])
                solution = '\n'.join(lines[i+1:])
                if solution.strip():
                    return prompt.strip(), solution.strip()
        return None, None
    
    # All other languages (C-style): split at first '{'
    # This covers: javascript, java, c_sharp, cpp, c, go, rust, php, typescript
    else:
        # Find the first line containing '{'
        for i, line in enumerate(lines):
            if '{' in line:
                # Include this line in prompt (it has the opening brace)
                prompt = '\n'.join(lines[:i+1])
                solution = '\n'.join(lines[i+1:])
                
                # Make sure we have meaningful content
                if len(prompt.strip()) >= 10 and len(solution.strip()) >= 5:
                    return prompt.strip(), solution.strip()
        
        # Fallback: just use first 30% as prompt, rest as solution
        split_point = max(1, len(lines) // 3)
        prompt = '\n'.join(lines[:split_point])
        solution = '\n'.join(lines[split_point:])
        if len(prompt.strip()) >= 10 and len(solution.strip()) >= 5:
            return prompt.strip(), solution.strip()
        
        return None, None



def load_data(path='data/CodeSearchNet', language='python', max_num=10000):

    all_prompts = []
    all_solutions = []

    if 'humaneval' in path:
        path_to_data = f'{path}/{language}/data/humaneval_{language}.jsonl.gz'

        logger.info(f'Loading data from {path_to_data}')

        with gzip.open(path_to_data, 'rb') as f:

            for line in f:
                data = json.loads(line)

                all_prompts.append(data['prompt'])
                all_solutions.append(data['canonical_solution'])

    elif 'CodeSearchNet' in path:

        path_to_data = f'{path}/{language}/train.jsonl'

        logger.info(f'Loading data from {path_to_data}')

        failed = 0
        success = 0

        max_prompt_len = 128
        min_prompt_len = 5

        max_solution_len = 256
        min_solution_len = 5

        with open(path_to_data, 'r') as f:

            count = 0
            for line in tqdm(f):

                data = json.loads(line)

                data['original_string'] = data['original_string'].replace("'''", '"""')
                try:
                    prompt = data['original_string'].split('"""')[0] + '"""' + data['original_string'].split('"""')[1] + '"""'
                    solution = data['original_string'].split('"""')[2]
                    success += 1
                except:
                    failed += 1


                if len(prompt.split()) > max_prompt_len or len(prompt.split()) < min_prompt_len:
                    continue

                if len(solution.split()) > max_solution_len or len(solution.split()) < min_solution_len:
                    continue

                all_prompts.append(prompt)
                all_solutions.append(solution)

        logger.info(f'Failed: {failed}, Success: {success}')

    elif "TheVault" in path or "dataset" in path:
        # Try small_train.jsonl first, fallback to train.jsonl
        path_to_data = f'{path}/{language}/small_train.jsonl'
        
        if not os.path.exists(path_to_data):
            path_to_data = f'{path}/{language}/train.jsonl'
        
        if not os.path.exists(path_to_data):
            logger.error(f'Dataset file not found: {path}/{language}/small_train.jsonl or train.jsonl')
            logger.error(f'Please run: python download_dataset.py {path} --set function')
            return [], []

        logger.info(f'Loading data from {path_to_data}')

        failed = 0
        success = 0

        max_prompt_len = 128
        min_prompt_len = 5

        max_solution_len = 256
        min_solution_len = 5

        with open(path_to_data, 'r', encoding='utf-8') as f:

            count = 0
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    failed += 1
                    continue

                if 'original_string' not in data:
                    failed += 1
                    continue

                code = data['original_string']
                
                # Language-specific prompt/solution extraction
                prompt, solution = extract_prompt_solution(code, language)
                
                if prompt is None or solution is None:
                    failed += 1
                    continue
                
                success += 1

                if len(prompt.split()) > max_prompt_len or len(prompt.split()) < min_prompt_len:
                    continue

                if len(solution.split()) > max_solution_len or len(solution.split()) < min_solution_len:
                    continue

                all_prompts.append(prompt)
                all_solutions.append(solution)

        logger.info(f'Failed: {failed}, Success: {success}')

    else:
        logger.error(f'Unknown dataset path format: {path}')
        logger.error('Expected path containing "TheVault", "CodeSearchNet", "humaneval", or "dataset"')
        return [], []

    logger.info(f'Loaded {len(all_prompts)} prompts and {len(all_solutions)} solutions')

    # Check if we have any data
    if len(all_prompts) == 0:
        logger.error('No valid prompts loaded! Check your dataset format.')
        logger.error('The dataset should contain JSONL with "original_string" field containing docstrings.')
        return [], []

    # analyze the lengths
    prompt_lengths = [len(prompt.split()) for prompt in all_prompts]
    solution_lengths = [len(solution.split()) for solution in all_solutions]
    logger.info(f'Prompt lengths: min: {min(prompt_lengths)}, max: {max(prompt_lengths)}, mean: {np.mean(prompt_lengths):.2f}, std: {np.std(prompt_lengths):.2f}')
    logger.info(f'Solution lengths: min: {min(solution_lengths)}, max: {max(solution_lengths)}, mean: {np.mean(solution_lengths)}, std: {np.std(solution_lengths)}')

    if len(all_prompts) > max_num:

        seed = 42
        np.random.seed(seed)
        indices = np.random.choice(len(all_prompts), max_num, replace=False)
        all_prompts = [all_prompts[i] for i in indices]
        all_solutions = [all_solutions[i] for i in indices]

        prompt_lengths = [len(prompt.split()) for prompt in all_prompts]
        solution_lengths = [len(solution.split()) for solution in all_solutions]

        logger.info(f'Sampled {len(all_prompts)} prompts and {len(all_solutions)} solutions')
        logger.info(f'Prompt lengths: min: {min(prompt_lengths)}, max: {max(prompt_lengths)}, mean: {np.mean(prompt_lengths)}, std: {np.std(prompt_lengths)}')
        logger.info(f'Solution lengths: min: {min(solution_lengths)}, max: {max(solution_lengths)}, mean: {np.mean(solution_lengths)}, std: {np.std(solution_lengths)}')

    return all_prompts, all_solutions


def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            re.escape('<|endoftext|>')
        ]
    ]


    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def load_model(model_name):
    """Load model and tokenizer once for reuse across languages."""
    logger.info(f'Loading model: {model_name}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if 'codegen-' in model_name.lower():
        tokenizer.pad_token_id = 50256
        tokenizer.padding_side = 'left'
    elif 'santa' in model_name.lower():
        tokenizer.pad_token_id = 49156  # https://huggingface.co/bigcode/santacoder/blob/main/special_tokens_map.json
        logger.info(f'pad_token: {tokenizer.pad_token}')
        tokenizer.padding_side = 'left'
    elif 'parrot' in model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f'pad_token: {tokenizer.pad_token}')
        tokenizer.padding_side = 'left'
    elif "incoder" in model_name.lower():
        tokenizer.pad_token_id = 1
        tokenizer.padding_side = 'left'
    elif "phi-1" in model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f'pad_token: {tokenizer.pad_token}')
        tokenizer.padding_side = 'left'
    else:
        # Default configuration for other models (decoder-only models need left padding)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f'Using default tokenizer config: pad_token={tokenizer.pad_token}, padding_side=left')

    # Track if we're using device_map (multi-GPU or auto placement)
    uses_device_map = False
    
    if 't5p' in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                      dtype=torch.float16,
                                                      trust_remote_code=True)
    elif "llama" in model_name.lower() or "wizard" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float16)
    elif "codegen2" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="balanced", trust_remote_code=True)
        uses_device_map = True

    # Only move to device if not using device_map (device_map handles placement automatically)
    if not uses_device_map:
        model = model.to(device)
        model_device = device
    else:
        # For device_map models, get the device of the first parameter
        model_device = next(model.parameters()).device
        logger.info(f'Model using device_map, input device: {model_device}')
    
    logger.info(f'Model loaded successfully!')
    return model, tokenizer, model_device


def generate_hf(model, tokenizer, model_device, model_name, prompts, solutions, batch_size=16, max_length_sample=128, max_length=128, do_sample=True, top_p=0.95, temperature=0.2):
    """Generate code using pre-loaded model and tokenizer."""
    
    all_outputs = []

    if 'starcoder' in model_name.lower() or "llama" in model_name.lower() or "wizard" in model_name.lower() or "codegen2" in model_name.lower():
        input_ids = [tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids for prompt in prompts]

        def_id = tokenizer('def', add_special_tokens=False).input_ids[0]
        try:
            def_with_space_id = tokenizer('def', add_prefix_space=True, add_special_tokens=False).input_ids[0]
        except:
            def_with_space_id = tokenizer(' def', add_special_tokens=False).input_ids[0]

        eos_id_list = [tokenizer.eos_token_id, def_id, def_with_space_id]
        logger.info(f'eos_id_list: {eos_id_list}')

        for input_ids in tqdm(input_ids, ncols=50):
            input_ids = input_ids.to(model_device)
            input_ids_len = input_ids.shape[1]
            logger.info(f'input_ids_len: {input_ids_len}')

            if max_length_sample >= 256:
                outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length_sample+input_ids_len, top_p=top_p, temperature=temperature, pad_token_id=tokenizer.pad_token_id, use_cache=True, eos_token_id=eos_id_list)
            else:
                outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length_sample+input_ids_len, top_p=top_p, temperature=temperature, pad_token_id=tokenizer.pad_token_id, use_cache=True)

            decoded_output = tokenizer.decode(outputs[0, input_ids_len:])
            # logger.info(f'decoded_output: {decoded_output}')
            all_outputs.append(decoded_output)
            outputs = all_outputs
    else:
        # Tokenize with attention_mask to avoid warning when pad_token == eos_token
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, return_attention_mask=True)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        input_ids_len = input_ids.shape[1]
        logger.info(f'input_ids_len: {input_ids_len}')

        # create a dataset from the samples (include attention_mask)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)

        if batch_size >= 4:
            num_workers = batch_size // 2
        else:
            num_workers = 1

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

        for batch in tqdm(dataloader, ncols=50):
            batch_input_ids = batch[0].to(model_device)
            batch_attention_mask = batch[1].to(model_device)
            outputs = model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask,
                do_sample=do_sample, 
                max_length=max_length_sample+input_ids_len, 
                top_p=top_p, 
                temperature=temperature, 
                pad_token_id=tokenizer.pad_token_id, 
                use_cache=True
            )
            
            all_outputs.append(outputs)

        samples = torch.cat(all_outputs, dim=0)
        outputs = tokenizer.batch_decode(samples[:, input_ids_len:, ...])

    # truncate the outputs (based on the original code of CodeGen)
    outputs = [truncate(output) for output in outputs]

    logger.info(f'Generated {len(outputs)} samples')

    logger.info("Showing first 3 samples")

    for i in range(min(3, len(prompts))):
        logger.info(f'Example {i}:')
        logger.info(f'Prompt: \n{prompts[i]}')
        logger.info(f'Output: \n{outputs[i]}')
        logger.info(f'Solution: \n{solutions[i]}')

    # pdb.set_trace()
    return prompts, outputs, solutions


# Supported languages (same as download_dataset.py)
SUPPORTED_LANGUAGES = ['python', 'javascript', 'java', 'c', 'cpp', 'go', 'rust', 'ruby', 'php', 'c_sharp']


def generate_for_language(model, tokenizer, model_device, path, language, model_name, max_num, temperature, batch_size, max_length, output_base_dir="dataset_ai"):
    """Generate code for a single language using pre-loaded model."""
    logger.info(f'\n{"="*60}')
    logger.info(f'Processing language: {language}')
    logger.info(f'{"="*60}')
    
    prompts, solutions = load_data(path=path, language=language, max_num=max_num)
    
    if len(prompts) == 0:
        logger.warning(f'No data found for {language}, skipping...')
        return None
    
    prompts, outputs, solutions = generate_hf(
        model, tokenizer, model_device, model_name, prompts, solutions, 
        max_length_sample=max_length,
        max_length=128, 
        do_sample=True, 
        top_p=0.95, 
        temperature=temperature, 
        batch_size=batch_size
    )
    
    model_short_name = model_name.split('/')[-1]
    
    logger.info(f'Generated {len(outputs)} outputs for {language}')
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, language)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write JSON output
    file_name = f'{output_dir}/outputs.txt'
    if os.path.exists(file_name):
        os.remove(file_name)
    
    with open(file_name, 'w+', encoding='utf-8') as f:
        for i in range(len(outputs)):
            results = {
                'language': language,
                'prompt': prompts[i], 
                'output': outputs[i], 
                'solution': solutions[i]
            }
            f.write(json.dumps(results))
            f.write('\n')
    
    logger.info(f'Saved {len(outputs)} samples to {file_name}')
    
    # Also write human-readable version
    file_name_v2 = f'{output_dir}/outputs_v2.txt'
    if os.path.exists(file_name_v2):
        os.remove(file_name_v2)
    
    with open(file_name_v2, 'a', encoding='utf-8') as f:
        for i in range(len(outputs)):
            f.write(f'Prompt: \n{prompts[i]}\n')
            f.write(f'Output: \n{outputs[i]}\n')
            f.write(f'Solution: \n{solutions[i]}\n')
    
    return {
        'language': language,
        'count': len(outputs),
        'output_file': file_name
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate AI code samples from human-written prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for Python only
  python generate.py --path ../dataset --language python --model_name Salesforce/codegen-350M-mono

  # Generate for all languages
  python generate.py --path ../dataset --language all --model_name Salesforce/codegen-350M-mono

  # Generate for specific languages
  python generate.py --path ../dataset --language python,java,javascript --model_name Salesforce/codegen-350M-mono

Supported languages: python, javascript, java, c, cpp, go, rust, ruby, php, c_sharp
"""
    )
    parser.add_argument('--path', type=str, default="../dataset",
                        help='Path to dataset directory')
    parser.add_argument('--language', '-l', type=str, default='python',
                        help='Language(s) to process: single language, comma-separated list, or "all"')
    parser.add_argument('--max_num', type=int, default=1000,
                        help='Maximum number of samples per language')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Generation temperature (higher = more creative)')
    parser.add_argument('--model_name', type=str, default='Salesforce/codegen-350M-mono',
                        help='HuggingFace model for code generation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum generation length')
    parser.add_argument('--output_dir', type=str, default='dataset_ai',
                        help='Base output directory for generated outputs')
    args = parser.parse_args()

    logger.info(f'Arguments: {args}')

    # Parse language argument
    if args.language.lower() == 'all':
        languages = SUPPORTED_LANGUAGES
    else:
        languages = [lang.strip() for lang in args.language.split(',')]
        # Validate languages
        for lang in languages:
            if lang not in SUPPORTED_LANGUAGES:
                logger.warning(f'Unknown language: {lang}. Supported: {SUPPORTED_LANGUAGES}')

    logger.info(f'Processing languages: {languages}')
    
    # Load model once for all languages
    logger.info(f'Loading model once for reuse across all languages...')
    model, tokenizer, model_device = load_model(args.model_name)
    
    # Track results
    results_summary = []
    
    # Process each language (reusing the same model)
    for language in languages:
        try:
            result = generate_for_language(
                model=model,
                tokenizer=tokenizer,
                model_device=model_device,
                path=args.path,
                language=language,
                model_name=args.model_name,
                max_num=args.max_num,
                temperature=args.temperature,
                batch_size=args.batch_size,
                max_length=args.max_length,
                output_base_dir=args.output_dir
            )
            if result:
                results_summary.append(result)
        except Exception as e:
            logger.error(f'Error processing {language}: {e}')
            continue
    
    # Print summary
    logger.info('\n' + '='*60)
    logger.info('GENERATION SUMMARY')
    logger.info('='*60)
    
    total_samples = 0
    for result in results_summary:
        logger.info(f"  {result['language']:<15} {result['count']:>6} samples -> {result['output_file']}")
        total_samples += result['count']
    
    logger.info('-'*60)
    logger.info(f'  Total: {total_samples} samples across {len(results_summary)} languages')
    logger.info('='*60)

