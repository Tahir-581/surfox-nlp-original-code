import json
import os
import re
import time
import asyncio
import aiohttp
import logging
import random

import requests
from typing import Dict, Any, List

# Initialize GPT-OSS API configuration
SERVER_URL = "http://192.168.1.150:8000"
MODEL_NAME = 'gpt-oss-20b'
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 5
CONCURRENT_REQUESTS = 60

# File paths
TITLES_FILE = 'Output Files/270k_titles_with_clusters_filtered.json'
GROUND_TRUTHS_FILE = 'Output Files/ground_truths.json'
OUTPUT_FILE = 'Output Files/270k_jan_2026_Clusters_Descriptions.json'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json(filepath):
    """Load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_json(filepath, data):
    """Save JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_and_parse_json(content: str) -> Dict[str, Any]:
    """Extract and parse JSON from response"""
    # Handle GPT-OSS delimiters
    delimiters = [
        '<|start|>assistant<|channel|>final<|message|>',
        '<|start|>assistant<|channel|>final',
        '<|end|><|start|>assistant<|channel|>final',
        'assistantfinal'
    ]
    for delimiter in delimiters:
        if delimiter in content:
            parts = content.split(delimiter)
            if len(parts) > 1:
                content = parts[-1].strip()
                # If we found a delimiter, we assume the content following it is the JSON
                # But we should still look for braces in case there is trailing text or whitespace
                break
    
    # Extract JSON from markdown or braces
    json_blocks = re.findall(r'```(?:json)?\s*\n?([^`]+)```', content, re.DOTALL)
    if json_blocks:
        content = json_blocks[-1].strip()
    
    # Try to parse the content directly first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # If direct parsing fails, try to find the LAST valid JSON object
    # This handles cases where "thinking" text contains JSON examples
    
    valid_jsons = []
    
    # Find all start braces
    start_indices = [i for i, char in enumerate(content) if char == '{']
    
    for start in start_indices:
        brace_count = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = content[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            valid_jsons.append(parsed)
                    except:
                        pass
                    break
    
    if valid_jsons:
        # Prefer the last one that has 'descriptions'
        for vj in reversed(valid_jsons):
            if 'descriptions' in vj and isinstance(vj['descriptions'], list) and len(vj['descriptions']) > 0:
                return vj
        # Fallback to just the last valid JSON
        return valid_jsons[-1]
        
    raise ValueError("Could not parse JSON from content")



def find_cluster_examples(ground_truths, silo_name):
    # Exact match
    if silo_name in ground_truths:
        cluster = ground_truths[silo_name]
        examples = [{'title': page['title'], 'description': page.get('description', '')}
                   for page in cluster.get('pages', [])
                   if page.get('title') and page.get('description')]
        return random.sample(examples, min(10, len(examples)))

    # Case-insensitive fallback
    for cluster_name, cluster_data in ground_truths.items():
        if cluster_name.lower() == silo_name.lower():
            examples = [{'title': page['title'], 'description': page.get('description', '')}
                       for page in cluster_data.get('pages', [])
                       if page.get('title') and page.get('description')]
            return random.sample(examples, min(10, len(examples)))

    return []


def create_description_prompt(cluster_name, examples, titles_batch):
    """Create a prompt for generating descriptions with semantic similarity focus"""
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Title: {ex['title']}\n"
        examples_text += f"Description: {ex['description']}\n\n"
    
    titles_list = '\n'.join([f"{i+1}. {title}" for i, title in enumerate(titles_batch)])
    
    prompt = f"""You are an expert content writer specializing in creating engaging, SEO-friendly article descriptions for dog breed content.

Context: You need to generate descriptions for articles about "{cluster_name}".

Reference Examples (these show the style and semantic similarity you should match):
{examples_text}

You need to generate descriptions for the following {len(titles_batch)} titles:
{titles_list}

Requirements:
1. Generate exactly {len(titles_batch)} descriptions - one for each title above
2. Each description should be semantically similar to the reference examples in style, tone, and content focus
3. Descriptions should be 50-150 characters in length
4. Descriptions must be engaging, informative, and appealing to dog enthusiasts
5. Match the semantic meaning and style of the reference examples - use similar phrasing patterns and keywords
6. Each description should clearly relate to its corresponding title
7. Output format: Return a valid JSON object with a single key "descriptions" containing a list of strings.
Example: {{ "descriptions": ["Description 1", "Description 2"] }}

Generate the JSON response now:"""
    
    return prompt

def check_gpt_oss_running():
    """Check if GPT-OSS server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_available_models():
    """Get list of available models from GPT-OSS server"""
    try:
        response = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            model_names = []
            for model in models:
                if isinstance(model, dict):
                    name = model.get('id', '')
                    if name:
                        model_names.append(name)
            return model_names
        return []
    except Exception as e:
        print(f"  Warning: Error getting models via REST API: {e}")
        return []

def check_model_available():
    """Check if GPT-OSS server is running and model exists"""
    print("  1. Checking if GPT-OSS server is running...")
    if not check_gpt_oss_running():
        print("  ✗ GPT-OSS server is not running!")
        print(f"  Please ensure GPT-OSS server is running at: {SERVER_URL}")
        return False
    else:
        print("  ✓ GPT-OSS server is running")
    
    print(f"  2. Checking if {MODEL_NAME} is in available models...")
    available_models = get_available_models()
    
    if available_models:
        print(f"  ✓ Found {len(available_models)} model(s)")
        model_found = False
        for model in available_models:
            if MODEL_NAME in model:
                print(f"  ✓ Model {MODEL_NAME} found in available models!")
                model_found = True
                break
        
        if not model_found:
            print(f"  ⚠ {MODEL_NAME} not in list, but will try to use it anyway...")
    else:
        print("  ⚠ Could not retrieve model list, but will attempt to use model...")
    
    print(f"  3. Ready to generate with {MODEL_NAME}")
    return True

async def generate_descriptions_async(session, prompt, num_titles, retry_count=0):
    """Generate descriptions using GPT-OSS with async and retry logic - retries until success"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 5000
        }
        
        async with session.post(
            f"{SERVER_URL}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=160)
        ) as response:
            response.raise_for_status()
            resp_json = await response.json()
            
            # Extract generated text
            generated_text = resp_json['choices'][0].get('text', '').strip()
            
            # Parse JSON using the custom extractor
            try:
                parsed_json = extract_and_parse_json(generated_text)
                descriptions = parsed_json.get('descriptions', [])
                if isinstance(descriptions, str):
                    descriptions = [descriptions]
                elif not isinstance(descriptions, list):
                    descriptions = []
            except Exception as e:
                # print(f"  ⚠️ JSON parsing failed: {e}")
                # print(f"  Raw text: {generated_text[:100]}...")
                descriptions = []
            
            # If we got enough valid descriptions, return them
            if descriptions and len(descriptions) >= num_titles:
                return descriptions[:num_titles]
            else:
                # Not enough descriptions, retry
                wait_time = 10 if retry_count == 0 else RETRY_DELAY * (retry_count + 2)
                desc_count = len(descriptions) if descriptions else 0
                print(f"⚠️ Not enough descriptions returned ({desc_count}/{num_titles}, attempt {retry_count + 1}). Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                return await generate_descriptions_async(session, prompt, num_titles, retry_count + 1)
            
    except Exception as e:
        error_msg = str(e).lower()
        error_str = str(e)
        
        # Retry indefinitely for server errors
        if '500' in error_str or '503' in error_str or 'memory' in error_msg:
            wait_time = 10 if retry_count == 0 else RETRY_DELAY * (retry_count + 2)
            print(f"⚠️ Server error (attempt {retry_count + 1}). Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return await generate_descriptions_async(session, prompt, num_titles, retry_count + 1)
        
        # Retry indefinitely for connection errors
        elif 'connection' in error_msg or 'closed' in error_msg or 'timeout' in error_msg:
            wait_time = RETRY_DELAY * (retry_count + 1)
            print(f"⚠️ Connection error (attempt {retry_count + 1}). Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return await generate_descriptions_async(session, prompt, num_titles, retry_count + 1)
        
        # For other errors, also retry
        wait_time = RETRY_DELAY * (retry_count + 1)
        print(f"⚠️ Error (attempt {retry_count + 1}): {e}. Retrying in {wait_time} seconds...")
        await asyncio.sleep(wait_time)
        return await generate_descriptions_async(session, prompt, num_titles, retry_count + 1)

async def process_cluster_batch(session, cluster_name, titles_data, ground_truths, cluster_progress, output_data, semaphore):
    """Process a single batch for a cluster - retries until success"""
    async with semaphore:
        progress = cluster_progress[cluster_name]
        titles_to_process = progress['titles_to_process']
        processed = progress['processed_count']
        remaining = len(titles_to_process) - processed
        
        if remaining <= 0:
            return False

        batch_end = min(processed + BATCH_SIZE, len(titles_to_process))
        titles_batch = titles_to_process[processed:batch_end]
        
        # Find examples
        examples = find_cluster_examples(ground_truths, cluster_name)
        if not examples:
            # print(f"  ⚠️ Warning: No matching examples found for {cluster_name}")
            pass
            
        # Create prompt
        prompt = create_description_prompt(cluster_name, examples, titles_batch)
        
        # Generate - this will retry until we get enough descriptions
        # print(f"  Generating {len(titles_batch)} descriptions for {cluster_name}...")
        descriptions = await generate_descriptions_async(session, prompt, len(titles_batch))
        
        # generate_descriptions_async guarantees we get at least len(titles_batch) descriptions
        # But as a safety measure, ensure we have exactly the right number
        if len(descriptions) < len(titles_batch):
            print(f"  ⚠️ Warning: Got {len(descriptions)} descriptions for {len(titles_batch)} titles, padding...")
            while len(descriptions) < len(titles_batch):
                descriptions.append("")
        
        # Add to output - store new pages temporarily
        new_pages = []
        for title, description in zip(titles_batch, descriptions):
            new_pages.append({
                'title': title,
                'description': description
            })
            # Mark this title as having a description now
            progress['titles_with_descriptions'].add(title)
        
        # Append new pages to output
        output_data[cluster_name]['pages'].extend(new_pages)
            
        progress['processed_count'] = batch_end
        
        # Rebuild pages list in correct order (original titles order)
        all_titles = titles_data[cluster_name]
        ordered_pages = []
        existing_pages_dict = {p['title']: p for p in output_data[cluster_name]['pages']}
        for title in all_titles:
            if title in existing_pages_dict:
                ordered_pages.append(existing_pages_dict[title])
        
        output_data[cluster_name]['pages'] = ordered_pages
        
        total_processed = len(progress['titles_with_descriptions'])
        print(f"  ✓ {cluster_name}: Processed {total_processed}/{progress['total_titles']} titles")
        return True

async def main():
    # Check if model is available
    print("Checking GPT-OSS connection and model availability...")
    if not check_model_available():
        return
    
    # Load data files
    print("\nLoading data files...")
    titles_data = load_json(TITLES_FILE)
    ground_truths = load_json(GROUND_TRUTHS_FILE)
    
    print(f"Found {len(titles_data)} clusters in titles file")
    
    # Check if output file exists and resume from it
    print("\nChecking for existing output file...")
    existing_output = load_json(OUTPUT_FILE)
    
    # Initialize output structure
    output_data = {}
    
    # Track progress for each cluster
    cluster_progress = {}
    for cluster_name, titles in titles_data.items():
        existing_pages = []
        titles_with_descriptions = set()
        
        # If output file exists, check which titles already have descriptions
        if existing_output and cluster_name in existing_output:
            existing_cluster_data = existing_output[cluster_name]
            existing_pages = existing_cluster_data.get('pages', [])
            
            # Create a set of titles that already have descriptions
            for page in existing_pages:
                title = page.get('title')
                description = page.get('description', '').strip()
                if title and description:
                    titles_with_descriptions.add(title)
        
        # Filter titles to only those that need descriptions
        titles_to_process = [title for title in titles if title not in titles_with_descriptions]
        
        processed_count = len(titles) - len(titles_to_process)
        print(f"  {cluster_name}: {processed_count}/{len(titles)} titles already have descriptions, {len(titles_to_process)} remaining")
        
        cluster_progress[cluster_name] = {
            'total_titles': len(titles),
            'titles_to_process': titles_to_process,  # Only titles that need processing
            'processed_count': 0,  # Count of titles processed in this run
            'titles_with_descriptions': titles_with_descriptions,  # Set of titles that already have descriptions
            'pages': []
        }
        
        # Populate output_data with existing pages
        output_data[cluster_name] = {
            'pages': existing_pages.copy() if existing_pages else []
        }
    
    # Print summary
    total_processed = sum(len(p['titles_with_descriptions']) for p in cluster_progress.values())
    total_remaining = sum(len(p['titles_to_process']) for p in cluster_progress.values())
    total_titles = sum(p['total_titles'] for p in cluster_progress.values())
    print(f"\nResume summary: {total_processed}/{total_titles} titles already have descriptions, {total_remaining} remaining to process")
    
    # Round-robin processing with concurrency
    round_number = 0
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        while True:
            round_number += 1
            print(f"\n=== Starting Round {round_number} ===")
            
            tasks = []
            active_clusters = 0
            
            for cluster_name in titles_data.keys():
                progress = cluster_progress[cluster_name]
                titles_to_process = progress['titles_to_process']
                processed = progress['processed_count']
                # Check if there are remaining titles to process
                if processed < len(titles_to_process):
                    active_clusters += 1
                    tasks.append(process_cluster_batch(
                        session, 
                        cluster_name, 
                        titles_data, 
                        ground_truths, 
                        cluster_progress, 
                        output_data, 
                        semaphore
                    ))
            
            if active_clusters == 0:
                break
                
            print(f"Processing {len(tasks)} batches concurrently...")
            results = await asyncio.gather(*tasks)
            
            # Save progress after each round
            save_json(OUTPUT_FILE, output_data)
            print(f"  ✓ Saved progress for round {round_number}")
            
            # Summary
            if round_number % 5 == 0:
                print(f"\n=== Round {round_number} Summary ===")
                completed = sum(1 for c in cluster_progress.values() 
                              if c['processed_count'] >= len(c['titles_to_process']))
                print(f"  Completed clusters: {completed}/{len(titles_data)}")

    # Final save
    print("\n✓ All clusters processed!")
    save_json(OUTPUT_FILE, output_data)
    
    # Print final summary
    print("\n=== Final Summary ===")
    total_pages = 0
    for cluster_name, cluster_data in output_data.items():
        page_count = len(cluster_data['pages'])
        total_pages += page_count
    
    print(f"\nTotal: {len(output_data)} clusters, {total_pages} pages")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    asyncio.run(main())
