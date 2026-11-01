import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import math
import re

def singularize_word(word):
    """
    Convert a single word from plural to singular form.
    """
    # Dictionary of common irregular plurals
    irregular_plurals = {
        'breeds': 'breed',
        'dogs': 'dog',
        'cats': 'cat',
        'people': 'person',
        'children': 'child',
        'geese': 'goose',
        'teeth': 'tooth',
        'feet': 'foot',
        'mice': 'mouse',
        'shelters': 'shelter',
        'families': 'family',
        'puppies': 'puppy',
        'kittens': 'kitten',
        'boxes': 'box',
        'foxes': 'fox',
        'bushes': 'bush',
        'glasses': 'glass',
        'wolves': 'wolf',
        'lives': 'life',
        'wives': 'wife',
        'knives': 'knife',
        'tzus': 'tzu',
        'frises': 'frise',
        'collies': 'collie',
        'poodles': 'poodle',
        'beagles': 'beagle',
        'poodle': 'poodle',  # Already singular
        'homes': 'home',
        'times': 'time',
    }
    
    # Check irregular plurals first
    if word in irregular_plurals:
        return irregular_plurals[word]
    
    # Apply regular English plural rules
    # Rule: words ending in 'ies' -> 'y'
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    # Rule: words ending in 'zes' -> 'z'
    elif word.endswith('zes') and len(word) > 3:
        return word[:-2]
    # Rule: words ending in 'ses' -> 's'
    elif word.endswith('ses') and len(word) > 3:
        return word[:-2]
    # Rule: words ending in 'xes', 'ches', 'shes' -> remove 'es'
    elif word.endswith(('xes', 'ches', 'shes')) and len(word) > 3:
        return word[:-2]
    # Rule: words ending in 'ves' -> 'f' or 'fe'
    elif word.endswith('ves') and len(word) > 3:
        return word[:-3] + 'f'
    # Rule: words ending in 'oes' -> 'o'
    elif word.endswith('oes') and len(word) > 3:
        return word[:-2]
    # Rule: standard 's' at end (but not always plural)
    elif word.endswith('s') and len(word) > 2:
        # Only remove 's' if it looks like a plural
        # Check if it ends with consonant + s
        if word[-2] not in 'aeiou' and word[-2:] not in ['ss', 'us', 'is']:
            return word[:-1]
    
    return word

def normalize_entity_text(text):
    """
    Intelligently normalize entity text for merging.
    Converts plural forms to singular and handles common variations.
    Returns a normalized key for grouping similar entities.
    Handles both single words and multi-word phrases.
    """
    # Convert to lowercase and strip whitespace
    normalized = text.lower().strip()
    
    # For multi-word phrases, singularize the last word (which is often the plural noun)
    if ' ' in normalized:
        words = normalized.split()
        # Singularize the last word
        last_word = words[-1]
        singularized_last = singularize_word(last_word)
        words[-1] = singularized_last
        normalized = ' '.join(words)
    else:
        # Single word - singularize it
        normalized = singularize_word(normalized)
    
    return normalized

def get_best_display_form(original_forms):
    """
    Select the best form to display from all original forms found.
    Prioritizes:
    1. Most frequently occurring form
    2. Longer forms (more complete)
    3. Capitalized forms over all lowercase
    """
    if not original_forms:
        return ""
    
    # Count occurrences of each form
    form_counts = defaultdict(int)
    for form in original_forms:
        form_counts[form] += 1
    
    # Sort by: count (descending), then by length (descending), then alphabetically
    best_form = sorted(
        form_counts.items(),
        key=lambda x: (-x[1], -len(x[0]), x[0])
    )[0][0]
    
    return best_form

def get_valid_json_files(folder_path):
    """Get all JSON files that contain valid entities"""
    valid_files = []
    
    print("\n" + "="*60)
    print("SCANNING FOR VALID JSON FILES...")
    print("="*60)
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return valid_files
    
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    if not json_files:
        print("❌ No JSON files found in the folder!")
        return valid_files
    
    print(f"📁 Found {len(json_files)} JSON file(s)\n")
    
    for idx, file in enumerate(json_files, 1):
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if entities exist and are valid
            entities = data.get('entities', [])
            if entities and len(entities) > 0:
                valid_files.append({
                    'number': idx,
                    'filename': file,
                    'path': file_path,
                    'entity_count': len(entities)
                })
                print(f"  [{idx}] ✓ {file} ({len(entities)} entities)")
            else:
                print(f"  [✗] {file} (no valid entities - skipped)")
        
        except json.JSONDecodeError:
            print(f"  [✗] {file} (invalid JSON - skipped)")
        except Exception as e:
            print(f"  [✗] {file} (error: {str(e)[:40]}...)")
    
    print(f"\n✅ Valid files: {len(valid_files)}/{len(json_files)}")
    return valid_files

def get_user_selection(valid_files):
    """Get user selection of files"""
    if not valid_files:
        return []
    
    print("\n" + "="*60)
    print("SELECT JSON FILES TO MERGE")
    print("="*60)
    print("Enter file numbers separated by commas (e.g., 1,2,3)")
    print("Or 'all' to select all files")
    print("Or 'quit' to cancel\n")
    
    while True:
        user_input = input("Enter selection: ").strip().lower()
        
        if user_input == 'quit':
            print("\n⚠️  Operation cancelled!")
            return []
        
        if user_input == 'all':
            selected = valid_files
            print(f"\n✅ Selected all {len(selected)} files")
            return selected
        
        try:
            numbers = [int(n.strip()) for n in user_input.split(',')]
            selected = [f for f in valid_files if f['number'] in numbers]
            
            if not selected:
                print("❌ No valid file numbers selected. Try again!")
                continue
            
            print(f"\n✅ Selected {len(selected)} file(s):")
            for f in selected:
                print(f"  • {f['filename']}")
            
            return selected
        
        except ValueError:
            print("❌ Invalid input! Please enter numbers separated by commas.")

def merge_entities(selected_files):
    """Merge entities from selected files"""
    print("\n" + "="*60)
    print("MERGING ENTITIES...")
    print("="*60)
    
    entity_map = defaultdict(lambda: {
        'count': 0,
        'relevance_sum': 0,
        'weightage_sum': 0,
        'label': None,
        'files': [],
        'counts': [],  # Track individual counts from each file
        'original_forms': []  # Track original text forms for intelligent display
    })
    
    # Statistics tracking
    total_entities = 0
    word_count_sum = 0
    heading_count_sum = 0
    para_count_sum = 0
    images_count_sum = 0
    file_count = 0
    
    for file_info in selected_files:
        try:
            with open(file_info['path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entities = data.get('entities', [])
            filename = file_info['filename']
            
            # Collect statistics from file
            word_count = data.get('word_count') or 0
            heading_count = data.get('heading_count') or 0
            para_count = data.get('para_count') or 0
            images_count = data.get('images_count') or 0
            
            word_count_sum += word_count
            heading_count_sum += heading_count
            para_count_sum += para_count
            images_count_sum += images_count
            file_count += 1
            
            print(f"\n📄 Processing: {filename}")
            print(f"   Entities found: {len(entities)}")
            
            for entity in entities:
                text = entity.get('text')
                label = entity.get('label')
                relevance = entity.get('relevance', 0)
                weightage = entity.get('weightage', 0)
                
                if text and text.lower() != 'n/a':
                    # Normalize the text to group similar entities
                    key = normalize_entity_text(text)
                    
                    if entity_map[key]['label'] is None:
                        entity_map[key]['label'] = label
                    
                    entity_map[key]['count'] += 1
                    entity_map[key]['relevance_sum'] += relevance
                    entity_map[key]['weightage_sum'] += weightage
                    entity_map[key]['files'].append(filename)
                    entity_map[key]['counts'].append(1)  # Track each occurrence
                    entity_map[key]['original_forms'].append(text.lower().strip())  # Track original forms
                    
                    total_entities += 1
            
            print(f"   ✓ Added to merge")
        
        except Exception as e:
            print(f"   ❌ Error reading file: {str(e)}")
    
    print(f"\n📊 Total entities processed: {total_entities}")
    print(f"🔀 Unique entities: {len(entity_map)}")
    
    # Calculate averages
    stats = {
        'total_files': file_count,
        'avg_word_count': round(word_count_sum / file_count, 2) if file_count > 0 else 0,
        'avg_heading_count': round(heading_count_sum / file_count, 2) if file_count > 0 else 0,
        'avg_para_count': round(para_count_sum / file_count, 2) if file_count > 0 else 0,
        'avg_images_count': round(images_count_sum / file_count, 2) if file_count > 0 else 0
    }
    
    return entity_map, stats

def create_output(entity_map, stats, output_folder):
    """Create output JSON with merged entities"""
    print("\n" + "="*60)
    print("GENERATING OUTPUT FILE...")
    print("="*60)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process entities for output
    merged_entities = []
    
    for text_lower, data in entity_map.items():
        avg_relevance = data['relevance_sum'] / data['count']
        avg_weightage = data['weightage_sum'] / data['count']
        
        # Calculate min and max counts from the counts list
        min_count = min(data['counts']) if data['counts'] else 0
        max_count = max(data['counts']) if data['counts'] else 0
        
        # Get the best display form from all original forms found
        display_text = get_best_display_form(data['original_forms'])
        
        merged_entities.append({
            'text': display_text.title() if len(display_text.split()) > 1 else display_text,
            'label': data['label'],
            'combined_count': data['count'],
            'min_count': min_count,
            'max_count': max_count,
            'average_relevance': round(avg_relevance, 4),
            'average_weightage': round(avg_weightage, 4),
            'found_in_files': data['files']
        })
    
    # Sort by average weightage (descending)
    merged_entities.sort(key=lambda x: x['average_weightage'], reverse=True)
    
    # Calculate adjusted weightage (weightage × competitor count multiplier)
    for entity in merged_entities:
        competitor_count = len(set(entity['found_in_files']))  # Count unique files
        entity['competitor_count'] = competitor_count
        
        # Apply capped multiplier based on competitor count
        if competitor_count >= 3:
            multiplier = 3
        elif competitor_count == 2:
            multiplier = 2
        else:  # competitor_count == 1
            multiplier = 1
        
        entity['competitor_multiplier'] = multiplier
        entity['adjusted_weightage'] = entity['average_weightage'] * multiplier
    
    # Re-sort by adjusted weightage (descending)
    merged_entities.sort(key=lambda x: x['adjusted_weightage'], reverse=True)
    
    # Calculate word_range for each entity
    # Step 1: Calculate 60% of average word count
    x_value = stats['avg_word_count'] * 0.60
    
    # Step 2: Sum all adjusted weightages to calculate probability
    total_adjusted_weightage = sum(e['adjusted_weightage'] for e in merged_entities)
    
    # Step 3: Calculate word_range for each entity using adjusted weightage
    for entity in merged_entities:
        # Calculate probability of this entity based on adjusted weightage
        probability = entity['adjusted_weightage'] / total_adjusted_weightage if total_adjusted_weightage > 0 else 0
        
        # Multiply probability by x_value
        word_range_value = probability * x_value
        
        # Round up to nearest integer
        entity['word_range'] = math.ceil(word_range_value)
    
    # Create output JSON with statistics
    output_data = {
        'merge_date': datetime.now().isoformat(),
        'total_files_processed': stats['total_files'],
        'average_statistics': {
            'avg_word_count': stats['avg_word_count'],
            'avg_heading_count': stats['avg_heading_count'],
            'avg_paragraph_count': stats['avg_para_count'],
            'avg_images_count': stats['avg_images_count'],
            'word_range_60_percent_value': round(x_value, 2),
            'total_adjusted_weightage': round(total_adjusted_weightage, 2)
        },
        'total_unique_entities': len(merged_entities),
        'total_entity_occurrences': sum(e['combined_count'] for e in merged_entities),
        'entities': merged_entities
    }
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'merged_entities_{timestamp}.json'
    output_path = os.path.join(output_folder, output_filename)
    
    # Write output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Output file created successfully!")
        print(f"   📁 Path: {output_path}")
        print(f"   📊 Files processed: {stats['total_files']}")
        print(f"   📈 Unique entities: {len(merged_entities)}")
        print(f"   📈 Total occurrences: {output_data['total_entity_occurrences']}")
        
        # Show statistics
        print(f"\n📊 AVERAGE STATISTICS:")
        print("-" * 60)
        print(f"   Average Word Count: {stats['avg_word_count']}")
        print(f"   Average Heading Count: {stats['avg_heading_count']}")
        print(f"   Average Paragraph Count: {stats['avg_para_count']}")
        print(f"   Average Images Count: {stats['avg_images_count']}")
        print(f"   60% of Avg Word Count (X): {round(x_value, 2)}")
        print(f"   Total Adjusted Weightage: {round(total_adjusted_weightage, 2)}")
        
        # Show top entities with word_range
        print(f"\n🏆 TOP 10 ENTITIES BY ADJUSTED WEIGHTAGE:")
        print("-" * 120)
        print(f"{'#':<3} {'Entity':<25} {'Competitors':<12} {'Multiplier':<12} {'Avg Weight':<12} {'Adj Weight':<12} {'Word Range':<12}")
        print("-" * 120)
        for i, entity in enumerate(merged_entities[:10], 1):
            print(f"{i:<3} {entity['text']:<25} {entity['competitor_count']:<12} "
                  f"{entity['competitor_multiplier']:<12} {entity['average_weightage']:<12.4f} "
                  f"{entity['adjusted_weightage']:<12.4f} {entity['word_range']:<12}")
        
        return output_path
    
    except Exception as e:
        print(f"\n❌ Error creating output file: {str(e)}")
        return None

def main():
    """Main function"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  NLP ENTITY MERGER - Multi-JSON Aggregator".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Get folder input from user
    folder_path = input("\nEnter folder path (or press Enter for current directory): ").strip()
    
    if not folder_path:
        folder_path = os.getcwd()
        print(f"Using current directory: {folder_path}")
    
    # Get valid JSON files
    valid_files = get_valid_json_files(folder_path)
    
    if not valid_files:
        print("\n❌ No valid files to process!")
        return
    
    # Get user selection
    selected_files = get_user_selection(valid_files)
    
    if not selected_files:
        return
    
    # Merge entities and collect statistics
    entity_map, stats = merge_entities(selected_files)
    
    if not entity_map:
        print("\n❌ No entities found to merge!")
        return
    
    # Create output
    output_folder = os.path.join(folder_path, 'merged_output')
    output_path = create_output(entity_map, stats, output_folder)
    
    if output_path:
        print("\n" + "="*60)
        print("✨ MERGE COMPLETED SUCCESSFULLY!")
        print("="*60)

if __name__ == '__main__':
    main()
