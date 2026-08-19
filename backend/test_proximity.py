#!/usr/bin/env python3
"""Test proximity checking logic"""

# Test data from the JSON file
title = "7 Big Dog Breeds With Calm Temperament For Easy Living"
description = "Meet giant dog breeds with calm, gentle temperaments. Perfect for families and first-time owners, these gentle giants bring loyalty and easygoing charm."
content = "7 Big Dog Breeds With Calm Temperament For Easy Living\nMeet giant dog breeds with calm, gentle temperaments. Perfect for families and first-time owners, these gentle giants bring loyalty and easygoing charm. Whether you're looking for a large canine companion that won't bounce off the walls with endless energy, or a pet that enjoys relaxation as much as playtime, this guide will help you find your ideal match. Let's explore seven of the calmest giant dog breeds that make excellent family pets.\n\nGreat Dane\nGreat Danes are often referred to as gentle giants, and for good reason. Despite their imposing size, these magnificent dogs are known for their calm, friendly, and patient temperament. They're excellent family dogs that are good with children and other pets. Great Danes have a relatively low energy level compared to other large breeds, preferring a comfortable couch to extensive outdoor activities. Their lifespan, though shorter than smaller breeds, doesn't diminish their value as loving companions.\n\nSaint Bernard\nSaint Bernards are another example of large, calm dog breeds. Originally bred as rescue dogs in the Swiss Alps, these gentle giants are known for their patient, friendly, and tolerant nature. They're great family dogs that are excellent with children and require moderate exercise. Their calm temperament and protective instincts make them wonderful home guardians.\n\nGreyhound\nWhile greyhounds are known for their speed, they're actually quite calm and gentle at home. Often called \"40 mph couch potatoes,\" greyhounds enjoy lounging and are content with moderate exercise. They have a calm, independent nature and are typically good with children and other pets."

print("=" * 80)
print("SOURCE DATA:")
print("=" * 80)
print(f"Title: {title}")
print(f"\nDescription: {description[:100]}...")
print(f"\nContent (first 200 chars): {content[:200]}...")

print("\n" + "=" * 80)
print("TESTING ENTITY MATCHING:")
print("=" * 80)

# Test entities from the JSON
test_entities = [
    "giant breeds",
    "breeds calm temperaments",
    "gentle giants",
    "Great Dane",
]

for entity in test_entities:
    entity_words = entity.lower().split()
    print(f"\nEntity: '{entity}' -> Words: {entity_words}")

    # Check in title
    title_words = title.lower().split()
    for i in range(len(title_words) - len(entity_words) + 1):
        if title_words[i:i+len(entity_words)] == entity_words:
            print(f"  + Found in title at position {i}")
            break
    else:
        print(f"  - Not in title")

    # Check in description
    desc_words = description.lower().split()
    for i in range(len(desc_words) - len(entity_words) + 1):
        if desc_words[i:i+len(entity_words)] == entity_words:
            print(f"  + Found in description at position {i}")
            break
    else:
        print(f"  - Not in description")

    # Check in first 10%
    content_words = content.split()
    first_10_percent = max(1, len(content_words) // 10)
    first_10_text = content_words[:first_10_percent]
    for i in range(len(first_10_text) - len(entity_words) + 1):
        if first_10_text[i:i+len(entity_words)] == [w.lower() for w in first_10_text[i:i+len(entity_words)]]:
            if first_10_text[i:i+len(entity_words)] == entity_words:
                print(f"  + Found in first 10% ({len(first_10_text)} words)")
                break
    else:
        print(f"  - Not in first 10% (only {len(first_10_text)} words)")

print("\n" + "=" * 80)
