"""Script to add category tags to JSON files using GPT-4.

This script processes all JSON files in a specified directory, extracts unique 'source' texts,
generates tags using GPT-4, reduces these tags to 8 categories, and assigns these categories
to each item in the JSON files. The updated data is written back to new JSON files prefixed
with 'output_'.

Usage:
    python ./data_collection_scripts/tag_sources.py --input_dir ./assign/batch_5_src_no_sports/results

Example:
    To process JSON files in the directory './assign/batch_5_src_no_sports/results', run:
    python ./data_collection_scripts/tag_sources.py --input_dir ./assign/batch_5_src_no_sports/results

Requirements:
    - Python 3.x
    - OpenAI Python library (install via `pip install openai`)
    - An OpenAI API key set as an environment variable 'OPENAI_API_KEY' or assigned directly in the script.

The script follows the Google Python Style Guide.

"""

import argparse
import json
import os
import sys
from hashlib import sha256

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up your OpenAI API key


def strip_quotes(text):
    """Strips quotes from the text.

    Args:
        text: The text to strip quotes from.

    Returns:
        The text with quotes removed.
    """
    return text.strip().strip("'")


def get_tags_from_gpt4(source_text):
    """Generates 2 unique tags for the source text using GPT-4.

    Args:
        source_text: The text to generate tags for.

    Returns:
        A list of up to 2 tags as strings.
    """
    prompt = (
        "Given the following text, provide 1 or 2 unique category tags that best describe it. "
        "Examples of tags are 'sports', 'news', 'technology', 'entertainment', etc.\n\n"
        f"Text:\n{source_text}\n\nTags:"
    )
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0)
        tags_text = response.choices[0].message.content.strip()
        tags = [strip_quotes(tag) for tag in tags_text.split(',') if tag.strip()]
        print(f"Tags generated for source '{source_text[0:40]}...': {tags}")
        return tags[:2]  # Ensure only 2 tags
    except Exception as e:
        print(f"Error generating tags for source: {e}")
        return []


def get_categories_from_gpt4(unique_tags):
    """Reduces the initial tags to 8 categories using GPT-4.

    Args:
        unique_tags: A set of unique initial tags.

    Returns:
        A list of 8 categories as strings.
    """
    tags_list = ', '.join(unique_tags)
    prompt = (
        "Given the following list of tags:\n"
        f"{tags_list}\n\n"
        "Group these tags into 8 categories by merging similar tags. "
        "Provide only the list of 8 categories, separated by commas."
    )
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0)
        categories_text = response.choices[0].message.content.strip()
        categories = [cat.strip() for cat in categories_text.split(',') if cat.strip()]
        return categories[:8]  # Ensure only 8 categories
    except Exception as e:
        print(f"Error generating categories: {e}")
        return []


def map_tags_to_categories(initial_tags, categories):
    """Maps each initial tag to one of the 8 categories using GPT-4.

    Args:
        initial_tags: A set of initial tags.
        categories: A list of 8 categories.

    Returns:
        A dictionary mapping tags to categories.
    """
    tags_list = ', '.join(initial_tags)
    categories_list = ', '.join(categories)
    prompt = (
        f"Given the list of tags:\n{tags_list}\n\n"
        f"And the list of categories:\n{categories_list}\n\n"
        "For each tag, assign it to the most appropriate category. "
        "Provide the mappings in the format:\n"
        "Tag: Category"
    )
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0)
        content = response.choices[0].message.content.strip()
        tag_to_category = {}
        print(f"Response from GPT-4:\n{content}")
        lines = content.splitlines()
        for line in lines:
            if ':' in line:
                tag, category = line.split(':', 1)
                tag = tag.strip()
                category = category.strip()
                if category in categories:
                    tag_to_category[tag] = category
                else:
                    tag_to_category[tag] = 'Other'
        print(f"Tags mapped to categories: {tag_to_category}")
        return tag_to_category
    except Exception as e:
        print(f"Error mapping tags to categories: {e}")
        return {}


def main():
    """Main function to process JSON files and add category tags."""
    parser = argparse.ArgumentParser(description='Add category tags to JSON files using GPT-4.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='The directory containing JSON files to process.')
    args = parser.parse_args()

    input_dir = args.input_dir

    if not os.path.isdir(input_dir):
        print(f"The directory {input_dir} does not exist.")
        sys.exit(1)

    # Step 1: Extract unique sources from all JSON files
    json_files = [file for file in os.listdir(input_dir) if file.endswith('.json')]
    unique_sources = set()
    data_files = {}  # Store data from each file

    for file in json_files:
        file_path = os.path.join(input_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file}: {e}")
                continue
            data_files[file] = data
            for item in data:
                source_text = item.get('source', '')
                if source_text:
                    unique_sources.add(source_text)
    print(f"Total unique sources: {len(unique_sources)}")

    # Step 2: Generate 2 unique tags for each source using GPT-4
    source_to_tags = {}
    cache = {}
    for source in unique_sources:
        # Use a hash for caching to avoid duplicate API calls
        source_hash = sha256(source.encode('utf-8')).hexdigest()
        if source_hash in cache:
            tags = cache[source_hash]
        else:
            tags = get_tags_from_gpt4(source)
            cache[source_hash] = tags
        source_to_tags[source] = tags

    # Collect all unique initial tags
    initial_tags = set()
    for tags in source_to_tags.values():
        initial_tags.update(tags)
    print(f"Total unique initial tags: {len(initial_tags)}")

    # Step 3: Reduce initial tags to 8 categories using GPT-4
    categories = get_categories_from_gpt4(initial_tags)
    print(f"Reduced to categories: {categories}")

    # Step 4: Map initial tags to categories using GPT-4
    tag_to_category = map_tags_to_categories(initial_tags, categories)

    # Map source's initial tags to categories
    source_to_categories = {}
    for source, tags in source_to_tags.items():
        categories_for_source = []
        for tag in tags:
            category = tag_to_category.get(tag, 'Other')
            print(f"Mapping tag '{tag}' to category '{category}'")
            categories_for_source.append(category)
        # Remove duplicates and limit to 1 categories
        categories_for_source = list(set(categories_for_source))[:1]
        source_to_categories[source] = categories_for_source

    # Step 5: Add tags to each item in the JSON files
    for file, data in data_files.items():
        for item in data:
            source_text = item.get('source', '')
            if source_text and source_text in source_to_categories:
                categories_for_source = source_to_categories[source_text]
                item['tags'] = categories_for_source
            else:
                item['tags'] = []
        # Write the modified data to a new file
        output_file = os.path.join(input_dir, f"output_{file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tags added to file: {output_file}")

    print("All files have been processed.")


if __name__ == '__main__':
    main()
