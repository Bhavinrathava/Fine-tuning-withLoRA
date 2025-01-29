import json
import os
import pandas as pd
from util import call_llm, add_to_dictionary, extract_json_from_response
from tqdm import tqdm

def load_prompt(blob):
    return f'''
    You are a helpful assistant that is expert at identifying the most important terms in a block of text. 
    Let's call these Anchor terms. 
    Anchor Terms are those that are most important to the meaning of the text. 
    These include unique names for People / Objects / Places (fictional) that give the unique meaning to the text and give the text its unique meaning / symbolism. 
    
    You are tasked with identifying the anchor terms in the following block of text and identifying an appropriate replacement for the anchor term in such a way that it retains the meaning of the text without referring to the actual word.

    You are expected to return only a dictionary of the anchor terms and their replacements similar to this : 

    "<AnchorTerm1>": "<Replacement1>", "<AnchorTerm2>": "<Replacement2>", "<AnchorTerm3>": "<Replacement3>"

    DO NOT RETURN ANYTHING OTHER THAN THE ACTUAL ANCHOR TERM DICTIONARY.

    Here is the block of text you need to process:

    {blob}
'''

def save_checkpoint(anchor_terms, checkpoint_file, processed_blocks):
    """
    Save checkpoint data to a file.
    """
    checkpoint_data = {
        "anchor_terms": anchor_terms,
        "processed_blocks": processed_blocks
    }
    with open(checkpoint_file, 'w') as file:
        json.dump(checkpoint_data, file)


def load_checkpoint(checkpoint_file):
    """
    Load checkpoint data from a file.
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            checkpoint_data = json.load(file)
        return checkpoint_data
    return {"anchor_terms": {}, "processed_blocks": 0}


def generate_dataset():
    # Initialize checkpoint file
    checkpoint_file = 'Data/checkpoint.json'
    checkpoint_data = load_checkpoint(checkpoint_file)

    # Load anchor terms and last processed block from checkpoint
    anchor_terms = checkpoint_data.get("anchor_terms", {})
    processed_blocks = checkpoint_data.get("processed_blocks", 0)

    # Load the text file into a string
    data_location = 'Data/TinyShakespeare.txt'
    with open(data_location, 'r') as file:
        text = file.read()

    # Convert the string into blocks
    block_size = 256
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]

    # Resume from the last processed block
    print(f"Resuming from block {processed_blocks}")
    for i in tqdm(range(processed_blocks, len(blocks))):
        block = blocks[i]
        prompt = load_prompt(block)
        response = call_llm(prompt)['response']

        try:
            response = extract_json_from_response(response)
            if response:
                anchor_terms = add_to_dictionary(anchor_terms, response)
        except Exception as e:
            print(f"Error processing block {i}: {e}")
            continue

        # Save checkpoint every 20 blocks
        if (i + 1) % 20 == 0:
            save_checkpoint(anchor_terms, checkpoint_file, i + 1)
            print(f"Checkpoint saved at block {i + 1}")

    # Save the final dictionary
    dictionary_location = 'Data/AnchorTerms.json'
    with open(dictionary_location, 'w') as file:
        json.dump(anchor_terms, file)

    print("Anchor Terms Generated and saved to file")

    # Replace the original text with the new text and create a [text, label] pair
    processed_text = []

    for block in blocks:
        text = block
        for key, value in anchor_terms.items():
            text = text.replace(key, value)
        processed_text.append([block, text])

    # Convert the [text, label] pair into a dataset
    dataset = []
    for text, label in processed_text:
        dataset.append({'text': text, 'label': label})

    dataset = pd.DataFrame(dataset)

    # Save the dataset to a new file
    dataset_location = 'Data/ProcessedDataset.csv'
    dataset.to_csv(dataset_location, index=False)

    print("Dataset generated and saved.")
    # Clean up the checkpoint file once done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed.")


if __name__ == '__main__':
    generate_dataset()
