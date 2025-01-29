import json 
from tqdm import tqdm
import re
import pandas as pd 

JSON_LOCATION = "Data/AnchorTerms.json"

# Load Anchor terms json 
with open(JSON_LOCATION, 'r') as file:
    anchor_terms = json.load(file)


print(len(anchor_terms))    # 3849

# load the original text as a string 
data_location = 'Data/TinyShakespeare.txt'
with open(data_location, 'r') as file:
    text = file.read()

# Convert the string into blocks
block_size = 256
blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]

replacements = []

processed_texts = []
    
# Sort anchor terms by length (longest first) to prevent partial replacements
sorted_anchors = sorted(anchor_terms.keys(), key=len, reverse=True)

# Create regex pattern with word boundaries
pattern = r'\b(' + '|'.join(map(re.escape, sorted_anchors)) + r')\b'

for text in tqdm(blocks):
    text = re.sub(pattern, lambda match: anchor_terms[match.group(0)], text)
    processed_texts.append(text)
    

print("Original Text : \n", blocks[0])    
print("Processed Text : \n", processed_texts[0])

dataset = {"text": blocks, "label": processed_texts}    

pd.DataFrame(dataset).to_csv("Data/processed_dataset.csv", index=False)

