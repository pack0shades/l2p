import json
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')


def get_wordnet_definition(label):
    synsets = wn.synsets(label.lower())
    if synsets:
        # Return the first definition if available
        return synsets[0].definition()
    else:
        return "Definition not found in WordNet"


def get_definition(file_path):
    # Fetch definitions for each label
    with open(file_path, 'r') as f:
        data = json.load(f)
        labels = data['labels']
        lables_definitions = {}
        for label in labels:
            lables_definitions[label] = get_wordnet_definition(label)
        return lables_definitions
    
def save_definitions(definitions, save_path='./text_encoder/data/text_definitions.json'):
    with open(save_path, 'w') as f:
        json.dump(definitions, f)


if __name__ == '__main__':
    # Save definitions to a JSON file
    path = 'text_definitions.json'
    file = './text_encoder/data/classlabels.json'
    definitions = get_definition(file)
    print (definitions)
    save_definitions(definitions)
    print(f"saved definition successfully at {path}")