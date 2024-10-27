import json
from nltk.corpus import wordnet as wn


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
            if label == "Normal Videos":
                lables_definitions[label] = "Regular video content without criminal activity."
            elif label == "RoadAccidents":
                lables_definitions[label] = "Accidents occurring on the road involving vehicles."
            else:
                lables_definitions[label] = get_wordnet_definition(label)
        return lables_definitions


if __name__ == '__main__':
    # Save definitions to a JSON file
    file = './data/ucf_crime_labels.json'
    definitions = get_definition(file)
    print (definitions)