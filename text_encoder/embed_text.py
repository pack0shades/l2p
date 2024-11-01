import torch
import torchvision.models as models
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import json 

'''model = models.video.r3d_18(pretrained=True)

model.eval()'''
def get_text_embeddings_BERT(definitions):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    embeddings_dict = {}
    for definition in definitions.values():
        inputs = tokenizer(definition, return_tensors='pt')
        outputs = model(**inputs)
        text_embedding = outputs.last_hidden_state
        
        embeddings.append(text_embedding)
    print(f"shape of embeddings:{len(embeddings)}")
    return embeddings

def get_text_embeddings_SBERT(definitions):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = []
    embedding_dict = {}
    for label, definition in definitions.items():
        text_embedding = model.encode(definition, convert_to_tensor=True)
        print(f"this is the embedding size for {label}:::{text_embedding.shape}")
        if text_embedding.shape != torch.Size([384]):
            print("arre bhai kyaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  krr rha haiii")
        embeddings.append(text_embedding)
        embedding_dict[label] = text_embedding
    
    print(f"Shape of embeddings: {len(embeddings)}")
    return embedding_dict, embeddings

def load_definitions(path='text_definitions.json'):
    with open(path, 'r') as f:
        return json.load(f)


def save_embeddings(definitionsembedding, save_path='text_embeddings.json'):
    # Convert all tensor values in the dictionary to lists
    serializable_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value  ## Convert lists back to tensors
                            for key, value in definitionsembedding.items()}                          #tensor_dict = {key: torch.tensor(value) for key, value in embeddings_dict.items()}
                            
    
    with open(save_path, 'w') as f:
        json.dump(serializable_dict, f)


if __name__ == '__main__':
    #definitions = {'Abuse': 'cruel or inhumane treatment', 'Arrest': 'the act of apprehending (especially apprehending a criminal)', 'Arson': 'malicious burning to destroy property', 'Assault': 'close fighting during the culmination of a military attack', 'Burglary': 'entering a building unlawfully with intent to commit a felony or to steal valuable property', 'Explosion': 'a violent release of energy caused by a chemical or nuclear reaction', 'Fighting': 'the act of fighting; any contest or struggle', 'Normal Videos': 'Regular video content without criminal activity.', 'RoadAccidents': 'Accidents occurring on the road involving vehicles.', 'Robbery': 'larceny by threat of violence', 'Shooting': 'the act of firing a projectile', 'Shoplifting': 'the act of stealing goods that are on display in a store', 'Stealing': 'the act of taking something from someone unlawfully', 'Vandalism': 'willful wanton and malicious destruction of the property of others'}
    path = './text_encoder/data/text_expanded_definitions.json'
    save_path = './text_encoder/data/text_embeddings.json'
    definitions = load_definitions(path=path)
    BERT_embeddings = get_text_embeddings_BERT(definitions)
    SBERT_embeddings_dict, SBERT_embeddings = get_text_embeddings_SBERT(definitions)
    save_embeddings(SBERT_embeddings_dict,save_path=save_path)
    print (definitions)
    '''print(BERT_embeddings)'''
    print(f"saving embeddings at {save_path}")