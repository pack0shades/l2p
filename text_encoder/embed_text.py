import torch
import torchvision.models as models
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

'''model = models.video.r3d_18(pretrained=True)

model.eval()'''
def get_text_embeddings_BERT(definitions):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
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
    for definition in definitions.values():
        text_embedding = model.encode(definition, convert_to_tensor=True)
        embeddings.append(text_embedding)
    
    print(f"Shape of embeddings: {len(embeddings)}")
    return embeddings


if __name__ == '__main__':
    definitions = {'Abuse': 'cruel or inhumane treatment', 'Arrest': 'the act of apprehending (especially apprehending a criminal)', 'Arson': 'malicious burning to destroy property', 'Assault': 'close fighting during the culmination of a military attack', 'Burglary': 'entering a building unlawfully with intent to commit a felony or to steal valuable property', 'Explosion': 'a violent release of energy caused by a chemical or nuclear reaction', 'Fighting': 'the act of fighting; any contest or struggle', 'Normal Videos': 'Regular video content without criminal activity.', 'RoadAccidents': 'Accidents occurring on the road involving vehicles.', 'Robbery': 'larceny by threat of violence', 'Shooting': 'the act of firing a projectile', 'Shoplifting': 'the act of stealing goods that are on display in a store', 'Stealing': 'the act of taking something from someone unlawfully', 'Vandalism': 'willful wanton and malicious destruction of the property of others'}
    BERT_embeddings = get_text_embeddings_BERT(definitions)
    SBERT_embeddings = get_text_embeddings_SBERT(definitions)
    print (definitions)
    '''print(BERT_embeddings)'''
    print(SBERT_embeddings)