from wordnet import get_definition
from description_generation import generate_expanded_definition
from embed_text import get_text_embeddings_BERT, get_text_embeddings_SBERT

def generate_final_text_embedding_BERT(file_path):
    word_def = get_definition(file_path)
    exp_def = generate_expanded_definition(word_def)
    print(exp_def)
    print(type(exp_def))
    bert_embeddings = get_text_embeddings_BERT(exp_def)
    return bert_embeddings

def generate_final_text_embedding_SBERT(file_path):
    word_def = get_definition(file_path)
    exp_def = generate_expanded_definition(word_def)
    print(exp_def)
    print(type(exp_def))
    bert_embeddings = get_text_embeddings_SBERT(exp_def)
    return bert_embeddings

if __name__ == '__main__':
    file = './data/ucf_crime_labels.json'
    BERT_embeddings = generate_final_text_embedding_BERT(file)
    SBERT_embeddings = generate_final_text_embedding_SBERT(file)
    print(BERT_embeddings)
    print(SBERT_embeddings)
    