import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import pickle

def main():
    input_path = 'data/cleansed_data.pkl'
    model_output_path = 'model_object/word2vec.model'
    embedding_output_path = 'data/data_with_embedding.pkl'
    df = pickle.load(open(input_path, 'rb'))

    # flatten the nested feature id list
    df['feature_id'] = df['feature_id'].apply(lambda x: [e for sublist in x for e in sublist])

    # train the Word2Vec model on the list of feature ids (i.e. diagnosis codes)
    embedding_dim = 100
    model = Word2Vec(df['feature_id'], vector_size=embedding_dim, window=5, min_count=1, workers=4)
    model.save(model_output_path)

    # create embeddings
    embeddings = []
    for features in df['feature_id']:
        # get the Word2Vec embeddings for each code in the document (patient EHRs)
        feature_embeddings = [model.wv[feature] for feature in features if feature in model.wv]
        if len(feature_embeddings) > 0:
            # average the embeddings to create a single vector representation of the patient EHR
            ehr_embedding = np.mean(feature_embeddings, axis=0)
            embeddings.append(ehr_embedding)
        else:
            embeddings.append(np.zeros(embedding_dim))

    df['embedding'] = embeddings

    pickle.dump(df, open(embedding_output_path, 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()







