from embeddings import Embeddings
import numpy as np
from nltk import tokenize
import pickle
import re
from statistics import mean


class Embedding_retrieval:
    """
    Information Retrieval class using embeddings trained on our dataset.

    Attributes
    ------------

    """

    def __init__(self, path = 'Data/wordvectors.kv', embeddings = None):

        if not embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = Embeddings(path=path)

        with open('Data/ranking_dict/document_frequencies_text_v8.p', 'rb') as fp:
            self.document_frequencies = pickle.load(fp)

        with open('Data/ranking_dict/term_frequencies_text_v8.p', 'rb') as fp:
            self.term_frequencies = pickle.load(fp)

        with open('Data/ranking_dict/document_length_text_v8.p', 'rb') as fp:
            self.document_length = pickle.load(fp)

        self.num_documents = len(self.term_frequencies)
        self.avg_length = mean(self.document_length.values())


    def get_closest_sentence_weighted(self, query, id, doc, topk = 3):
        """
        Given a query and a document (with the id) returns the top-k similar sentences within the document.
        :param query:
        :param id: paper_id for the document.
        :param doc: document text. <str>
        :param topk: number of sentences to return.
        :return: list of sentences.
        """
        k = 1.5
        b = 0.75
        weights = []
        for term in re.findall(r"[\w']+|[.,!?;]", query):
            term = term.lower()
            if not term in self.document_frequencies:
                continue
            df = self.document_frequencies[term]
            idf = np.log((self.num_documents - df + 0.5) / (df + 0.5))
            document_dict = self.term_frequencies[id]
            if not term in document_dict:
                weights.append(0)
                continue
            tf = document_dict[term]
            wd = ((tf * (k + 1)) / (tf + k * (1 - b + b * self.document_length[id] / self.avg_length))) + 1
            weights.append(idf*wd)

        query_embedding = self.weighted_embedding(query, weights)
        doc_embedding = []
        tokenized_sent = tokenize.sent_tokenize(doc)
        for sent in tokenized_sent:
            doc_embedding.append(self.sent_embedding(sent))

        scores = []
        query_norm = np.linalg.norm(query_embedding)
        for i, emb in enumerate(doc_embedding):
            sent_norm = np.linalg.norm(emb)
            if sent_norm == 0:
                scores.append((i, 0))
            else:
                scores.append((i, np.dot(emb,query_embedding)/(sent_norm*query_norm)))

        scores = sorted(scores, key=lambda x:x[1], reverse=True)

        most_similar = []
        similarities = []
        for index, similarity in scores[:topk]:
            most_similar.append(tokenized_sent[index])
            similarities.append(similarity)

        return zip(most_similar,similarities)

    def get_closest_sentence(self, query, id, doc, topk = 3):
        """
        Given a query and a document (with the id) returns the top-k similar sentences within the document.
        :param query:
        :param id: paper_id for the document.
        :param doc: document text. <str>
        :param topk: number of sentences to return.
        :return: list of sentences.
        """

        query_embedding = self.weighted_embedding(query)
        doc_embedding = []
        tokenized_sent = tokenize.sent_tokenize(doc)
        for sent in tokenized_sent:
            doc_embedding.append(self.sent_embedding(sent))

        scores = []
        query_norm = np.linalg.norm(query_embedding)
        for i, emb in enumerate(doc_embedding):
            sent_norm = np.linalg.norm(emb)
            if sent_norm == 0:
                scores.append((i, 0))
            else:
                scores.append((i, np.dot(emb,query_embedding)/(sent_norm*query_norm)))

        scores = sorted(scores, key=lambda x:x[1], reverse=True)

        most_similar = []
        similarities = []
        for index, similarity in scores[:topk]:
            most_similar.append(tokenized_sent[index])
            similarities.append(similarity)

        return zip(most_similar,similarities)


    def weighted_embedding(self, query, weights=None):
        """
        Returns the embeddings of the query.
        :param query:
        :param weights: bm25f scores for each word.
        :return: weighted embeddings (for now weights are not being used)
        """
        #sum_weights = sum(weights)
        #weights = [w/sum_weights for w in weights]
        embeddings = []
        for term in re.findall(r"[\w']+|[.,!?;]", query):
            term = term.lower()
            if term in self.embeddings.wv:
                embeddings.append(self.embeddings.wv[term])
            else:
                continue
        ones = np.ones(len(embeddings))/len(embeddings)
        return np.dot(ones, embeddings)

    def sent_embedding(self, sentence):
        """
        Returns embedding of a sentence. Average of word embeddings.
        :param sentence: sentence to return the embedding.
        :return: embedding vector. <np.array>
        """
        embeddings = None
        count = 0
        for term in re.findall(r"[\w']+|[.,!?;]", sentence):
            term = term.lower()
            if term in self.embeddings.wv.vocab:
                if embeddings is None:
                    embeddings = self.embeddings.get_embedding(term)
                else:
                    embeddings = np.add(embeddings, self.embeddings.get_embedding(term))
                count += 1
            else:
                pass
                #print(term)
        if embeddings is None:
            #print('Embeddings none for sentence: {}'.format(sentence))
            return np.zeros(100)
        return embeddings/count

