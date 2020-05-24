import numpy as np
import re
from statistics import mean
import pickle

class Ranking:
    """
    Class that handles ranking algorithms.

    Attributes
    -----------
    data:       list of strings
        Set of documents.
    num_documents:  int
        Number of documents in the corpus
    ids:       list of strings
        List with all the paper_id
    document_frequencies:   dict
        number of documents containing a given term. term -> number of documents
    term_frequencies:       dict of dicts
        dict of dicts of term frequencies within a document. paper_id -> term -> frequency
    document_length:        dict
        number of words in a document. paper_id -> document_length
    avg_length:             int
        average number of words in a document.


    Methods
    --------
    _get_dictionaries:
        Loads the dictionaries needed to compute the scores.
    get_tfidf_scores:
        Computes the similarity between documents.
    """

    def __init__(self, data='texts', path = 'Data/ranking_dict'):
        """
        :param data: whether you want to look through texts or articles. ('abstracts' or 'texts')
        """

        if not data in ['texts', 'abstracts']:
            raise Exception('The data parameter must be "texts" or "abstracts"')

        self.data = data
        self._get_dictionaries(path)
        self.num_documents = len(self.document_frequencies)
        self.ids = list(self.term_frequencies.keys())

    def _get_dictionaries(self, path):
        """
        loads the dictionaries needed to calculate the scoring functions.
        """
        
        if self.data == 'abstracts':
            with open(path + '/document_frequencies_abstract_v5.p', 'rb') as fp:
                self.document_frequencies = pickle.load(fp)

            with open(path + '/term_frequencies_abstract_v5.p', 'rb') as fp:
                self.term_frequencies = pickle.load(fp)

            with open(path + '/document_length_abstract_v5.p', 'rb') as fp:
                self.document_length = pickle.load(fp)
        else:
            with open(path + '/document_frequencies_text_v5.p', 'rb') as fp:
                self.document_frequencies = pickle.load(fp)

            with open(path + '/term_frequencies_text_v5.p', 'rb') as fp:
                self.term_frequencies = pickle.load(fp)

            with open(path + '/document_length_text_v5.p', 'rb') as fp:
                self.document_length = pickle.load(fp)

        self.avg_length = mean(self.document_length.values())


    def get_tfidf_scores(self, query, paper_ids = None):
        """
        Given a query, computes the score of every document in the corpus
        :param query:
        :return: list of (index, score)
        """
        length = {}
        scores = {}
        if paper_ids is None:
            self.ids = paper_ids

        for id in self.ids:
            scores[id] = 0
            length[id] = 0

        for term in re.findall(r"[\w']+|[.,!?;]", query.strip()):
            term = term.lower()
            if not term in self.document_frequencies:
                continue
            df = self.document_frequencies[term]
            wq = np.log(self.num_documents/df)
            for id in self.ids:
                document_dict = self.term_frequencies[id]
                if not term in document_dict:
                    scores[id] += 0
                    continue
                tf = document_dict[term]
                length[id] += tf**2
                wd = 1 + np.log(tf)
                scores[id] += wq * wd
        for id in self.ids:
            if length[id] == 0:
                continue
            scores[id] /= np.sqrt(length[id])

        return scores

    def get_bm25_scores(self, query, paper_ids = None):
        """
        Given a query, computes the score of every document in the corpus using BM25+
        :param query:
        :return: list of (index, score)
        """
        k = 1.5
        b = 0.75
        scores = {}
        if not paper_ids is None:
            self.ids = paper_ids.values

        for id in self.ids:
            scores[id] = 0

        for term in re.findall(r"[\w']+|[.,!?;]", query.strip()):
            term = term.lower()
            if not term in self.document_frequencies:
                continue
            df = self.document_frequencies[term]
            idf = np.log((self.num_documents - df + 0.5) / (df + 0.5))
            for id in self.ids:
                document_dict = self.term_frequencies[id]
                if not term in document_dict:
                    scores[id] += 0
                    continue
                tf = document_dict[term]
                wd = ((tf * (k+1)) / (tf + k*(1-b+b*self.document_length[id]/self.avg_length))) + 1
                scores[id] += idf * wd

        return scores

    def most_similar(self, query, df, k=10, func = 'bm25', data = 'abstract'):
        """
        Given a query it returns the k most relevant documents. It returns the abstract of the document.
        :param query:
        :param df: dataframe with columns [paper_id, title, abstract, text]
        :param k: <int> number of most relevant documents to return. (default: 10)
        :param func: whether you want to use bm25f or tfidf scoring. (defualt: bm25)
        :param data: whether you want the abstract or text of the most similar. (default: abstract)
        :return: list of strings
        """
        if func == 'bm25':
            scores = self.get_bm25_scores(query, paper_ids=df.paper_id)
        else:
            scores = self.get_tfidf_scores(query)
        most_similar = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        ids = [a for a,b in most_similar[:k]]

        if data == 'abstract':
            return list(df.loc[df['paper_id'].isin(ids)].abstract.values)
        else:
            return list(df.loc[df['paper_id'].isin(ids)].text.values)


    def extend_query(self, query, embeddings, k=3):
        """
        Extend the original query by adding the top-k similar embeddings of each word.
        TODO Adapt k depending on TFIDF score
        :param query: original query. <str>
        :param embeddings: Embeddings object.
        :param k: number of embeddings to add per word. <int>
        :return: extended query. <str>
        """
        new_query = query.split()
        for term in query.split():
            similar_words = [word for word, sim in embeddings.most_similar(word, k= k)]
            new_query += similar_words

        return ' '.join(new_query)


