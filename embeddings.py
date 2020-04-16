from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import string
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Embeddings():
    """
    Class that handles word embeddings
    """

    def __init__(self, df=None, path = None, column = 'text', embedding_dim = 100, embedding_window = 5, min_count = 100):
        """
        :param df:  dataframe with columns ['paper_id', 'title', 'abstract','text']
        :param path: path to pretrained embeddings.
        :param column: either abstract or text (default: text) <str>
        :param embedding_dim: word vectors dimension.
        :param embedding_window: window size to compute embeddings.
        :param min_count: minimum frequency of a word in the corpus to consider it.
        """
        if path is None and df is None:
            raise Exception('You must provide either a df or a path')
        if df is None:
            self.load(path)
        else:
            model = self._get_embeddings(df, column, embedding_dim, embedding_window, min_count)
            self.wv = model.wv

    def _get_embeddings(self, df, column = 'text', embedding_dim = 100, embedding_window = 5, min_count = 100):
        """
        Returns gensim Word2Vec model trained on the column given.
        """

        l = []
        for des in df[column]:
            des = des.translate(des.maketrans({key: ' ' for key in string.punctuation}))
            des = des.lower()
            l.append(re.findall(r"[\w']+|[.,!?;]", des.strip()))
        model = Word2Vec(l, workers=4, size=embedding_dim, min_count=min_count, window=embedding_window,
                         sample=1e-3, sg = 1)

        return model

    def most_similar(self, word, k):
        """
        Get the k nearest neighbors to the word.
        :param word: word to find nearest neighbors.
        :param k: number of neighbors to return
        :return: list of (word, similarity)
        """
        return self.wv.most_similar(word,  topn = k)

    def get_embedding(self,word):
        """
        Method to obtain the embedding vector of a word.
        :param word: word to obtain embedding.
        :return: word vector <np.array>
        """
        return self.wv[word]

    def save(self, path = 'Data/wordvectors.kv'):
        self.wv.save(path)

    def load(self, path = 'Data/wordvectors.kv'):
        self.wv = KeyedVectors.load(path)

    def plot_embedding(self):
        vocab = list(self.wv.vocab)
        X = self.wv.vocab

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(df['x'], df['y'])

        for word, pos in df.iterrows():
            ax.annotate(word, pos)

        plt.show()


