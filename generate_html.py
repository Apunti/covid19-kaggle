from create_html import *
from get_result import *
from ranking import Ranking
from information_retrieval import Embedding_retrieval
from extract_features_refactored import FeatureExtractor

import numpy as np
import pandas as pd
from tqdm import tqdm
import io

html_name = 'Resposta_'

df = pd.read_csv('Data/processed_data_v6.csv', sep=';')
df = df.loc[(df.after_dec == True) & (df.tag_disease_covid == True)]

ranking = Ranking('texts')
inforet = Embedding_retrieval()

fe = FeatureExtractor(bert_config_file ="./models/biobert_v1.1_pubmed/bert_config.json",
                      init_checkpoint = "./models/biobert_v1.1_pubmed/model.ckpt-1000000",
                      vocab_file = "./models/biobert_v1.1_pubmed/vocab.txt",
                      batch_size = 32, # Batch size for predictions
                      max_seq_length = 128, # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
                      verbose=0)

path = 'questions.txt'
dic = {}
with open(path, 'r') as f:
    for line in f.readlines():
        print(line)
        if line[0] == '-':
            line = line[1:].strip()
            dic[line] = []
            actual_key = line
        else:
            dic[actual_key].append(line.strip())
print(dic)

res_dict = [dic]
n_task = 'questions'
for n_question, question_dict in enumerate(dic.keys()):
    print(question_dict)
    for n_subquestion, subquestion in enumerate(list(dic[question_dict])):
        print(subquestion)
        print('#################')
        print(n_question)
        print(n_subquestion)
        print('#################')
        get_answers_files(df, subquestion, ranking, ('BM25', inforet), doc_k = 3, sent_k =3,
                                         only_top_doc = False,
                                         task = n_task,
                                         question = n_question,
                                         subquestion = n_subquestion, method = 'word2vec')

for n_question, question_dict in enumerate(dic.keys()):
    for n_subquestion, subquestion in enumerate(list(dic[question_dict])):
        print(subquestion)
        get_answers_files(df, subquestion, ranking, ('BERT', fe), doc_k = 3, sent_k =3, par_k = 4,
                                         inforet_sentence = inforet, only_top_doc = True,
                                         task = n_task,
                                         question = n_question,
                                         subquestion = n_subquestion, method = 'BERT')

html_bert = create_html(res_dict, n_task, path_bert = 'json_answers/BERT/task_questions/', path_word2vec = 'json_answers/word2vec/task_questions/', array_bert = [])
html_word2vec = create_html(res_dict, n_task, path_bert = 'json_answers/BERT/task_questions/', path_word2vec = 'json_answers/word2vec/task_questiossns/', array_bert = [0,1,2,3,4,5,6,7,8,9,10])

with io.open(html_name + 'W.html', "w", encoding="utf-8") as f:
    f.write(html_bert)

with io.open(html_name + 'B.html', "w", encoding="utf-8") as f:
    f.write(html_word2vec)