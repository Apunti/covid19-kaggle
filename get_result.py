import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import itertools
import json
import os

def get_ranking_nearest(query, ranking, df, doc_k):
    covid_papers = df[(df.after_dec == True) & (df.tag_disease_covid == True)].paper_id
    scores = ranking.get_bm25_scores(query, covid_papers)
    sorted_paper_id = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranking_nearest = [a for a, b in sorted_paper_id][:doc_k]  # Ranking function nearest
    return ranking_nearest
    
def get_result(df, query, ranking, inforet_tuple, doc_k = 5, sent_k = 3):
    """
    'inforet_tuple' is a tuple (information_retrieval_type, information_retrieval_instance), eg:
        ('BERT', instance of 'FeatureExtractor' from 'extract_features_refactored.py' )
        ('TFIDF', instance 'Embedding_retrieval' from 'information_retrieval.py')
    """   
    query = filtered_query(query)
    ranking_nearest = get_ranking_nearest(query, ranking, df, doc_k)
    # provide additional preprocessingfor BERT
    if (len(inforet_tuple)==2):
        (inforet_type, inforet) = inforet_tuple
        if (inforet_type=='BERT'):
            query = inforet.prepare_embedding_csv(query, None, False).values
    else: 
        inforet = inforet_tuple
    
    # get the list of tuples (paper_id, sentence, similarity) for "topk" sentences
    doc_info = []
    for paper_id in ranking_nearest:
        actual_doc = []
        row = df.loc[df.paper_id == paper_id]
        if len(row.text.values) == 0 or not row['after_dec'].values[0]:
            continue
            
        # get 'topk' closest sentences/paragraphs
        similar_sent = inforet.get_closest_sentence(query, paper_id, row.text.values[0], topk=10)
                        
        for sent, sim in similar_sent:
            actual_doc.append((paper_id, sent, sim))
        doc_info.append(actual_doc)

    # create a new df with  a "sentences" column 
    columns = list(df.columns)
    columns.append('sentences')
    result = pd.DataFrame(columns=columns)
    sentences = []
    
    # take the first 'doc_k' higher ranked files
    for doc in doc_info:
        paper_id = doc[0][0]
        
        # exclude short documents
        if len(doc) < sent_k:
            print('DOC TOO SHORT')
            continue
            
        # add 'sent_k' closest sentence
        actual_sent = []
        for j in range(sent_k):
            actual_sent.append(doc[j][1])
            
        # add to df the meta data on the document    
        row = df[df.paper_id == paper_id]
        result = result.append(row, ignore_index=True)
        
        # update the array of sentences ('sent_k' sentences per document)
        sentences.append(actual_sent)

    result['sentences'] = pd.Series(sentences)

    return result

def filtered_query(query):
    """
    Method to obtain cleaned query from a given query.
    :param query: given text/abstract/paragraph <string>
    :return: list of strings
    """
    query = " ".join(remove_punct(query.split())).lower()
    stop_words = set(stopwords.words('english'))
    for term in ['et', 'al', 'also', 'fig']:
        stop_words.add(term)
    word_tokens = word_tokenize(query)

    cleaned_query = [w for w in word_tokens if not w in stop_words]

    return ' '.join(cleaned_query)

def remove_punct(text):
    """
    An additional function to remove punctuation from the query
    :param text: given text/abstract/paragraph <string>
    :return: string
    """
    new_words = []
    for word in text:
        w = re.sub(r'[^\w\s]','',word) #remove everything except words and space
        w = re.sub(r'\_','',w) #to remove underscore as well
        new_words.append(w)
    return new_words

def get_answers_files(df, query, ranking, inforet_tuple, doc_k, sent_k, par_k = None, inforet_sentence = None, only_top_doc = False, task = None, question = None, subquestion = None):
    """
    'inforet_tuple' is a tuple (information_retrieval_type, information_retrieval_instance), eg:
        ('BERT', instance of 'FeatureExtractor' from 'extract_features_refactored.py' )
        ('TFIDF', instance 'Embedding_retrieval' from 'information_retrieval.py')
    """
    if not task is None:
        directory = 'json_answers/task_{}/'.format(task)
    else:
        directory = 'json_answers/test/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    query = filtered_query(query)
    ranking_nearest = get_ranking_nearest(query, ranking, df, doc_k)
    # provide additional preprocessingfor BERT
    if (len(inforet_tuple) == 2):
        (inforet_type, inforet) = inforet_tuple
        if (inforet_type == 'BERT'):
            bert_query = inforet.prepare_embedding_csv(query, None, False).values
            #print('query processed')
    else:
        inforet = inforet_tuple

    #print('Ranking processed')

    # get the list of tuples (paper_id, sentence, similarity) for "topk" sentences
    doc_info = []
    for paper_id in ranking_nearest:
        actual_doc = []
        row = df.loc[df.paper_id == paper_id]
        if len(row.text.values) == 0 or not row['after_dec'].values[0]:
            continue

        # get 'topk' closest sentences/paragraphs
        if (inforet_type == 'BERT'):
            similar_par = inforet.get_closest_sentence(bert_query, paper_id, row.text.values[0], par_k)
            top_paragraphs = ''
            for n_par in range(par_k):
                if len(similar_par) < par_k:
                    print('DOC TOO SHORT')
                    continue
                top_paragraphs += similar_par[n_par][0] + '\n'
            similar_sent = inforet_sentence.get_closest_sentence(query, paper_id, top_paragraphs, sent_k)
        else:
            similar_sent = inforet.get_closest_sentence(query, paper_id, row.text.values[0], sent_k)
        for sent, sim in similar_sent:
            actual_doc.append((paper_id, sent, sim))
        doc_info.append(actual_doc)

    #print('Calculated similarities for the top_k doc')

    output_dict = []

    if only_top_doc:
        # take the first 'doc_k' higher ranked files
        for doc in doc_info:

            # exclude short documents
            if len(doc) < sent_k:
                print('DOC TOO SHORT')
                continue

            paper_id = doc[0][0]

            # add 'sent_k' closest sentence
            actual_sent = []
            for j in range(sent_k):
                actual_sent.append(doc[j][1])

            row = df[df.paper_id == paper_id]
            actual_dict = get_dictionary(row)
            actual_dict['sentences'] = actual_sent
            output_dict.append(actual_dict)

    else:
        all_sent = list(itertools.chain.from_iterable(doc_info))
        sorted_doc= sorted(all_sent, key=lambda x: x[2], reverse=True)
        for i in range(sent_k):
            paper_id, sent, _ = sorted_doc[i]

            row = df[df.paper_id == paper_id]
            actual_dict = get_dictionary(row)

            actual_dict['sentences'] = sent
            output_dict.append(actual_dict)

    if question is None or subquestion is None:
        path = directory + 'test.json'
    else:
        name = 'question_{}_'.format(question) + 'subquestion_{}'.format(subquestion)
        path = directory + name + '.json'
    with open(path, 'w') as outfile:
        json.dump(output_dict, outfile)

def get_dictionary(row):
    """
    Get the dict to create the pickle without the closest sentences.
    :param row: row of the original df (read_data) for a given paper_id
    :return: dictionary with keys: [title, date, authors, url, design, evidence]
    """

    date = row['date'].values[0]
    evidence = get_level_evidence(row)
    design = get_design(row)
    if type(row['authors'].values[0]) == str:
        authors = row['authors'].values[0]
    else:
        authors = '-'
    url = row['url'].values[0]
    title = row['title'].values[0]

    return {'title': title, 'date': date, 'authors': authors, 'url': url, 'design': design, 'evidence': evidence}

def get_design(row):
    if row['tag_design_simulation'].values[0]:
        return 'Simulation'
    elif row['tag_design_case_study'].values[0]:
        return 'Case study'
    elif row['tag_design_prevalence_survey'].values[0]:
        return 'Prevalence survey'
    elif row['tag_design_time_series_analysis'].values[0]:
        return 'Time series analysis'
    elif row['tag_design_generic_case_control'].values[0]:
        return 'Generic case-control'
    elif row['tag_design_retrospective_cohort'].values[0]:
        return 'Retrospective cohort'
    elif row['tag_design_cross_sectional_case_control'].values[0]:
        return 'Cross sectional case control'
    elif row['tag_design_matched_case_control'].values[0]:
        return 'Matched case-control'
    elif row['tag_design_pseudo_randomized_control'].values[0]:
        return 'Pseudo-randomized controlled trials'
    elif row['tag_design_randomized_control'].values[0]:
        return 'Randomized controlled trials'
    elif row['tag_design_systematic_review'].values[0]:
        return 'Systematic review and meta-analysis'
    return 'None'

def get_level_evidence(row):
    if row['tag_design_simulation'].values[0]:
        return '6'
    elif row['tag_design_case_study'].values[0]:
        return '6'
    elif row['tag_design_prevalence_survey'].values[0]:
        return '6'
    elif row['tag_design_time_series_analysis'].values[0]:
        return '5'
    elif row['tag_design_generic_case_control'].values[0]:
        return '4'
    elif row['tag_design_retrospective_cohort'].values[0]:
        return '4'
    elif row['tag_design_cross_sectional_case_control'].values[0]:
        return '4'
    elif row['tag_design_matched_case_control'].values[0]:
        return '4'
    elif row['tag_design_pseudo_randomized_control'].values[0]:
        return '3'
    elif row['tag_design_randomized_control'].values[0]:
        return '2'
    elif row['tag_design_systematic_review'].values[0]:
        return '1'
    return 'None'

    return 'None'