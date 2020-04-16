import pandas as pd
import json
import os
from tqdm import tqdm
import sys
from langdetect import detect
import re
import pickle
import string
import datetime

from datetime import datetime
from nltk.stem import WordNetLemmatizer, SnowballStemmer

def read_files():
    """
    Creates a dataframe with processed text and abstract from the json files.
    :return: dataframe with columns ['paper_id', 'title', 'abstract', 'text']
    """
    # Creates a dataframe with the needed columns
    df = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'text'])

    # Path to the directories with the json
    bioarxiv_dir = 'Data/biorxiv_medrxiv/biorxiv_medrxiv'
    comm_use_dir = 'Data/comm_use_subset/comm_use_subset'
    custom_license_dir = 'Data/custom_license/custom_license'
    noncomm_use_dir = 'Data/noncomm_use_subset/noncomm_use_subset'

    data_directories = [bioarxiv_dir, comm_use_dir, custom_license_dir, noncomm_use_dir]

    num_documents = 0
    print('Reading files...\n')
    for dir in data_directories:
        print('Reading {} files'.format(dir))
        for filename in tqdm(os.listdir(dir)):
            filename = os.path.join(dir, filename)
            with open(filename, 'rb') as f:
                row = _read_file(f)
                if row is None:
                    continue
                df = df.append(row, ignore_index=True)
            num_documents += 1

    print('\nThe dataset consists of {} documents'.format(num_documents))

    # Calculate the size of the dataframe
    dec = sys.getsizeof(df) % 1000000
    df_size = sys.getsizeof(df)//1000000

    print('The size of the dataframe is of {},{} MB'.format(df_size, dec))

    return df


def _read_file(json_file):
    """
    Function that reads a json file and outputs a dict to append to a df.
    It doesn't take a document into account if:
        - Not written in English
        - Number of words in text < 500.
    If there is no abstract, it takes the title as the abstract
    TODO Think if we want the authors too
    :param json_file: path to the jsonfile
    :return: dict with keys: [paper_id, title, abstract, text]
    """
    # Load json file
    file_dict = json.load(json_file)

    # Get information
    paper_id = file_dict['paper_id']
    title = file_dict['metadata']['title'] # string
    abstract = '\n'.join([paragraph['text'] for paragraph in file_dict['abstract']]) # string

    # Check if the abstract is written in English
    if type(abstract) == str and len(abstract) > 10:
        if detect(abstract) != 'en':
            return None
    else:
        abstract = title

    text = [paragraph['text'] for paragraph in file_dict['body_text']]
    text = '\n'.join(text) # uncomment this line if instead of list of paragraph you want the whole text as a string

    # If there are less than 500 words, don't consider the document
    if len(re.findall(r"[\w']+|[.,!?;]", text)) < 500:
        return None

    # Check if the text is written in English
    if detect(text) != 'en':
        return None

    # If there is no abstract, check if text is written in English
    if type(abstract) != str:
        if detect(text) != 'en':
            return None

    return {'paper_id': paper_id, 'title': title, 'abstract': abstract, 'text': text}


def save_dictionaries(df):
    """
    Calculates the dictionaries needed to compute TFIDF score and saves them.
        term_frequencies: dict of dicts of term frequencies within a document.
        document_frequencies: number of documents containing a given term.
        document_length: number of words in a document.
    The dictionaries will be stored in the directory Data/ranking_dict/
    """

    directory = 'Data/ranking_dict/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # CREATE TEXT DICTIONARIES

    term_frequencies = {}  # dict of dicts paper_id -> word -> frequency within a document
    document_frequencies = {}  # dict word -> number of documents containing the term
    document_length = {}  # dict paper_id -> document length

    for id in list(df.paper_id):
        term_frequencies[id] = {}
        document_length[id] = 0

    print('Processing the corpus...\n')
    for id, document in tqdm(zip(list(df.paper_id), list(df.text))):
        actual_frequencies = {}
        words_set = set()
        length = 0
        if type(document) != type('a'):
            continue
        for word in re.findall(r"[\w']+|[.,!?;]", document.strip()):
            word = word.lower()
            if word in string.punctuation:
                continue
            length += 1
            if word in actual_frequencies:
                actual_frequencies[word] += 1
            else:
                actual_frequencies[word] = 1
            words_set.add(word)
        for word in words_set:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1
        document_length[id] = length
        term_frequencies[id] = actual_frequencies

    # Save dictionaries into files
    print('Saving dictionaries v5...')
    with open('Data/ranking_dict/document_frequencies_text_v5.p', 'wb') as fp:
        pickle.dump(document_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/term_frequencies_text_v5.p', 'wb') as fp:
        pickle.dump(term_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/document_length_text_v5.p', 'wb') as fp:
        pickle.dump(document_length, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # REPEAT WITH ABSTRACTS

    term_frequencies = {}  # dict of dicts paper_id -> word -> frequency within a document
    document_frequencies = {}  # dict word -> number of documents containing the term
    document_length = {}  # dict paper_id -> document length

    for id in list(df.paper_id):
        term_frequencies[id] = {}
        document_length[id] = 0

    print('Processing the corpus...\n')
    for id, document in tqdm(zip(list(df.paper_id), list(df.abstract))):
        actual_frequencies = {}
        words_set = set()
        length = 0
        if type(document) != type('a'):
            continue
        for word in re.findall(r"[\w']+|[.,!?;]", document.strip()):
            word = word.lower()
            if word in string.punctuation:
                continue
            length += 1
            if word in actual_frequencies:
                actual_frequencies[word] += 1
            else:
                actual_frequencies[word] = 1
            words_set.add(word)
        for word in words_set:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1
        document_length[id] = length
        term_frequencies[id] = actual_frequencies

    # Save dictionaries into files
    with open('Data/ranking_dict/document_frequencies_abstract_v5.p', 'wb') as fp:
        pickle.dump(document_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/term_frequencies_abstract_v5.p', 'wb') as fp:
        pickle.dump(term_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/document_length_abstract_v5.p', 'wb') as fp:
        pickle.dump(document_length, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # CREATE TEXT DICTIONARIES

    term_frequencies = {}  # dict of dicts paper_id -> word -> frequency within a document
    document_frequencies = {}  # dict word -> number of documents containing the term
    document_length = {}  # dict paper_id -> document length

    stemmer = SnowballStemmer("english")

    for id in list(df.paper_id):
        term_frequencies[id] = {}
        document_length[id] = 0

    print('Processing the corpus...\n')
    for id, document in tqdm(zip(list(df.paper_id), list(df.text))):
        actual_frequencies = {}
        words_set = set()
        length = 0
        if type(document) != type('a'):
            continue
        for word in re.findall(r"[\w']+|[.,!?;]", document.strip()):
            word = stemmer.stem(WordNetLemmatizer().lemmatize(word.lower(), pos='v'))
            if word in string.punctuation:
                continue
            length += 1
            if word in actual_frequencies:
                actual_frequencies[word] += 1
            else:
                actual_frequencies[word] = 1
            words_set.add(word)
        for word in words_set:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1
        document_length[id] = length
        term_frequencies[id] = actual_frequencies

    # Save dictionaries into files
    with open('Data/ranking_dict/document_frequencies_text_proc_v5.p', 'wb') as fp:
        pickle.dump(document_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/term_frequencies_text_proc_v5.p', 'wb') as fp:
        pickle.dump(term_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/document_length_text_proc_v5.p', 'wb') as fp:
        pickle.dump(document_length, fp, protocol=pickle.HIGHEST_PROTOCOL)

def read_new_files():
    """
    Creates a dataframe with processed text and abstract from the json files.
    :return: dataframe with columns ['paper_id', 'title', 'abstract', 'text']
    """
    # Path to the metadata file
    metadata_path = 'Data/metadata.csv'

    # Creates a dataframe with the needed columns
    df = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'text', 'date'])

    # Path to the directories with the json
    bioarxiv_dir = 'Data/biorxiv_medrxiv/biorxiv_medrxiv'
    comm_use_dir = 'Data/comm_use_subset/comm_use_subset'
    custom_license_dir = 'Data/custom_license/custom_license'
    noncomm_use_dir = 'Data/noncomm_use_subset/noncomm_use_subset'

    data_directories = {'biorxiv_medrxiv': bioarxiv_dir,
                        'comm_use_subset': comm_use_dir,
                        'custom_license': custom_license_dir,
                        'noncomm_use_subset': noncomm_use_dir}

    metadata = pd.read_csv(metadata_path, low_memory=False)
    for _, row in tqdm(metadata.iterrows()):
        has_pdf_parse = row['has_pdf_parse']
        has_pmc_xml_parse = row['has_pmc_xml_parse']
        path = row['full_text_file']
        if type(path) != str:
            continue
        path = data_directories[path]
        paper_id = row['cord_uid']
        url = row['url']
        title = row['title']
        abstract = row['abstract']
        authors = row['authors']
        date = row['publish_time']
        if has_pdf_parse:
            sha = row['sha']
            if ';' in sha:
                continue
            pdf_path = path + '/pdf_json/' + sha + '.json'
            if has_pmc_xml_parse:
                pmcid = row['pmcid']
                pmc_path = path + '/pmc_json/' + pmcid + '.xml.json'
                text = get_text(pmc_path)
            else:
                text = get_text(pdf_path)
            if type(abstract) != str:
                abstract = get_abstract(pdf_path)
            if len(text.split()) < 200:
                continue
            else:
                if detect(text) != 'en':
                    continue
            new_row = {'paper_id': paper_id, 'authors': authors,'title': title, 'abstract': abstract, 'text': text, 'date': date, 'url': url}
            df = df.append(new_row, ignore_index=True)

    #print('\nThe dataset consists of {} documents'.format(num_documents))

    # Calculate the size of the dataframe
    dec = sys.getsizeof(df) % 1000000
    df_size = sys.getsizeof(df) // 1000000

    data = '2020-01-01'
    data = datetime.strptime(data, '%Y-%m-%d').date()

    tag_disease = {'covid': ['covid', '2019 ncov', 'sars cov 2', 'sars-cov-2', r'coronavirus 2\b',
                             'coronavirus 2019', 'wuhan coronavirus', 'coronavirus disease 19',
                             'ncov 2019', 'wuhan pneumonia', '2019ncov', 'wuhan virus',
                             r'2019n cov\b', r'2019 n cov\b', 'r\bn cov 2019', 'wuhan'],
                   'sars': [r'\bsars\b', 'severe acute respiratory syndrome'],
                   'mers': [r'\bmers\b', 'middle east respiratory syndrome'],
                   'corona': ['corona', r'\bcov\b']}

    tag_design = {'generic_case_control': ['adjusted odds ratio', 'AOR', 'non-response bias', 'potential confounders',
                                           'psychometric evaluaton of instrument', 'questionnaire development',
                                           'response rate', 'survey instrument', 'data collection instrument',
                                           'eligibility criteria'],
                  'retrospective_cohort': ['data collection instrument', 'eligibility criteria', 'recruitment',
                                           'potential confounders', 'data abstraction forms', 'inter-rater reliability',
                                           "cohen's kappa", "'data abstraction forms'"],
                  'cross_sectional_case_control': ['Adjusted Odds Ratio', 'AOR', 'response rate',
                                                   'questionnaire development', 'psychometric evaluation of instrument',
                                                   'eligibility criteria', 'recruitment', 'potential confounders',
                                                   'non-response bias'],
                  'matched_case_control': ['Adjusted Odds Ratio', 'AOR', 'data collection instrument',
                                           'survey instrument', 'response rate', 'questionnaire development',
                                           'psychometric evaluation of instrument', 'eligibility criteria',
                                           'recruitment', 'potential confounders', 'matching criteria',
                                           'non-response bias'],
                  'prevalence_survey': ['random sample', 'response rate', 'questionnaire development',
                                        'psychometric evaluation of instrument', 'eligibility criteria', 'recruitment',
                                        'potential confounders', ' data collection instrument', 'survey instrument'],
                  'time_series_analysis': ['eligibility criteria', 'recruitment', 'potential confounders',
                                           'adjusted hazard ratio', 'multivariate hazard ratio'],
                  'systematic_review': ['Cochrane review', 'PRISMA', 'protocol', 'registry', 'search string',
                                        'search criteria', 'search strategy', 'eligibility criteria',
                                        'inclusion criteria', 'exclusion criteria', 'interrater reliability',
                                        "cohen's kappa", 'databases searched', 'risk of bias', 'heterogeneity', 'i2',
                                        'publication bias'],
                  'randomized_control': ['CONSORT', 'double-blind', 'eligibility', 'power', 'risk of bias', 'baseline',
                                         'protocol', 'registry'],
                  'pseudo_randomized_control': ['double-blind', 'eligibility', 'power', 'risk of bias', 'baseline',
                                                'protocol', 'registry'],
                  'case_study': ['clinical findings', 'symptoms', 'diagnosis', 'interventions', 'outcomes', 'dosage',
                                 'strength', 'duration', 'follow-up', 'adherence', 'tolerability'],
                  'simulation': ['AUC', 'area under the curve', 'receiver-operator curve', 'ROC', 'model fit', 'AIC',
                                 'Akaike Information Criterion']}

    df['after_dec'] = df.apply(lambda row: check_date(data, row['date']), axis=1)
    print('tagging_disease')
    for key in tag_disease:
        col_name = 'tag_disease_' + key
        df[col_name] = df.apply(lambda row: check_disease(tag_disease[key], row['text']), axis=1)

    print('tagging design')
    for key in tag_design:
        col_name = 'tag_design_' + key
        df[col_name] = df.apply(lambda row: check_design(tag_design[key], row['text']), axis=1)


    print('The size of the dataframe is of {},{} MB'.format(df_size, dec))

    return df

def get_text(json_file):
    with open(json_file, 'r') as f:
        file_dict = json.load(f)

    body_text = file_dict['body_text']
    text = ''
    new_text = None
    section = None
    for el in body_text:
        if len(el['text']) == 0 or not any(c.isalpha() for c in el['text']):
            continue
        #print(el['text'])
        if new_paragraph(el['text'], 0):
            if not new_text is None:
                text += new_text + '\n'
                new_text = None
            text += el['text'] + '\n'
        #new_section = el['section']
        #if new_section != section:
            #text += new_section + '\n'
            #section = new_section
        else:
            if new_text is None:
                new_text = el['text']
            else:
                new_text += ' ' + el['text']
    return text

def new_paragraph(string, count):
    if count > 4:
        return True
    if string[0].islower():
        return False
    elif not string[0].isalpha():
        return new_paragraph(string[1:], count +1)
    return True

def get_abstract(json_file):
    with open(json_file, 'r') as f:
        file_dict = json.load(f)

    section_abstract = file_dict['abstract']
    abstract = ''
    for el in section_abstract:
        abstract += el['text'] + '\n'
    return abstract

def check_disease(l, text):
    trans = str.maketrans('-',' ')
    text = text.lower()
    for word in l:
        if word in text.translate(trans):
            return True
    return False

def check_design(l, text):
    count = 0
    for word in l:
        if word.lower() in text.lower():
            count += 1
            if count > 2:
                return True
    return False

def check_date(data, date):
    if len(date.split('-')) < 3:
        date += "-01-01"
    date = datetime.strptime(date, '%Y-%m-%d').date()
    return date > data

tag_disease =  {'covid': ['covid', '2019 ncov', 'sars cov 2', 'sars-cov-2', r'coronavirus 2\b',
                'coronavirus 2019', 'wuhan coronavirus', 'coronavirus disease 19',
                'ncov 2019', 'wuhan pneumonia', '2019ncov', 'wuhan virus',
                r'2019n cov\b', r'2019 n cov\b', 'r\bn cov 2019', 'wuhan'],
                'sars': [r'\bsars\b', 'severe acute respiratory syndrome'] ,
                'mers': [r'\bmers\b', 'middle east respiratory syndrome'],
                'corona': ['corona', r'\bcov\b']}

tag_design = {'generic_case_control': ['adjusted odds ratio', 'AOR', 'non-response bias', 'potential confounders', 'psychometric evaluaton of instrument', 'questionnaire development', 'response rate', 'survey instrument', 'data collection instrument', 'eligibility criteria'],\
              'retrospective_cohort': ['data collection instrument', 'eligibility criteria', 'recruitment', 'potential confounders', 'data abstraction forms', 'inter-rater reliability', "cohen's kappa", "'data abstraction forms'"],
              'cross_sectional_case_control': ['Adjusted Odds Ratio', 'AOR', 'response rate', 'questionnaire development', 'psychometric evaluation of instrument', 'eligibility criteria', 'recruitment', 'potential confounders', 'non-response bias'],
              'matched_case_control': ['Adjusted Odds Ratio', 'AOR', 'data collection instrument', 'survey instrument', 'response rate', 'questionnaire development', 'psychometric evaluation of instrument', 'eligibility criteria', 'recruitment', 'potential confounders', 'matching criteria', 'non-response bias'],
              'prevalence_survey': ['random sample', 'response rate', 'questionnaire development', 'psychometric evaluation of instrument', 'eligibility criteria', 'recruitment', 'potential confounders', ' data collection instrument', 'survey instrument'],
              'time_series_analysis': ['eligibility criteria', 'recruitment', 'potential confounders', 'adjusted hazard ratio', 'multivariate hazard ratio'],
              'systematic_review': ['Cochrane review', 'PRISMA', 'protocol', 'registry', 'search string', 'search criteria', 'search strategy', 'eligibility criteria', 'inclusion criteria', 'exclusion criteria', 'interrater reliability', "cohen's kappa", 'databases searched', 'risk of bias', 'heterogeneity', 'i2', 'publication bias'],
              'randomized_control': ['CONSORT', 'double-blind', 'eligibility', 'power', 'risk of bias', 'baseline', 'protocol', 'registry'],
              'pseudo_randomized_control': ['double-blind', 'eligibility', 'power', 'risk of bias', 'baseline', 'protocol', 'registry'],
              'case_study': ['clinical findings', 'symptoms', 'diagnosis', 'interventions', 'outcomes', 'dosage', 'strength', 'duration', 'follow-up', 'adherence', 'tolerability'],
              'simulation': ['AUC', 'area under the curve', 'receiver-operator curve', 'ROC', 'model fit', 'AIC', 'Akaike Information Criterion']}