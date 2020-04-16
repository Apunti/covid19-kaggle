import os
import sys
import json
import glob
import time
import pandas as pd

from datetime import datetime
from tqdm import tqdm
import numpy as np

from extract_features_refactored import FeatureExtractor

#------------------------------------------------------------------
stop_frases = ["The copyright holder for this preprint is",
               "doi: bioRxiv preprint",
               "author/funder",
               "All rights reserved", 
               "No reuse allowed without permission."]

def clean_stop_frases(field_text):
    # ToDO: replace with '' symbols like '•' 
    for frase in stop_frases:
        field_text = field_text.replace(frase,"")
    return field_text

def get_paragraph_texts(json_file, text_field = 'body_text'):    
    full_text = []
    for ii in range(len(json_file[text_field])):
        field_text = json_file[text_field][ii]['text']
        field_text = clean_stop_frases(field_text)
        full_text.append(field_text)

    full_text = "\n".join(full_text)
    return full_text

#-----------------------------------------------------------------
# The config json file corresponding to the pre-trained BERT model. Specifies the model architecture.
bert_config_file = "./models/biobert_v1.1_pubmed/bert_config.json"
# Initial checkpoint from a pre-trained BERT model
init_checkpoint = "./models/biobert_v1.1_pubmed/model.ckpt-1000000"
# The vocabulary file that the BERT model was trained on.
vocab_file = "./models/biobert_v1.1_pubmed/vocab.txt"

# Batch size for predictions.
batch_size = 32

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
max_seq_length = 128

fe = FeatureExtractor(bert_config_file,
                      init_checkpoint,
                      vocab_file,
                      batch_size, 
                      max_seq_length,
                      verbose=0)  

# Creaye a csv file for each json file with article text
data_dir = './Data/2020-04-03/'
json_filepaths = glob.glob(f'{data_dir}/**/**/**/*.json', recursive=True)

data_csv = "./Data/processed_data_v3.csv"
output_path = "./Data/BERT_encodings/"
df_data = pd.read_csv(data_csv, sep=';')

data = '2020-01-01'
data = datetime.strptime(data, '%Y-%m-%d').date()
def check_date_intrvl(date, data_min=None, data_max=None):
    if len(date.split('-')) < 3:
        date += "-01-01"
    date = datetime.strptime(date, '%Y-%m-%d').date()
    
    if data_min is None:
        data_min = datetime.strptime('1970-01-01', '%Y-%m-%d').date()       
    if data_max is None:
        data_max = datetime.strptime('2070-01-01', '%Y-%m-%d').date()          
    return (date > data_min) & (date <= data_max)

df_data['in_interval'] = df_data.apply(lambda row: check_date_intrvl(row['date'], 
                                                                     datetime.strptime('2019-08-01', '%Y-%m-%d').date(),
                                                                     datetime.strptime('2018-12-01', '%Y-%m-%d').date()),
                                        axis = 1)
print("Division of articles wrt the date ", data, " is ", df_data.in_interval.value_counts())

df_data = df_data[df_data.in_interval == True]
print("Total number of files to process is ", len(df_data))

# traverse through files with full text
def prepare_emb_from_row(row):
    print("Processing paper_id ", row["paper_id"])
    return fe.prepare_embedding_csv(input_data = row["text"], 
                                   csv_filename = os.path.join(output_path, row["paper_id"]+".csv"), 
                                   is_file=False)
tqdm.pandas()
df_data.progress_apply(lambda x: prepare_emb_from_row(x), axis =1)

#for filepath in tqdm(json_filepaths):   
#    print("Processes the file ", filepath)
#    # read json files
#    with open(filepath) as file:
#         json_file = json.load(file)
#        
#    # Here `text_field` in {'body_text', 'abstract'}
#    full_text = get_paragraph_texts(json_file, text_field = 'body_text') 
#    # prepare a DataFrame and write it down
#    fe.prepare_embedding_csv(input_data = full_text, 
#                             csv_filename = filepath[:-4]+'csv', 
#                             is_file=False)

print("DONE!")    
