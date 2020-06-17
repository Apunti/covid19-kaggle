from transformers import pipeline
from ranking import Ranking
from get_result import get_level_evidence, get_design

import pandas as pd

def get_answer_from_doc(query, doc, qa_model):

    output = []
    
    seq_length = 250
    stride = 150
    splitted_doc = doc.split(' ')
    
    i = 0
    while len(splitted_doc) > seq_length:
        paragraph = ' '.join(splitted_doc[:seq_length])
        m_input = {'question': query,
                 'context': paragraph}
        try:
            output_dict = qa_model(m_input)
        except:
            print('### i = {} ###'.format(i))
            print('### CONTEXT ###')
            print(m_input['context'])
            if i!=0:
                splitted_doc = splitted_doc[stride:]
                i = 0
            i += 1
            continue
        answer = output_dict['answer']
        score = output_dict['score']

        output.append((answer, score, paragraph))
        
        splitted_doc = splitted_doc[stride:]
        i = 0
        
    if len(splitted_doc) > 127:
        
        paragraph = ' '.join(splitted_doc)
        m_input = {'question': query,
                 'context': paragraph}
        print('processing paragraph...', end= '')
        output_dict = qa_model(m_input)
        answer = output_dict['answer']
        score = output_dict['score']

        output.append((answer, score, paragraph))
    
    sorted_output = sorted(output, key=lambda x: x[1], reverse=True)
    
    print('### ANSWER ###')
    print(sorted_output[0])

    if sorted_output[0][1] > 0.3:
        return sorted_output[0][0], sorted_output[0][2]
    else:
        return '-', sorted_output[0][2]
    
def get_documents(dataset, ranking, query, top_k = 10):

    similar = ranking.most_similar(query, dataset, k = top_k, func='bm25', data='text')
    print('similar length: {}'.format(len(similar)))

    return similar

def get_information(row):
    date = row['date'].values[0]
    url = row['url'].values[0]
    authors = row['authors'].values[0]
    title = row['title'].values[0]
    design = get_design(row)
    level_of_evidence = get_level_evidence(row)
    
    new_line = {'date': date,
                'title': '<a href="' + url + '">' + title + '</a>',
                'authors': authors,
                'design': design,
                'level_of_evidence': level_of_evidence}
    
    return new_line
    

def get_csv(df, csv_path, risk_factor, questions, top_k = 1, device = -1, dict_path = 'Data/ranking_dict'):

    dataset = df #pd.read_csv(df_path, sep=';')

    ranking = Ranking('texts', path= dict_path)
    qa_model = pipeline('question-answering', device=device, model='bert-large-uncased-whole-word-masking-finetuned-squad')

    print('All loaded')

    all_query = ' '.join(questions)
    documents = get_documents(dataset, ranking, all_query, top_k = top_k)
    print('Length documents: {}'.format(len(documents)))

    results = pd.DataFrame(columns=['date', 'title', 'authors', 'design', 'level_of_evidence'] + questions)

    print('Starting docs for {}: \n'.format(risk_factor))

    for doc in documents:
        row = dataset.loc[dataset.text == doc]
        new_line = get_information(row)
        #new_line = {'paper_id': paper_id}
        for query in questions:
            answer, paragraph = get_answer_from_doc(query, doc, qa_model)
            new_line[query] = str(paragraph.replace(answer, f"<mark>{answer}</mark>")) 
            print(answer)
        results = results.append(new_line, ignore_index=True)
        
    results.to_csv(csv_path, sep=';', index=False)