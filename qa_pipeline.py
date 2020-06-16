from transformers import pipeline
from ranking import Ranking

import pandas as pd

def get_answer_from_doc(query, doc, qa_model):

    output = []

    for paragraph in doc.strip().split('/n'):
        input = {'question': query,
                 'context': paragraph}
        print('processing paragraph...', end= '')
        output_dict = qa_model(input)
        answer = output_dict['answer']
        score = output_dict['score']

        output.append((answer, score))

    sorted_output = sorted(output, key=lambda x: x[1], reverse=True)

    if sorted_output[0][1] > 0.4:
        return sorted_output[0][0]
    else:
        return '-'


def get_documents(dataset, ranking, query, top_k = 10):

    similar = ranking.most_similar(query, dataset, k = top_k, func='bm25', data='text')
    print('similar length: {}'.format(len(similar)))

    return similar

def get_csv(df, csv_path, risk_factor, questions, top_k = 1, device = -1, dict_path = 'Data/ranking_dict'):

    dataset = df #pd.read_csv(df_path, sep=';')

    ranking = Ranking('texts', path= dict_path)
    qa_model = pipeline('question-answering', device=device)

    print('All loaded')

    all_query = ' '.join(questions)
    documents = get_documents(dataset, ranking, all_query, top_k = top_k)
    print('Length documents: {}'.format(len(documents)))

    results = pd.DataFrame(columns=['paper_id'] + questions)

    print('Starting docs: \n')

    for doc in documents:
        print('NEW DOC')
        paper_id = dataset.loc[dataset.text == doc].paper_id.values[0]
        new_line = {'paper_id': paper_id}
        for query in questions:
            answer = get_answer_from_doc(query, doc, qa_model)
            new_line[query] = answer
            print(answer)
        results = results.append(new_line, ignore_index=True)
        print(results)

    print(results)
    results.to_csv(csv_path, sep=';', index=False)


if __name__ == '__main__':
    factor = 'cancer'
    questions2 = ['What is the severity of ' + factor,
                  'What is the fatality of ' + factor,
                  'What is the study population',
                  'Which type of study is it',
                  'Sample size of the study']
    questions = ['Study Type',
                 'Severity of ' + factor,
                 'Severity lower bound of ' + factor,
                 'Severity upper bound of ' + factor,
                 'Severity p-value of ' + factor,
                 'Severe significance of ' + factor,
                 'Severe adjusted of ' + factor,
                 'Hand-calculated Severe of ' + factor,
                 'Fatality of ' + factor,
                 'Fatality lower bound of ' + factor,
                 'Fatality upper bound of ' + factor,
                 'Fatality p-value of ' + factor,
                 'Fatality significance of ' + factor,
                 'Fatality adjusted of ' + factor,
                 'Hand-calculated Fatality of ' + factor,
                 'Multivariate adjustment of ' + factor,
                 'Sample size',
                 'Study population']
    get_csv('Data/processed_data_v6.csv', 'csv/test3.csv', 'risk_factor_' + factor, questions)

