import json


def create_html(res_dict, task, path_bert, path_word2vec, array_bert):
    answers_dict = {}
    
    for n_question, question_dict in enumerate(res_dict[task]):
        answers_dict[n_question] = {}
        for n_subquestion, _ in enumerate(list(question_dict.values())[0]):
            answers_dict[n_question][n_subquestion] = 0

    for n_question, question_dict in enumerate(res_dict[task]):
        bert = False
        for n_subquestion, subquestion in enumerate(list(question_dict.values())[0]):
            name = 'question_{}_'.format(n_question) + 'subquestion_{}'.format(n_subquestion)
            if n_question in array_bert:
                bert = True
                file = path_bert + name + '.json'
            else:
                file = path_word2vec + name + '.json'

            with open(file) as f:
                answers_dict[n_question][n_subquestion] = json.load(f)

    html_string = '<!DOCTYPE html><html><head><meta charset="UTF-8"/><title> COVID-19 Challenge</title></head>' \
                  '<header class="header-38e00bc3"><div class="overlay-38e00bc3"><div class="header-title-38e00bc3">' \
                  'COVID-19 Challenge</div></div></header><body class="body-38e00bc3"><nav class="container-38e00bc3">' \
                  '<ul class="questions-wrapper-38e00bc3">'

    for n_question, question_dict in enumerate(res_dict[task]):
        question = list(question_dict.keys())[0]
        subquestions = list(question_dict.values())[0]
        html_string += create_question(question, subquestions, n_question, answers_dict, bert)

    html_string += '</ul></nav></body>'

    html_string += '<style>*{padding: 0; margin: 0; box-sizing: border-box;}.header-38e00bc3{background: url(\'https://www.schwarzwaelder-bote.de/media.media.c5d3a492-5f32-4bcc-83f3-27e779ad4d46.original1024.jpg\')!important;text-align: center !important;width: 100% !important;height: 250px !important;background-size: cover !important; position: relative !important;overflow: hidden !important;border-radius: 0 0 85% 85% / 30% !important;}.header-38e00bc3 .overlay-38e00bc3{width: 100% !important; height: 100% !important;padding: 50px !important;color: #FFF !important;text-shadow: 1px 1px 1px #333 !important;}.header-title-38e00bc3{text-shadow: 5px 5px 15px black !important;font-size: 40px !important;color: white !important;margin-top: 40px !important; font-weight: bold !important;}.body-38e00bc3{background: #f2f2f2 !important;font-size: 1.25em !important;font-family: \'Raleway\', sans-serif !important; height: 900px !important;}.container-38e00bc3{margin: 50px auto !important; text-align: center !important;}.container-38e00bc3 .questions-wrapper-38e00bc3{margin: auto !important;padding-left: 0 !important;width: 90% !important;list-style: none !important;position: relative !important;}.container-38e00bc3 .subquestions-wrapper-38e00bc3{margin: auto !important;padding-left: 0 !important;width: 100% !important;list-style: none !important;position: relative !important;}.container-38e00bc3 .result-container-38e00bc3{padding: 20px 15px !important;}.container-38e00bc3 .result-title-38e00bc3{margin-bottom: 25px !important;font-size: 18px !important;}.container-38e00bc3 .result-authors-38e00bc3,.container-38e00bc3 .result-date-38e00bc3{font-style: italic !important; font-size: 18px !important;margin-bottom: 10px !important;}.container-38e00bc3 .result-text-38e00bc3{font-size: 16px;text-indent: 20px !important;text-align: justify !important;line-height: 22px !important;}.container-38e00bc3 .result-highlight-38e00bc3{background: yellow !important;padding: 2px 6px !important; box-shadow: 1px 1px 10px grey !important;}</style><style>.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3{display: block !important; padding: 15px 10px !important;color: #05a520 !important; text-decoration: none !important;font-size: 20px !important;background: #cefdd4 !important;border-bottom: 3px solid #05a520 !important;}.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3{display: block !important; padding: 10px 25px !important;color: #ff8000 !important; text-decoration: none !important;font-size: 20px !important;background: #ffffcc !important;border-bottom: 2px solid #ff8000 !important;}.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3 .triangle-38e00bc3,.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3 .triangle-38e00bc3{float: right !important;width: 0 !important; height: 0 !important;margin-top: 10px !important; border-left: 10px solid transparent !important; border-right: 10px solid transparent !important;}.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3 .triangle-38e00bc3{border-top: 10px solid #05a520 !important;}.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3 .triangle-38e00bc3{border-top: 10px solid #ff8000 !important;}.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3.active-38e00bc3 .triangle-38e00bc3{border-bottom: 10px solid #05a520 !important;border-top: none !important;margin-top: 5px !important;}.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3.active-38e00bc3 .triangle-38e00bc3{border-bottom: 10px solid #ff8000 !important;border-top: none !important;margin-top: 5px !important;}.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3:hover{background: #05a520 !important;color: #fff !important;}.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3:hover{background: #ff8000 !important;color: #fff !important;}.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3:hover .triangle-38e00bc3{border-top: 10px solid #fff !important;}.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3:hover .triangle-38e00bc3{border-top: 10px solid #fff !important;}.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3.active-38e00bc3:hover .triangle-38e00bc3{border-bottom: 10px solid #fff !important;border-top: none !important;margin-top: 5px !important;}.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3:hover .triangle-38e00bc3{border-top: 10px solid #fff !important;}</style><style>.container-38e00bc3 .accordeon-38e00bc3{cursor: pointer !important; width: 100% !important; border: none !important; text-align: left !important; outline: none !important; transition: 0.4s !important;}.container-38e00bc3 .panel-38e00bc3{text-align: left; display: none; overflow: hidden;}</style>'
    html_string += '<script>const accordeon=document.getElementsByClassName("accordeon-38e00bc3");for (let i=0; i < accordeon.length; i +=1){accordeon[i].addEventListener("click", function(){this.classList.toggle("active-38e00bc3");const panel=this.nextElementSibling;panel.style.display=panel.style.display==="block" ? "none" : "block";});}</script>'
    html_string += '</html>'
    return html_string


def create_question(question, subquestions, n_question, answers_dict, bert):
    new_question = '<li class="question-38e00bc3"><button class="accordeon-38e00bc3">'
    new_question += question
    new_question += '<div class="triangle-38e00bc3"></div></button><div class="panel-38e00bc3">' \
                    '<ul class="subquestions-wrapper-38e00bc3">'
    for n_subquestion, subquestion in enumerate(subquestions):
        new_question += create_subquestion(subquestion, n_question, n_subquestion, answers_dict, bert)

    new_question += '</ul></div></li>'

    return new_question


def create_subquestion(subquestion, n_question, n_subquestion, answers_dict, bert):
    new_subquestion = '<li class="subquestion-38e00bc3"><button class="accordeon-38e00bc3">'
    new_subquestion += subquestion
    new_subquestion += '<div class="triangle-38e00bc3"></div></button><div class="panel-38e00bc3 result-container-38e00bc3">'

    for element in answers_dict[n_question][n_subquestion]:
        new_subquestion += create_element(element, bert)

    new_subquestion += '</div></li>'

    return new_subquestion


def create_element(element, bert):
    title = element['title']
    url = element['url']
    authors = element['authors']
    evidence = element['evidence']
    design = element['design']
    date = element['date']
    sentences = element['sentences']

    new_element = '<div class="result-title-38e00bc3">' \
                  '<a href="' + url + '" target="_blank">' + title + '</a></div><p class="result-authors-38e00bc3">'
    new_element += authors + '</p><p class="result-date-38e00bc3">' + date + '</p><p class="result-text-38e00bc3">'
    if bert:
        new_element += '<br>'.join(sentences)
    else:
        if type(sentences) == str:
            new_element += '<br>' + sentences + '<br>'
        else:
            new_element += '<br>'.join(sentences)
        
    new_element += '</p>'

    return new_element