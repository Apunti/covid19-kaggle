import json


def create_html(res_dict, task, path='../input/results-json/task_0/'):
    answers_dict = {}
    for n_question, question_dict in enumerate(res_dict[task]):
        answers_dict[n_question] = {}
        for n_subquestion, _ in enumerate(list(question_dict.values())[0]):
            answers_dict[n_question][n_subquestion] = 0

    for n_question, question_dict in enumerate(res_dict[task]):
        for n_subquestion, subquestion in enumerate(list(question_dict.values())[0]):
            name = 'question_{}_'.format(n_question) + 'subquestion_{}'.format(n_subquestion)
            file = path + name + '.json'

            with open(file) as f:
                answers_dict[n_question][n_subquestion] = json.load(f)

    html_string = '<!DOCTYPE html><html><head><meta charset="UTF-8"/><title> COVID-19 Challenge</title></head>' \
                  '<header class="header-38e00bc3"><div class="overlay-38e00bc3"><h1 class="header-title-38e00bc3">' \
                  'COVID-19 Challenge</h1></div></header><body class="body-38e00bc3"><nav class="container-38e00bc3">' \
                  '<ul class="questions-wrapper-38e00bc3">'

    for n_question, question_dict in enumerate(res_dict[task]):
        question = list(question_dict.keys())[0]
        subquestions = list(question_dict.values())[0]
        html_string += create_question(question, subquestions, n_question, answers_dict)

    html_string += '</ul></nav></body>'

    html_string += '<!-- This is general styles for the page -->\r\n\t<style>\t\r\n\t\t*{padding: 0; margin: 0; ' \
                   'box-sizing: border-box;}\r\n\r\n\t\t.header-38e00bc3 {\r\n\t\t\tbackground: url(\'https://www.sc' \
                   'hwarzwaelder-bote.de/media.media.c5d3a492-5f32-4bcc-83f3-27e779ad4d46.original1024.jpg\') !important;' \
                   '\r\n\t\t\ttext-align: center !important;\r\n\t\t\twidth: 100% !important;\r\n\t\t\theight: 250px ' \
                   '!important;\r\n\t\t\tbackground-size: cover !important;\r\n\t\t\tposition: relative !important;' \
                   '\r\n\t\t\toverflow: hidden !important;\r\n\t\t\tborder-radius: 0 0 85% 85% / 30% !important;\r\n\t\t}' \
                   '\r\n\r\n\t\t.header-38e00bc3 .overlay-38e00bc3 {\r\n\t\t\twidth: 100% !important;\r\n\t\t\theight: 100% ' \
                   '!important;\r\n\t\t\tpadding: 50px !important;\r\n\t\t\tcolor: #FFF !important;\r\n\t\t\ttext-shadow: ' \
                   '1px 1px 1px #333 !important;\r\n\t\t}\r\n\r\n\t\t.header-title-38e00bc3 {\r\n\t\t\ttext-shadow: ' \
                   '5px 5px 15px black !important;\r\n\t\t\tfont-size: 40px !important;\r\n\t\t\tcolor: white !important;' \
                   '\r\n\t\t\tmargin-top: 40px !important;\r\n\t\t\tfont-weight: bold !important;\r\n\t\t}\r\n\r\n\t\t.body-38e00bc3 ' \
                   '{\r\n\t\t\tbackground: #f2f2f2 !important;\r\n\t\t\tfont-size: 1.25em !important;\r\n\t\t\t' \
                   'font-family: \'Raleway\', sans-serif !important;\r\n\t\t\theight: 900px !important;' \
                   '\r\n\t\t}\r\n\t\t\r\n\t\t.container-38e00bc3 {\r\n\t\t\tmargin: 50px auto !important; \r\n\t\t\t' \
                   'text-align: center !important;\r\n\t\t}\r\n\t\t\r\n\t\t.container-38e00bc3 .questions-wrapper-38e00bc3 ' \
                   '{\r\n\t\t\tmargin: auto !important;\r\n\t\t\tpadding-left: 0 !important;\r\n\t\t\twidth: 90% !important;' \
                   '\r\n\t\t\tlist-style: none !important;\r\n\t\t\tposition: relative !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 ' \
                   '.subquestions-wrapper-38e00bc3 {\r\n\t\t\tmargin: auto !important;\r\n\t\t\tpadding-left: 0 !important;\r\n\t\t\t' \
                   'width: 100% !important;\r\n\t\t\tlist-style: none !important;\r\n\t\t\tposition: relative !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .result-container-38e00bc3 {\r\n\t\t\tpadding: 20px 15px !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .result-title-38e00bc3 {\r\n\t\t\tmargin-bottom: 25px !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .result-authors-38e00bc3,\r\n\t\t.container-38e00bc3 .result-date-38e00bc3 {\r\n\t\t\
                   font-style: italic !important;\r\n    \t\tfont-size: 18px !important;\r\n\t\t\tmargin-bottom: 10px !important;' \
                   '\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .result-text-38e00bc3 {\r\n\t\t\tfont-size: 16px;\r\n\t\t\t' \
                   'text-indent: 20px !important;\r\n\t\t\ttext-align: justify !important;\r\n\t\t\tline-height: 22px !important;' \
                   '\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .result-highlight-38e00bc3 {\r\n\t\t\tbackground: yellow !important;' \
                   '\r\n\t\t\tpadding: 2px 6px !important;\r\n    \t\tbox-shadow: 1px 1px 10px grey !important;\r\n\t\t}\r\n\t</style>' \
                   '\r\n\r\n\t<!-- This is styles for the menu items -->\r\n\t<style>\r\n\t\t.container-38e00bc3 .question-38e00bc3 ' \
                   '.accordeon-38e00bc3 {\r\n\t\t\tdisplay: block !important; \r\n\t\t\tpadding: 15px 10px !important;\r\n\t\t\t' \
                   'color: #05a520 !important; \r\n\t\t\ttext-decoration: none !important;\r\n\t\t\tfont-size: 20px !important;\r\n\t\t\t' \
                   'background: #cefdd4 !important;\r\n\t\t\tborder-bottom: 3px solid #05a520 !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 ' \
                   '.subquestion-38e00bc3 .accordeon-38e00bc3 {\r\n\t\t\tdisplay: block !important; \r\n\t\t\tpadding: 10px 25px !important;\r\n\t\t\t' \
                   'color: #ff8000 !important; \r\n\t\t\ttext-decoration: none !important;\r\n\t\t\tfont-size: 20px !important;\r\n\t\t\t' \
                   'background: #ffffcc !important;\r\n\t\t\tborder-bottom: 2px solid #ff8000 !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3 .triangle-38e00bc3,\r\n\t\t.container-38e00bc3 .subquestion-38e00bc3 ' \
                   '.accordeon-38e00bc3 .triangle-38e00bc3 {\r\n\t\t\tfloat: right !important;\r\n\t\t\twidth: 0 !important; \r\n\t\t\t' \
                   'height: 0 !important;\r\n\t\t\tmargin-top: 10px !important;\r\n\t\t    ' \
                   'border-left: 10px solid transparent !important; \r\n\t\t    border-right: 10px solid transparent !important; ' \
                   '\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3 .triangle-38e00bc3 {\r\n\t\t\t' \
                   'border-top: 10px solid #05a520 !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3 ' \
                   '.triangle-38e00bc3 {\r\n\t\t\tborder-top: 10px solid #ff8000 !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 ' \
                   '.question-38e00bc3 .accordeon-38e00bc3.active-38e00bc3 .triangle-38e00bc3 {\r\n\t\t\tborder-bottom: 10px solid #05a520 ' \
                   '!important;\r\n\t\t\tborder-top: none !important;\r\n\t\t\tmargin-top: 5px !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3.active-38e00bc3 .triangle-38e00bc3 {\r\n\t\t\t' \
                   'border-bottom: 10px solid #ff8000 !important;\r\n\t\t\tborder-top: none !important;\r\n\t\t\t' \
                   'margin-top: 5px !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3:hover {\r\n\t\t\t' \
                   'background: #05a520 !important;\r\n\t\t\tcolor: #fff !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .subquestion-38e00bc3 ' \
                   '.accordeon-38e00bc3:hover {\r\n\t\t\tbackground: #ff8000 !important;\r\n\t\t\tcolor: #fff !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .question-38e00bc3 .accordeon-38e00bc3:hover .triangle-38e00bc3 {\r\n\t\t\tborder-top: 10px solid #fff ' \
                   '!important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3:hover .triangle-38e00bc3 ' \
                   '{\r\n\t\t\tborder-top: 10px solid #fff !important;\r\n\t\t}\r\n\r\n\t\t.container-38e00bc3 .question-38e00bc3 ' \
                   '.accordeon-38e00bc3.active-38e00bc3:hover .triangle-38e00bc3 {\r\n\t\t\tborder-bottom: 10px solid #fff ' \
                   '!important;\r\n\t\t\tborder-top: none !important;\r\n\t\t\tmargin-top: 5px !important;\r\n\t\t}\r\n\r\n\t\t' \
                   '.container-38e00bc3 .subquestion-38e00bc3 .accordeon-38e00bc3:hover .triangle-38e00bc3 {\r\n\t\t\t' \
                   'border-top: 10px solid #fff !important;\r\n\t\t}\r\n\t\r\n\t</style>\r\n\r\n\t<!-- This is styles for the accordeon functionality -->' \
                   '\r\n\t<style>\r\n\t\t.container-38e00bc3 .accordeon-38e00bc3 {\r\n\t\t  cursor: pointer !important;\r\n\t\t  width: 100% !important;' \
                   '\r\n\t\t  border: none !important;\r\n\t\t  text-align: left !important;\r\n\t\t  outline: none !important;\r\n\t\t' \
                   '  transition: 0.4s !important;\r\n\t\t}\r\n\t\t\r\n\t\t.container-38e00bc3 .panel-38e00bc3 {\r\n\t\t  text-align: left;' \
                   '\r\n\t\t  display: none;\r\n\t\t  overflow: hidden;\r\n\t\t}\r\n\t</style>\r\n\r\n\t<script>\r\n\t\t' \
                   'const accordeon = document.getElementsByClassName("accordeon-38e00bc3");\r\n\t\t\r\n\t\t' \
                   'for (let i = 0; i < accordeon.length; i += 1) {\r\n\t\t\taccordeon[i].addEventListener("click", function() ' \
                   '{\r\n\t\t\t\tthis.classList.toggle("active-38e00bc3");\r\n\r\n\t\t\t\tconst panel = this.nextElementSibling;' \
                   '\r\n\t\t\t\tpanel.style.display = panel.style.display === "block" ? "none" : "block";\r\n\t\t    });\r\n\t\t}' \
                   '\r\n\t</script>\r\n\r\n</html>\r\n\r\n\r\n'
    return html_string


def create_question(question, subquestions, n_question, answers_dict):
    new_question = '<li class="question-38e00bc3"><button class="accordeon-38e00bc3">'
    new_question += question
    new_question += '<div class="triangle-38e00bc3"></div></button><div class="panel-38e00bc3">' \
                    '<ul class="subquestions-wrapper-38e00bc3">'
    for n_subquestion, subquestion in enumerate(subquestions):
        new_question += create_subquestion(subquestion, n_question, n_subquestion, answers_dict)

    new_question += '</ul></div></li>'

    return new_question


def create_subquestion(subquestion, n_question, n_subquestion, answers_dict):
    new_subquestion = '<li class="subquestion-38e00bc3"><button class="accordeon-38e00bc3">'
    new_subquestion += subquestion
    new_subquestion += '<div class="triangle-38e00bc3"></div></button><div class="panel-38e00bc3 result-container-38e00bc3">'

    for element in answers_dict[n_question][n_subquestion]:
        new_subquestion += create_element(element)

    new_subquestion += '</div></li>'

    return new_subquestion


def create_element(element):
    title = element['title']
    url = element['url']
    authors = element['authors']
    evidence = element['evidence']
    design = element['design']
    date = element['date']
    sentences = element['sentences']

    new_element = '<h3 class="result-title-38e00bc3">' \
                  '<a href="' + url + '" target="_blank">' + title + '</a></h3><p class="result-authors-38e00bc3">'
    new_element += authors + '</p><p class="result-date-38e00bc3">' + date + '</p><p class="result-text-38e00bc3">'
    new_element += '<br>'.join(sentences)
    new_element += '</p>'

    return new_element