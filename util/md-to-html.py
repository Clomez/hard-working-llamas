from markdown2 import Markdown
import sys
from pathlib import Path
from datetime import datetime
import time

file_to_load = "./huberman_final01.md"
output_file = "output/huberman_index.html"
title="Huberman index test 02"
date_time = datetime.now()

def head():    
    html = "<!doctype html>\n"
    html += "<html>\n"
    return html

def meta(html):
    html += "<head>\n"


    html += f"<title>{title}</title>\n"
    html += f"<meta name='creation-date' content='{date_time}' >\n"
    html += "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css' >\n"
    
    html += "</head>\n"
    return html

def startBody():
    return """
    <body>\n
    <section>
    <div class="container">
        <div class="row">
            <div class="col-sm-6 col-md-8 col-14 pb-6">
    """

def convertmd(html):
    markdowner = Markdown()

    with open(file_to_load, 'r') as f:
        raw_text = f.read()
        text = raw_text.replace("Answer:", "")
        text = text.replace("---------------------", "")

        raw_html = markdowner.convert(text)

    html += raw_html
    html += """
    </div>
    </div>
    </div>
    </body>
    </html>
    """

    time.sleep(1)
    with open(output_file, 'w') as z:
        z.write(html)

html = head()
html = meta(html)
html += startBody()
convertmd(html)