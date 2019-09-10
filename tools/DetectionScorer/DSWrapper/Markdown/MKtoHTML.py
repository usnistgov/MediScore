import os
import time
import codecs
import argparse
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from pathlib import Path

from mdx_gfm import GithubFlavoredMarkdownExtension
from jinja2 import Environment, FileSystemLoader
# from bs4 import BeautifulSoup as bs

def create_html(markdown_html, output_path, markdown_folder_path):
    file_loader = FileSystemLoader(str(markdown_folder_path))
    env = Environment(loader=file_loader)
    template = env.get_template("base.html")
    template_variables = {"css_path": markdown_folder_path / "github-markdown.css", 
                          "markdown_body": markdown_html}    
    html = template.render(template_variables)
    output_file = codecs.open(output_path, "w", encoding="utf-8", errors="xmlcharrefreplace")
    output_file.write(html)
    output_file.write("\n")

# def prettify_html(html_path):
#     with open(html_path, "r") as f:
#         html_string = f.read()
#     soup = bs(html_string, features="html.parser")
#     pretty_html = soup.prettify(formatter="html")
#     with open(html_path, "w") as f:
#         f.write(pretty_html)

parser = argparse.ArgumentParser(description=None)
parser.add_argument("-m", "--markdown-path", help="path to the markdown file", type=Path, required=True)
parser.add_argument("-o", "--html-output", help="path to html output", type=Path, required=True)
args = parser.parse_args()

readme_name = "ReadMe.mk"
html_output_name = "ReadMe.html"
readme_path = args.markdown_path
html_path = args.html_output
markdown_folder_path = Path.cwd()

print("Converting ReadMe markdown file to html... ", end='', flush=True)
start = time.time()
input_file = codecs.open(readme_path, mode="r", encoding="utf-8")
text = input_file.read()
html = markdown.markdown(text, extensions=[GithubFlavoredMarkdownExtension(), CodeHiliteExtension()])
# html = markdown.markdown(text, extensions=[CodeHiliteExtension(css_class="markdown-body")])
print("Done. ({:.2f}s)".format(time.time() - start))

print("Writing html to '{}'".format(html_path))
create_html(html, html_path, markdown_folder_path)

# print("Prettifying html.")
# prettify_html(html_path)

print("Exit.")
