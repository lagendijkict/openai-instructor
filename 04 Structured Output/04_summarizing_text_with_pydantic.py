from dotenv import load_dotenv

import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-2024-08-06"

# --------------------------------------------------------------
# Text summarization
# --------------------------------------------------------------

"""
From: https://cookbook.openai.com/examples/structured_outputs_intro
"""


def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    html_content = soup.find("div", class_="mw-parser-output")
    content = "\n".join(p.text for p in html_content.find_all("p"))
    return content


urls = [
    # Article on CNNs
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    # Article on LLMs
    "https://wikipedia.org/wiki/Large_language_model",
    # Article on MoE
    "https://en.wikipedia.org/wiki/Mixture_of_experts",
]

content = [get_article_content(url) for url in urls]


summarization_prompt = """
You will be provided with content from an article about an invention.
Your goal will be to summarize the article following the schema provided.
Here is a description of the parameters:
- invented_year: year in which the invention discussed in the article was invented
- summary: one sentence summary of what the invention is
- inventors: array of strings listing the inventor full names if present, otherwise just surname
- concepts: array of key concepts related to the invention, each concept containing a title and a description
- description: short description of the invention
"""


class ArticleSummary(BaseModel):
    invented_year: int
    summary: str
    inventors: list[str]
    description: str

    class Concept(BaseModel):
        title: str
        description: str

    concepts: list[Concept]


def get_article_summary(text: str):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": summarization_prompt},
            {"role": "user", "content": text},
        ],
        response_format=ArticleSummary,
    )

    return completion.choices[0].message.parsed


summaries = []

# for i in range(len(content)):
#     print(f"Analyzing article #{i+1}...")
#     summaries.append(get_article_summary(content[i]))
#     print("Done.")

# print(summaries)
