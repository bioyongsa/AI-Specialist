from typing import List
import requests
import numpy as np
import bz2
import json
from blingfire import text_to_sentences_and_offsets
from collections import defaultdict
from typing import Any, Dict, List
from bs4 import BeautifulSoup
import os
import openai

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, get_response_synthesizer, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

import textwrap

import nltk
nltk.download('punkt')

def extract_chunks(search_results):
    # Initialize a list to hold all extracted sentences from the search results.
    all_chunks = []

    # Process each HTML text from the search results to extract text content.
    for html_text in search_results:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_text["page_result"], features="lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
        if not text:
            # If no text is extracted, add an empty string as a placeholder.
            all_chunks.append("")
        else:

            # Extract offsets of sentences from the text
            _, offsets = text_to_sentences_and_offsets(text)

            # Initialize a list to store sentences
            chunks = []

            # Iterate through the list of offsets and extract sentences
            for start, end in offsets:
                # Extract the sentence and limit its length
                chunk = text[start:end][:1000]
                all_chunks.append(chunk)

    return all_chunks

class BaseRetriever:
  def __init__(self):
    self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

  def retrieve(self, query, search_results, topk):
    # get chunks
    all_chunks = extract_chunks(search_results)

    # Generate embeddings for all chunks and the query.
    all_embeddings = self.sentence_model.encode(all_chunks, normalize_embeddings=True)
    query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)[None, :]

    # Calculate cosine similarity between query and sentence embeddings, and select the top sentences.
    cosine_scores = (all_embeddings * query_embedding).sum(1)
    top_k_chunks = np.array(all_chunks)[(-cosine_scores).argsort()[:topk]]

    return top_k_chunks

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class LlamaIndexRetriever:
  def __init__(self):
      self.parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

  def retrieve(self, query, search_results, topk):
      documents = []
      for html_text in search_results:
        soup = BeautifulSoup(html_text["page_result"], features="lxml")
        text = soup.get_text(" ", strip=True)
        if not text:
            documents.append(Document(text=""))
        else:
            documents.append(Document(text=text))

      base_index = VectorStoreIndex.from_documents(documents = documents, transformations=[self.parser])

      base_retriever = base_index.as_retriever(similarity_top_k=topk)

      retrieved_nodes = base_retriever.retrieve(query)

      retrieved_results = [retrieved_node.node.get_content().strip() for retrieved_node in retrieved_nodes]

      return retrieved_results
  
  ### YOUR CODE HERE ###

entity_extract_template = """
You are given a Query and Query Time. Do the following:

1) Determine the domain the query is about. The domain should be one of the following: "finance", "sports", "music", "movie", "encyclopedia". If none of the domain applies, use "other". Use "domain" as the key in the result json.

2) Extract structured information from the query. Include different keys into the result json depending on the domains, and put them DIRECTLY in the result json. Here are the rules:

For `finance` queries, these are possible keys:
- `market_identifier`: stock identifiers including individual company names, stock symbols.
- `metric`: financial metrics that the query is asking about. This must be one of the following: `price`, `dividend`, `P/E ratio`, `EPS`, `marketCap`, and `other`.
- `datetime`: time frame that query asks about. When datetime is not explicitly mentioned, use `Query Time` as default.


Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

### YOUR CODE HERE ###

def prompt_generator(query):
    user_message = ""
    user_message += f"Query: {query}\n"

    llm_input = [
      {"role": "system", "content": entity_extract_template},
      {"role": "user", "content": user_message},
    ]

    return llm_input

### YOUR CODE HERE ###

import json
from openai import OpenAI
from json import JSONDecoder

oai_client = OpenAI()

def generate_query(query):
    llm_input = prompt_generator(query)
    completion = oai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=
    llm_input
    ).choices[0].message.content

    try:
        completion = json.loads(completion)
    except:
        completion = extract_json_objects(completion)

    if "domain" in completion.keys():
        domain = completion["domain"]
        is_finance = domain == "finance"
    else:
        is_finance = False

    return completion, is_finance

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data
    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results

### YOUR CODE HERE ###

from datetime import timedelta
from dateutil import parser
import pytz
import re

def normalize_key(key):
    return re.sub(r'[^a-zA-Z0-9]', '', key).lower()

def get_metric_from_response(response, metric):
    normalized_metric = normalize_key(metric)
    if response != None:
        for key, value in response.items():
            if normalize_key(key) == normalized_metric:
                return value
    return None


def convert_to_standard_format(date_string):
    try:
        dt = parser.parse(date_string)

        est = pytz.timezone('US/Eastern')

        if dt.tzinfo is None:
            dt = est.localize(dt)
        else:
            dt = dt.astimezone(est)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        return formatted_date
    except (ValueError, OverflowError) as e:
        return date_string

def add_one_day(date_string):
    try:
        dt = parser.parse(date_string)

        est = pytz.timezone('US/Eastern')

        if dt.tzinfo is None:
            dt = est.localize(dt)
        else:
            dt = dt.astimezone(est)

        dt_plus_one = dt + timedelta(days=1)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        formatted_date = dt_plus_one.strftime('%Y-%m-%d %H:%M:%S %Z')
        return formatted_date
    except (ValueError, OverflowError) as e:
        return f"Invalid date string: {e}"

def subtract_one_day(date_string):
    try:
        dt = parser.parse(date_string)

        est = pytz.timezone('US/Eastern')

        if dt.tzinfo is None:
            dt = est.localize(dt)
        else:
            dt = dt.astimezone(est)

        dt_minus_one = dt - timedelta(days=1)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        formatted_date = dt_minus_one.strftime('%Y-%m-%d %H:%M:%S %Z')
        return formatted_date
    except (ValueError, OverflowError) as e:
        return f"Invalid date string: {e}"
    
class CRAG(object):
    def __init__(self, server = None):
        if server == None:
            self.server = os.environ.get('CRAG_SERVER', "http://127.0.0.1:8000")
        else:
            self.server = server
            
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def finance_get_company_name(self, query:str):
        url = self.server + '/finance/get_company_name'
        headers={'accept': "application/json"}
        data = {'query': query}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_ticker_by_name(self, query:str):
        url = self.server + '/finance/get_ticker_by_name'
        headers={'accept': "application/json"}
        data = {'query': query}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_price_history(self, ticker_name:str):
        url = self.server + '/finance/get_price_history'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_detailed_price_history(self, ticker_name:str):
        url = self.server + '/finance/get_detailed_price_history'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_dividends_history(self, ticker_name:str):
        url = self.server + '/finance/get_dividends_history'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_market_capitalization(self, ticker_name:str):
        url = self.server + '/finance/get_market_capitalization'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_eps(self, ticker_name:str):
        url = self.server + '/finance/get_eps'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_pe_ratio(self, ticker_name:str):
        url = self.server + '/finance/get_pe_ratio'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_info(self, ticker_name:str):
        url = self.server + '/finance/get_info'
        headers={'accept': "application/json"}
        data = {'query': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    
### YOUR CODE HERE ###

import copy

class KGQueryEngine:
    def __init__(self, server = None):
        self.api = CRAG(server)

    def get_finance_kg_results(self, generated_query):
        formatted_time_list = []
        if 'datetime' in generated_query:
            datetime_list = generated_query['datetime'].split(' - ')
            for datetime in datetime_list:
                formatted_time_list.append(convert_to_standard_format(datetime.strip()))


        kg_results = []
        res = ""
        if "market_identifier" in generated_query.keys() and generated_query["market_identifier"] is not None:
            if isinstance(generated_query["market_identifier"], str):
                company_names = generated_query["market_identifier"].split(",")
            else:
                company_names = generated_query["market_identifier"]

            for company_name in company_names:
                try:
                    res = self.api.finance_get_company_name(company_name)["result"]

                    if res == []:
                        ticker_name = company_name.upper()
                    else:
                        ticker_name = self.api.finance_get_ticker_by_name(res[0])["result"]

                    if generated_query['metric'].lower().strip() == 'price':
                        response = self.api.finance_get_price_history(ticker_name)['result']
                    elif generated_query['metric'].lower().strip() == 'dividend':
                        response = self.api.finance_get_dividends_history(ticker_name)['result']
                    elif generated_query['metric'].lower().strip() == 'p/e ratio':
                        response = self.api.finance_get_pe_ratio(ticker_name)['result']
                    elif generated_query['metric'].lower().strip() == 'eps':
                        response = self.api.finance_get_eps(ticker_name)["result"]
                    elif generated_query['metric'].lower().strip() == 'marketcap' :
                        response = self.api.finance_get_market_capitalization(ticker_name)['result']
                    else:
                        response = self.api.finance_get_info(ticker_name)['result']
                        metric_value = get_metric_from_response(response, generated_query['metric'])
                        if metric_value is not None:
                            response = metric_value

                    try:
                        for formatted_time in formatted_time_list:
                            if formatted_time in response:
                                filtered_response = copy.deepcopy(response[formatted_time])
                            elif add_one_day(formatted_time) in response:
                                filtered_response = copy.deepcopy(response[add_one_day(formatted_time)])
                            elif subtract_one_day(formatted_time) in response:
                                filtered_response = copy.deepcopy(response[subtract_one_day(formatted_time)])
                            else:
                                filtered_response = copy.deepcopy(response)
                            kg_results.append({company_name + " " + generated_query["metric"]: filtered_response, 'time': formatted_time})
                    except:
                        kg_results.append({company_name + " " + generated_query["metric"]: response})

                except Exception as e:
                    print("Fail to parse the generated query")
                    pass

        kg_results = "<DOC>\n".join([str(res) for res in kg_results]) if len(kg_results) > 0 else ""
        return  kg_results

    def query(self, query):
        generated_query, is_finance = generate_query(query)
        
        if is_finance:
            kg_results = self.get_finance_kg_results(generated_query)
        else:
            kg_results = ""

        return kg_results

    def generate_query(self, query):
        llm_input = self.prompt_generator(query)
        completion = oai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        top_p=0,
        messages=
        llm_input
        ).choices[0].message.content

        try:
            completion = json.loads(completion)
        except:
            completion = extract_json_objects(completion)

        if "domain" in completion.keys():
            domain = completion["domain"]
            is_finance = domain == "finance"
        else:
            is_finance = False

        return completion, is_finance
    
    def prompt_generator(self, query):
        user_message = ""
        user_message += f"Query: {query}\n"

        llm_input = [
          {"role": "system", "content": entity_extract_template},
          {"role": "user", "content": user_message},
        ]

        return llm_input
    
    
### YOUR CODE HERE ###

from openai import OpenAI

oai_client = OpenAI()

class Reader:
  def __init__(self):

    self.system_prompt = """
    You are provided with a question and various references.
    Your task is to answer the question succinctly, using the fewest words possible.
    If the references do not contain the necessary information to answer the question, respond with 'I don't know'.
    There is no need to explain the reasoning behind your answers.
    """

  def generate_response(self, question: str, top_k_chunks: list) -> str:
      """
      Generate answer from context.
      """
      llm_input = self.prompt_generator(question, top_k_chunks)
      completion = oai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0,
      messages=
      llm_input
      ).choices[0].message.content
      return completion

  def prompt_generator(self, query, top_k_chunks):
      user_message = ""
      references = ""

      if len(top_k_chunks) > 0:
          references += "# References \n"
          # Format the top sentences as references in the model's prompt template.
          for chunk_id, chunk in enumerate(top_k_chunks):
              references += f"- {chunk.strip()}\n"

      references = references[:4000]
      # Limit the length of references to fit the model's input size.

      user_message += f"{references}\n------\n\n"
      user_message
      user_message += f"Using only the references listed above, answer the following question: \n"
      user_message += f"Question: {query}\n"

      llm_input = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_message},
      ]

      return llm_input
