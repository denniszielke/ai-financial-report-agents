import os
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import streamlit as st
import random
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Annotated, Literal, Sequence, TypedDict, Optional
from bs4 import BeautifulSoup
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import List
from azure.search.documents.models import (
    VectorizedQuery
)
from langchain_core.tools import tool

dotenv.load_dotenv()

chat_model: AzureChatOpenAI = None
llm: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None
llm = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2024-08-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)
chat_model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
    temperature=0,
    streaming=True
)
embeddings_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]) if len(os.environ["AZURE_AI_SEARCH_KEY"]) > 0 else DefaultAzureCredential()

index_name = os.getenv("AZURE_AI_SEARCH_INDEX")
search_client = SearchClient(
    endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"], 
    index_name=index_name,
    credential=credential
)

def get_embedding(text, model=embeddings_model):
    if len(text) == 0:
        return llm.embeddings.create(input = "no description", model=model).data[0].embedding
    return llm.embeddings.create(input = [text], model=model).data[0].embedding

@tool
def search_for_company(question: str) -> str:
    """This tool will return more detailed information about companies when given a question about companies. Returns top 10 results."""
    # create a vectorized query based on the question
    print("searching for company:")
    print(question)
    vector = VectorizedQuery(vector=get_embedding(question), k_nearest_neighbors=10, fields="contentVector")

    found_docs = list(search_client.search(
        search_text=None,
        query_type="semantic", query_answer="extractive",
        query_answer_threshold=0.8,
        semantic_configuration_name="default",
        vector_queries=[vector],
        select=["id", "title", "content", "filepath", "url"],
        top=12
    ))

    print("found docs:")

    print(found_docs)
    found_docs_as_text = " "
    for doc in found_docs:   
        print(doc) 
        found_docs_as_text += " "+ "Title: {}".format(doc["title"]) +" "+ "Content: {}".format(doc["content"]) +" "+ "Url: {}".format(doc["url"]) +" "

    return found_docs_as_text

tools = [search_for_company]

def retrieve_information(input:str) -> str:
    """This tool retrieves information from the web and returns the content"""

    chat_model.bind_tools(tools)

    ai_msg = chat_model.invoke(input)
    print (ai_msg)
    return ai_msg


