import os
import time
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

st.set_page_config(
    page_title="AI financial analyst agents",
)

st.title("💬 Agentic Finanical Analysts")
st.caption("🚀 A set of financial agents that can generate, validate and iterate on financial statements")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "previous_prompt" not in st.session_state:
    st.session_state.previous_prompt = ""

if "previous_analysis" not in st.session_state:
    st.session_state.previous_analysis = ""

if "human_feedback" not in st.session_state:
    st.session_state.human_feedback = ""

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        with st.chat_message("Tool"):
            st.markdown(message.content)
    else:
        with st.chat_message("Agent"):
            st.markdown(message.content)

# Using object notation

focus = st.sidebar.text_area("Focus in", 
    "variable market dynamics, customer demographics and long running trends across segments and geographical regions",
    height=150
)

loops = st.sidebar.slider("Number of iterations", min_value=2, max_value=10, value=3, step=1)

add_analyst_prompt = st.sidebar.text_area("Analyst Prompt",
    "Analyse the financial statements of the company and provide valuable insights for market opportunities and unique challenges.\
Financial performance analysis is the process of assessing a company’s financial health and making informed decisions by analyzing key metrics and techniques. It involves reviewing financial statements, such as the balance sheet, income statement, cash flow statement, and annual report, to gain insights into profitability, liquidity, solvency, efficiency, and valuation.\n \
Financial KPIs, or key performance indicators, are metrics used to track, measure, and analyze the financial health of a company. These KPIs fall under categories like profitability, liquidity, efficiency, solvency, and valuation. They include metrics such as gross profit margin, current ratio, inventory turnover, debt-to-equity ratio, and price-to-earnings ratio.\
Improve the given statements given this feedback and the available raw insights. Make sure you name the url for reference for the major statements.",
    height=300
)

report_length = st.sidebar.select_slider("Report Length", options=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], value=400)

add_reviewer_prompt = st.sidebar.text_area("Reviewer Prompt",
    "You need to review the financial analyst statements, analyse the insights available and the conclusions made from these statements.\
Put a special focus on the insights and the statements under the given specialization and provide concrete feedback on how this should be improved.\
Make sure that url references are provided for the major statements. \
Make sure that you highlight the key points of your thinking in the reasoning output. Make sure that your feedback is presented as a numbered list of recommendations.",
    height=300
)

credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]) if "AZURE_AI_SEARCH_KEY" in os.environ else DefaultAzureCredential()
index_name = os.getenv("AZURE_AI_SEARCH_INDEX")
search_client = SearchClient(
    endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"], 
    index_name=index_name,
    credential=credential
)

chat_model: AzureChatOpenAI = None
openai: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None

if "AZURE_OPENAI_API_KEY" in os.environ:
    openai = AzureOpenAI(
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
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    openai = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version = "2024-08-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    chat_model = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )

def llm(x):
    return chat_model.invoke(x).content

class Rating(BaseModel):
    feedbackResolved: bool = Field(
        ...,
        description="Has the feedback been resolved in the statements",
    )
    reasoning: str = Field(
        ...,
        description="The reasoning behind the rating",
   )  

def model_rating(input) -> Rating:
    completion = openai.beta.chat.completions.parse(
        model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        messages = [{"role" : "assistant", "content" : f""" Help me understand the following by giving me a response to the question, a short reasoning on why the response is correct:  {input}"""}],
        response_format = Rating)
    
    return completion.choices[0].message.parsed

class Statement(BaseModel):
    response: str = Field(
        ...,
        description="The response to the question",
    )
    reasoning: str = Field(
        ...,
        description="The reasoning behind the response",
    )
    certainty: float = Field(
        ...,
        description="The certainty of the correctness of the response",
    )

def model_response(input) -> Statement:
    completion = openai.beta.chat.completions.parse(
        model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        messages = [{"role" : "assistant", "content" : f""" Help me understand the following by giving me a response to the question, a short reasoning on why the response is correct and a rating on the certainty on the correctness of the response:  {input}"""}],
        response_format = Statement)
    
    return completion.choices[0].message.parsed

class Objective(BaseModel):
    urls: List[str]
    question: str

def model_objective(input) -> Objective:
    completion = openai.beta.chat.completions.parse(
        model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        messages = [{"role" : "assistant", "content" : f"""Extract all the urls in the following input and the objective that was asked in the form of a question. Input: {input}"""}],
        response_format = Objective)
    
    print(completion)
    return completion.choices[0].message.parsed

def get_embedding(text, embeddingsmodel=embeddings_model):
    if len(text) == 0:
        return openai.embeddings.create(input = "no description", model=embeddingsmodel).data[0].embedding
    return openai.embeddings.create(input = [text], model=embeddingsmodel).data[0].embedding

@tool
def search_for_company(question: str) -> str:
    """This will return more detailed information about companies. Returns top 10 results."""
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

    ai_msg = chat_model.invoke(input).content
    
    print("found information:")

    return ai_msg

@st.cache_data
def load_financial_report(url: Annotated[str, "Full qualified url of the report to download. Example: https://annualreport2023.volkswagen-group.com/divisions/volkswagen-financial-services.html"]) -> str:
    """This tool loads financial reports from the web and returns the content"""

    doc = WebBaseLoader(url).load()[0]
    print("loaded document:")
    # print(doc)

    content = "Reference: " + doc.metadata["title"] + " URL: " + url + "content: " + doc.page_content
    return content

def prepare_flow(input:str) -> TypedDict:

    with st.spinner('Retrieving content..'):

        objective = model_objective(input)
        question = ""
        reports = ""
        
        if st.session_state.previous_prompt is not None:
            question += "This was the original objective: "+ st.session_state.previous_prompt + ". I want to start over with the analysis and also include this feedback: " + objective.question
            st.session_state.human_feedback = input
            st.session_state.previous_prompt += objective.question
        else:
            question = objective.question
            st.session_state.previous_prompt = objective.question

        if st.session_state.previous_analysis is not None:
            reports += "This was the output with the previous analyis: "+ st.session_state.previous_analysis + " This is the input that was used to generate this output: "

        reports += retrieve_information(input)  

        for url in objective.urls:
            reports += load_financial_report(url)

    st.success("Done!")

    inputs = {
        "history": "",
        "insights":reports,
        "statements": "",
        'original_statements':"",
        "specialization":question,
        'iterations':0}       

    return inputs

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

st.image("diagram.png", caption="Process description")

class ReportState(TypedDict):
    all_feedback_resolved: Optional[bool] = None
    feedback: Optional[str] = None
    history: Optional[str] = None
    statements: Optional[str] = None
    specialization: Optional[str] = None
    insights: Optional[str] = None
    rating: Optional[str] = None
    iterations: Optional[int]=None
    statements_compare: Optional[str] = None
    original_statements: Optional[str] = None
    messages: Annotated[Sequence[BaseMessage], add_messages] = []

workflow = StateGraph(ReportState)

### Nodes

reviewer_start= "You are a senior financial reviewer with extensive experience in comparing financial reports with a special attentions to {}.\
{} \
Your job is to write pointed and focussed feedback on the financial analyst statements and recommend at most three concrete recommendations on what should be added, removed or changed to improve the quality of the insights in the report.\
If the statements are suitable for the provided objective, you should state that the statements are suitable for the objective and nothing needs to be changed. \
You should also make sure that the document is structured in a way that is easy to read and understand according to the specialisation and focus requirements.\
Your feedback should be clear and concise and should be based on the insights and the statements provided in the form of a numbered list of recommendations.\
Your feedback should be about 500 words and be shared in the form of a numbered list of proposed changes.\
{} \
Insights:\n {} \
Statements: \n {}"

def handle_reviewer(state):
    print("starting reviewer ...")
    # print(state)
    history = state.get('history', '').strip()
    statements = state.get('statements', '').strip()
    insights = state.get('insights', '').strip()
    specialization = state.get('specialization','').strip()
    iterations = state.get('iterations','')
    messages = state.get('messages')
    print("reviewer working...")

    human_feedback = ""
    if st.session_state.human_feedback is not None:
        human_feedback = "Make sure that you include this feedback in your analysis: " + st.session_state.human_feedback

    feedback = model_response(reviewer_start.format(specialization, add_reviewer_prompt, human_feedback, insights,statements))
    
    if iterations > 0:
        st.info('This is iteration ' + str(iterations), icon="ℹ️")
        messages.append(AIMessage(content="Reviewer (" + str(feedback.certainty) +  "): "+feedback.reasoning + " \n\n My Feedback: \n\n " + feedback.response))
    print("reviewer done")
    print(feedback)

    return {'history':history+"\n REVIEWER:\n"+feedback.response,'feedback':feedback.response,'iterations':iterations+1, 'insights': insights, 'messages':messages}

analyst_start = "You are an financial consultant specialized and focussed on {}.\
{} \
Feedback:\n {} \n \
Statements:\n {} \n \
Insights:\n {} \n \
Output just the revised statements and add nothing else. Make sure that you document your reasoning for the major statements in the output.\
Your output should not exceed {} words."

def handle_analyst(state):
    print("starting analyst...")
    # print(state)
    history = state.get('history', '').strip()
    feedback = state.get('feedback', '').strip()
    insights = state.get('insights', '').strip()
    statements =  state.get('statements','').strip()
    iterations = state.get('iterations','')
    specialization = state.get('specialization','').strip()
    messages = state.get('messages')
    print("analyst rewriting...")
    
    analsis = model_response(analyst_start.format(specialization,add_analyst_prompt, feedback,statements,insights,report_length))

    if iterations > 0:
        st.info('This is iteration ' + str(iterations), icon="ℹ️")
        messages.append(AIMessage(content="Analyst (" + str(analsis.certainty) +  "): \n\n "+analsis.reasoning))
        messages.append(SystemMessage(content=statements))

    print("analyst done")
    return {'history':history+'\n STATEMENTS:\n'+analsis.response,'statements':analsis.response, 'iterations':iterations, 'insights': insights, 'messages':messages}

statement_comparison = "Compare the two statements and rate on a scale of 10 to both. Dont output the statements. Revised statements: \n {} \n Original statements: \n {}"

rating_start = "Rate the skills of the financial insights on a scale of 10 given the statement review cycle with a short reason.\
Statement review:\n {} \n "

def handle_result(state):
    print("Review done...")
    
    history = state.get('history', '').strip()
    code1 = state.get('statements', '').strip()
    code2 = state.get('original_statements', '').strip()
    messages = state.get('messages')
    iterations = state.get('iterations','')
    rating  = model_response(rating_start.format(history))
    
    messages.append(AIMessage(content="Rating (" + str(rating.certainty) +  "): "+rating.reasoning))

    statements_compare = llm(statement_comparison.format(code1,code2))

    messages.append(AIMessage(content="Result: "+statements_compare))

    messages.append(SystemMessage(content=code1))

    return {'rating':rating,'code_compare':statements_compare, 'iterations':iterations, 'messages':messages}

classify_feedback_start = "Are most of the important feedback points mentioned resolved in the statements? Output just Yes or No with a reason.\
Statements: \n {} \n Feedback: \n {} \n"

def classify_feedback(state):
    print("Classifying feedback...")
    # print(state)
    rating = model_rating(classify_feedback_start.format(state.get('statements'),state.get('feedback')))

    state['all_feedback_resolved'] = rating.feedbackResolved
    messages = state.get('messages')
    messages.append(AIMessage(content="Feedback resolved: "+ str(rating.feedbackResolved) + " \n\n Reasoning: "+rating.reasoning))
    state['messages'] = messages

    return state

# Define the nodes we will cycle between
workflow.add_node("handle_reviewer",handle_reviewer)
workflow.add_node("handle_analyst",handle_analyst)
workflow.add_node("handle_result",handle_result)
workflow.add_node("classify_feedback",classify_feedback)

def deployment_ready(state):
    deployment_ready = state['all_feedback_resolved']
    print("Deployment ready: " + str(deployment_ready))
    total_iterations = 1 if state.get('iterations')>5 else 0
    # print(state)
    if state.get('iterations')>loops:
        print("Iterations exceeded")
        return "handle_result"
    return "handle_result" if  deployment_ready or total_iterations else "handle_analyst" 


workflow.add_conditional_edges(
    "classify_feedback",
    deployment_ready,
    {
        "handle_result": "handle_result",
        "handle_analyst": "handle_analyst"
    }
)

workflow.set_entry_point("handle_analyst")
workflow.add_edge('handle_analyst', "handle_reviewer")
workflow.add_edge('handle_reviewer', "classify_feedback")
workflow.add_edge('handle_result', END)

# Compile

app = workflow.compile()

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    inputs = prepare_flow(human_query)

    config = {"recursion_limit":20}

    with st.chat_message("Human"):
        st.markdown(human_query)

    for event in app.stream(inputs, config):   
        print ("message TO streamlit: ")
        for value in event.values():
            if ( value["messages"].__len__() > 0 ):
                for message in value["messages"]:
                    if (message.content.__len__() > 0):
                        ticks = time.time()    
                        key = message.id + "-" + str(ticks)
                        if ( isinstance(message, AIMessage) ):
                            with st.chat_message("AI"):
                                st.write(message.content)
                        elif ( isinstance(message, SystemMessage) ):
                            with st.chat_message("human"):
                                st.write(message.content)
                                st.session_state.previous_analysis = message.content
                        else:
                            with st.chat_message("Agent"):
                                st.write(message.content)        
