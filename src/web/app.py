import os
import time
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import streamlit as st
import random
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
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

# enable langchain instrumentation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry import trace, trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from token_counter import TokenCounterCallback
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

dotenv.load_dotenv()

st.set_page_config(
    page_title="AI financial analyst agents",
)

instrumentor = LangchainInstrumentor()

@st.cache_resource
def setup_tracing():

    exporter = AzureMonitorTraceExporter.from_connection_string(
        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
    )
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=60000)
    trace.get_tracer_provider().add_span_processor(span_processor)
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()
    return tracer

tracer = setup_tracing()

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

st.sidebar.title("Agents prompts")

focus = st.sidebar.text_area("Focus in", 
    "variable market dynamics, customer demographics and long running trends across segments and geographical regions",
    height=150
)

loops = st.sidebar.slider("Maximum number of iterations", min_value=2, max_value=10, value=3, step=1)

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

callback = TokenCounterCallback()

chat_model: AzureChatOpenAI = None
openai: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None

if "AZURE_OPENAI_API_KEY" in os.environ:
    openai = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = "2024-08-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    chat_model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True,
        # callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = "2024-08-01-preview", # os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    openai = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version = "2024-08-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    chat_model = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version= "2024-08-01-preview", #os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True,
        # callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )

def llm(x):
    return chat_model.invoke(x).content

class Statement(BaseModel):
    '''Statement response to a question with the reasoning and certainty'''
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
    statement_model = chat_model.with_structured_output(Statement)
    statement_prompt = f""" Help me understand the following by giving me a response to the question, a short reasoning on why the response is correct and a rating on the certainty on the correctness of the response:  {input}"""
    completion = statement_model.invoke(statement_prompt)
    
    return completion 

class Objective(BaseModel):
    '''Objective of the model to extract urls and the question'''
    urls: List[str] = Field(
        ...,
        description="A list of urls that were extracted from the input",
    )
    question: str = Field(
        ...,
        description="The question that was asked in the form of a question",
    )

def model_objective(input) -> Objective:
    objective_model = chat_model.with_structured_output(Objective)
    objective_prompt = f"""Extract all the urls in the following input and the objective that was asked in the form of a question. Ignore all the urls that end with pdf. Do not generate new urls that are not in the input. Input: {input}"""
    completion = objective_model.invoke(objective_prompt)
    
    return completion

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

    # print(found_docs)
    found_docs_as_text = " "
    for doc in found_docs:   
        # print(doc) 
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
    print(url)

    content = "Reference: " + doc.metadata["title"] + " URL: " + url + "content: " + doc.page_content
    return content

def prepare_flow(input:str) -> TypedDict:
    
    downloadbar = st.progress(0, "Retrieving content..")

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

    if objective.urls and len(objective.urls) > 0:
        status = round(100/len(objective.urls))
        counter = 1

        for url in objective.urls:
            currentstatus = counter * status
            downloadbar.progress(currentstatus, "Downloading content from " + url + "...")
            reports += load_financial_report(url)
            counter = counter + 1

    downloadbar.empty()

    inputs = {
        "history": "",
        "insights":reports,
        "statements": "",
        'original_statements':"",
        "specialization":question,
        'iterations':0}       

    print("inputs:")
    print(inputs)

    return inputs

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    # st.write("You are running in session: " + st.session_state["session_id"])

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

    with st.spinner('Reviewer checking content..'):   
        feedback = model_response(reviewer_start.format(specialization, add_reviewer_prompt, human_feedback, insights,statements))

    message_id = str(iterations) + "-reviewer-" + str(hash(feedback.response))

    messages.append(AIMessage(id=message_id, name="Reviewer (gpt-4o - v0.1)", content= "My reasoning:  \n\n " + feedback.reasoning + " \n\n My Feedback: \n\n " + feedback.response))
    print("reviewer done " + message_id)

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

    history = state.get('history', '').strip()
    feedback = state.get('feedback', '').strip()
    insights = state.get('insights', '').strip()
    statements =  state.get('statements','').strip()
    iterations = state.get('iterations','')
    specialization = state.get('specialization','').strip()
    messages = state.get('messages')
    print("analyst rewriting...")
    
    with st.spinner('Analyst generating content..'):
        analsis = model_response(analyst_start.format(specialization,add_analyst_prompt, feedback,statements,insights,report_length))

    ticks = time.time()
    message_id = str(iterations) + "-analyst-" + str(ticks)
    messages.append(AIMessage(id=message_id, name="Analyst (gpt-4 - v0.2)", content= "My reasoning:  \n\n "+ analsis.reasoning + " \n\n My Statements: \n\n " + analsis.response))

    print("analyst done " + message_id)
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
    
    with st.spinner('Checking if the statements are suitable for the objective..'):
        rating  = model_response(rating_start.format(history))
        messages.append(AIMessage(name="Inspector (gpt-4 - v0.2)", content="Rating (" + str(rating.certainty) +  "): "+rating.reasoning))
        
        statements_compare = llm(statement_comparison.format(code1,code2))
        messages.append(SystemMessage(content=code1))

    return {'rating':rating,'code_compare':statements_compare, 'iterations':iterations, 'messages':messages}

class Rating(BaseModel):
    '''Rating of the feedback'''
    feedbackResolved: bool = Field(
        ...,
        description="Has the feedback been resolved in the statements true or false",
    )
    reasoning: str = Field(
        ...,
        description="The reasoning behind the rating with a small explanation",
   )  

def model_rating(input) -> Rating:
    rating_model = chat_model.with_structured_output(Rating)
    rating_prompt = f""" Help me understand the following by giving me a response to the question,
      a short reasoning on why the response is correct:  {input}"""
    completion = rating_model.invoke(rating_prompt)    
    return completion

classify_feedback_start = "Are most of the important feedback points mentioned resolved in the statements? Output just Yes or No with a reason.\
Statements: \n {} \n Feedback: \n {} \n"

def classify_feedback(state):
    print("Classifying feedback...")
    iterations = state.get('iterations','')
    with st.spinner('Inspector checking if the feedback from the reviewer has been implemented..'):
        rating = model_rating(classify_feedback_start.format(state.get('statements'),state.get('feedback')))   
 
    state['all_feedback_resolved'] = rating.feedbackResolved
    messages = state.get('messages')
    ticks = time.time()
    message_id = str(iterations) + "-inspector-" + str(ticks)
    messages.append(AIMessage(id=message_id, name= "Inspector (gpt-4o - v0.1)", content="Feedback resolved: " + str(rating.feedbackResolved) + " \n\n Reasoning: "+rating.reasoning))
    state['messages'] = messages
    st.info('This is iteration ' + str(iterations + 1), icon="ℹ️")
    print("Inspector done " + message_id)
    return state

# Define the nodes we will cycle between different states
workflow.add_node("handle_reviewer",handle_reviewer)
workflow.add_node("handle_analyst",handle_analyst)
workflow.add_node("handle_result",handle_result)
workflow.add_node("classify_feedback",classify_feedback)

# Define the conditional edges to decide if we should continue to the next state
def report_ready(state):
    report_ready = state['all_feedback_resolved']
    print("Deployment ready: " + str(report_ready))
    total_iterations = 1 if state.get('iterations')>5 else 0
    # print(state)
    if state.get('iterations')>loops:
        print("Iterations exceeded")
        return "handle_result"
    return "handle_result" if  report_ready or total_iterations else "handle_analyst" 

# Determine if the report is ready or not
workflow.add_conditional_edges(
    "classify_feedback",
    report_ready,
    {
        "handle_result": "handle_result",
        "handle_analyst": "handle_analyst"
    }
)

# Define the entry point and the end point
workflow.set_entry_point("handle_analyst")
workflow.add_edge('handle_analyst', "handle_reviewer")
workflow.add_edge('handle_reviewer', "classify_feedback")
workflow.add_edge('handle_result', END)

# Compile

app = workflow.compile()

st.title("💬 Agentic Finanical Copilot")
# st.caption("🚀 An agentic financial copilot that can research, generate, validate and iterate on financial statements")
st.html(
    "<p><span style=''>🚀 An agentic copilot that can research, generate, validate and iterate on financial statements. <a href='https://github.com/denniszielke/ai-financial-report-agents'>GitHub</a></span></p>"
)
st.write("<br>", unsafe_allow_html=True)
st.image("diagram.svg", caption="Process description of the financial analyst agents", use_column_width=True)
st.write("<br>", unsafe_allow_html=True)
human_query = st.chat_input("Type your analyst input prompt here...", key="human_query")

messages = {}

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))
    with st.spinner('Researcher preparing briefing package...'):
        inputs = prepare_flow(human_query)
    st.success("Briefing package created!")

    config = {"recursion_limit":20}

    with st.chat_message("Human"):
        st.write(human_query)

    with tracer.start_as_current_span("agent-chain") as span:
        for event in app.stream(inputs, config):   
            print ("message TO streamlit: ")
            for value in event.values():
                if ( value["messages"].__len__() > 0 ):
                    for message in value["messages"]:
                        if (message.content.__len__() > 0):
                            
                            if (message.id in messages):
                                continue

                            messages[message.id] = True

                            if ( isinstance(message, SystemMessage) ):
                                with st.chat_message("human"):
                                    st.write(message.content)
                                    st.session_state.previous_analysis = message.content
                            else:
                                if ( isinstance(message, AIMessage) ):
                                    with st.expander(message.name, expanded=False):
                                        with st.chat_message("AI"):
                                            st.write(message.content)
                                else:
                                    with st.chat_message("Agent"):
                                        st.write(message.content)
        # span.set_attribute("gen_ai.response.completion_token",callback.completion_tokens) 
        # span.set_attribute("gen_ai.response.prompt_tokens", callback.prompt_tokens) 
        # span.set_attribute("gen_ai.response.total_tokens", callback.total_tokens)