from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_core.messages import SystemMessage



def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
Generate an SQL query for the 'all_products' table based on the user's question and calculate saving by old price - price.

Schema: {schema}
History: {chat_history}

Question: {question}

SQL Query:

Examples:
1. Question: Find Coca-Cola products when find any product use this command put search term into '' as coca-cola in the command
   SQL: SELECT brand, name, price, unit, 
       MATCH(name, english_name) AGAINST('Coca-Cola' IN NATURAL LANGUAGE MODE) AS relevance
        FROM all_products
        WHERE MATCH(name, english_name) AGAINST('coca-cola' IN NATURAL LANGUAGE MODE)
          AND available_now = TRUE
        ORDER BY relevance DESC
        LIMIT 5;


2. Question: List the cheapest discounted products available
   SQL: WITH discounted_products AS (SELECT brand, name, price, unit, (COALESCE(old_price, 0) - COALESCE(price, 0)) AS discount_amount FROM all_products WHERE COALESCE(old_price, 0) > COALESCE(price, 0) AND available_now = TRUE) SELECT brand, name, price, unit FROM discounted_products ORDER BY price ASC;

Your Turn:
Question: {question}

SQL:
    """
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(model="gpt-4")
  #llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  def get_schema(_):
    return db.get_table_info(table_names=['all_products'])
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(model="gpt-4")
  #llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    
  
def agent_sql(query,db: SQLDatabase, chat_history: list,SQL_PREFIX):

  llm = ChatOpenAI(temperature=0, verbose=True)
  agent_executor = create_sql_agent(
      llm=llm,
      toolkit=SQLDatabaseToolkit(db=db, llm=llm),
      verbose=True,
      agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  )
  response = agent_executor.invoke(
    {
        "input": f"{query}",
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": f"{chat_history}",
        "description": f"{SQL_PREFIX}",
    }
)
  return response 
  
  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm an assistant. Ask me anything about the products."),
    ]

load_dotenv()

st.set_page_config(page_title="BargainB Assistaint", page_icon=":speech_balloon:")

st.title("BargainB Assistant")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="34.71.156.167", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="", key="Password")
    st.text_input("Database", value="dutch_markets", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        # response = agent_sql(user_query, st.session_state.db, st.session_state.chat_history,SQL_PREFIX)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
    




