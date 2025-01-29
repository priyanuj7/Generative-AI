# This program is intended to use the following
# - Extracts the contents of a webpage, chunks and loads the chunks (documents) into FAISS db
# Creates a sample conversation, uses a "create_history_aware_retriever" retrieval chain
# - The last step is to pass the history and ask a follow up question


from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# loading the webpage
loader = WebBaseLoader("https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c")
docs = loader.load()

# The recursive character text splitter takes a large text and splits it based on a specified chunk size
# It does this by using a set of characters. The defauls characters provided to it are ["\n\m", "\n", " ", ""].
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

# Instantiating the LLM
llm = ChatOpenAI()

# Instantiating the Embedding model
embeddings = OpenAIEmbeddings()

vector = FAISS.from_documents(documents, embeddings)

# Creating the retriever object
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retreiver_chain = create_history_aware_retriever(llm, retriever, prompt)

sample_answer = """
A deep learning Reinforcement Learning is a type of Machine Learning where an agent learns how to behave in an environment, 
by performing actions and getting rewarded or punished based on the outcome of the action taken...
and so on...
Note: Answers can be in pointers as well
"""

chat_history = [HumanMessage(content = "What is Deep Reinforcement Learning?"),
    AIMessage(content = sample_answer)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer teh user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("user", "{input}"),
])

# This method is used to pull the data from the staff database
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retreiver_chain, document_chain)

output = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Can you explain on where it can be used?"
})

# note that here, we are using "it" to Refer to RL. We are not mentioning RL.

print(output["answer"])
