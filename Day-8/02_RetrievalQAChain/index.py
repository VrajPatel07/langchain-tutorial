from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


loader = TextLoader("docs.txt")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever()


system_prompt = """
    Use the given context to answer the question. If you don't know the answer, say you don't know.
    Context : {context}
"""


prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", "{input}")
])


question_answer_chain = create_stuff_documents_chain(llm = model, prompt = prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)


query = "What is React?"


result = chain.invoke({"input" : query})


print(result["answer"])