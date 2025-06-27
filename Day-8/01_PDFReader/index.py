from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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


query = "What is React?"
retrieved_docs = retriever.invoke(query)


retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])


prompt = f"Based on the following text, answer the question : {query} \n \n {retrieved_text}"


result = model.invoke(prompt)


print(result.content)