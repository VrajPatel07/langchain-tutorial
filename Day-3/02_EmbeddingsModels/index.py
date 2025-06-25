from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

docs = [
    "Good morning",
    "How are you?"
]

# vector = embeddings.embed_query("Hello World")

vector = embeddings.embed_documents(docs)

print(len(vector))