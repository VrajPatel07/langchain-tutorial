from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)