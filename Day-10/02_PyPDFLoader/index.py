from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(f"Total number of pages : {len(docs)}")

for document in docs:
    print(document.metadata)