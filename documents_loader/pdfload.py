from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("documents_loader/deeplearning.pdf")
docs=loader.load()
print(docs[3].page_content)
print(len(docs))
