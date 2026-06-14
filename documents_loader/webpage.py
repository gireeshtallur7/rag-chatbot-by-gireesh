from langchain_community.document_loaders import WebBaseLoader
url=WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
doc=url.load()
print(len(doc))