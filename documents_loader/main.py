from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

data_from_notes=TextLoader("documents_loader/notes.txt")
doc=data_from_notes.load()
data_from_pdf=PyPDFLoader("documents_loader/deeplearning.pdf")
doc_from_pdf=data_from_pdf.load()

templates=ChatPromptTemplate.from_messages([('system',"you are ai assistant to summarize the text"),
                                          ("human","{data}")])


llm_model=ChatMistralAI(model="mistral-small-2603")
prompt=templates.format_messages(data=doc_from_pdf)
result=llm_model.invoke(prompt)
print(result.content)



