from dotenv import load_dotenv
# from langchain_mistralai import MistralAIembeddings 
from langchain_community import vectorstores
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader
load_dotenv()
data=TextLoader("notes.txt")

doc=data.load()
print(doc)



