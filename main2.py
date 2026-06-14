from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
data=PyPDFLoader("documents_loader/deeplearning.pdf")
doc=data.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(doc)

embedding_model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorestore=Chroma.from_documents(documents=chunks,
                                   persist_directory="chroma_db",
                                   embedding=embedding_model)
collection = vectorestore.get()

print("Number of chunks:", len(collection["documents"]))

for i in range(min(5, len(collection["documents"]))):
    print(f"\nChunk {i+1}:")
    print(collection["documents"][i][:500])

retriver=vectorestore.as_retriever(search_type="mmr",search_kwargs={"k":4,"fetch_k":10,"lambda_mult":0.5}) 

llm_model=ChatMistralAI(model="mistral-small-2603")

## Promt template=ChatPromptTemplate.from_messages([('system',"you are ai assistant to answer the question based on the context"),

template=ChatPromptTemplate.from_messages([('system',"""you are a helpful ai assistant and use only the provided context to answer the question.
                                            if answer not present in the context, say: "i could not find the answer in this document." """),
                                            ("human","context: {context} \n question: {question}")
                                           ])

print("RAG System is Created Successfully")

print("press 0 to exit")

while True:
    query=input("Enter your question: ")
    if query=="0":
        print("Exiting the RAG system. Goodbye!")
        break
    docs=retriver.invoke(query)
    # relevant_docs=retriver.get_relevant_documents(query)
    context="\n".join([doc.page_content for doc in docs])
    final_prompt=template.invoke({"context" :context,"question":query})

    
    result=llm_model.invoke(final_prompt)

    print(f"\n AI :{result.content}\n")