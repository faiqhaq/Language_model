import os
import time
import textract
import subprocess
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

# Constants
PDF_PATH = "" #Documents path
OLLAMA_MODEL = 'llama2-uncensored'
COLLECTION_NAME = "rag-chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path):
    return textract.process(pdf_path).decode('utf-8')

def start_ollama_service():
    command = "nohup ollama serve &"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Ollama service started. Process ID: {process.pid}")
    time.sleep(5)  # Wait for the service to start

def create_vectorstore(text, embeddings):
    document = Document(page_content=text)
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=20)
    splits = text_splitter.split_documents([document])
    
    return Chroma.from_documents(
        documents=splits,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

def setup_rag_chain(retriever):
    template = """Answer the question based mostly on the following context:
    {context}

    Question: {question} 
    """
    prompt = ChatPromptTemplate.from_template(template)
    model_local = ChatOllama(model=OLLAMA_MODEL)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model_local
        | StrOutputParser()
    )

def main():
    os.environ['OLLAMA_MODEL'] = OLLAMA_MODEL
    start_ollama_service()

    text = extract_text_from_pdf(PDF_PATH)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = create_vectorstore(text, embeddings)
    retriever = vectorstore.as_retriever()

    chain = setup_rag_chain(retriever)
    
    #Quesion 
    import time
    question=True
    while question:
        question=input('ask question:')
        if question=="exit":
            break
        answer = chain.invoke(question)
        print('~~~ Answer ~~~')
        print(f'Answer: {answer}')

if __name__ == "__main__":
    main()