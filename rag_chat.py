import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


load_dotenv()   
openai_key = os.getenv("OPENAI_API_KEY")


def load_and_embed_pdfs(file_paths):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for idx, path in enumerate(file_paths):
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        doc_pages = loader.load()
        for page in doc_pages:
            page.metadata["source"] = f"resume_{idx+1}"  # label each resume separately
        docs.extend(doc_pages)

    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = FAISS.from_documents(split_docs, embedding=embeddings)
    vectordb.save_local("vectorstore_chat")
    return vectordb


def get_chat_chain(vectordb):
    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(
        temperature=0.3,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_key
    )

    template = """
You are a helpful HR assistant who compares two candidate resumes based on their content.
Use the following extracted document text to answer the question.
If you cannot answer from the context, say you don't know.

Context:
{context}

Chat history:
{chat_history}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
