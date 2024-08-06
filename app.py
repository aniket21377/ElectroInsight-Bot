from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
import pinecone


class ChatBot:
    load_dotenv()

    # Initialize variables
    docs_folder = "C:/Users/HP/Desktop/ChatBot/docs"
    chunk_size = 100000
    chunk_overlap = 4
    api_key = "6a34859c-5085-406c-9973-55d6c8c22ad9"
    index_name = "langchain-demo"
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    hf_api_token = "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"

    # Load and process documents
    documents = []
    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_folder, filename), encoding='utf-8')
            documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    pinecone.init(api_key=api_key, environment="gcp-starter")

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=768)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50,"max_new_tokens": 10000},
        huggingfacehub_api_token=hf_api_token,
    )

    from langchain.prompts import PromptTemplate

    template = """
    You are a Product Recommender which finds products made by companies like STMicroelectronics and other similar companies. Additionally, you provide information of products along with availability of these products in stores along with websites or places on the internet where the products are being sold. Your role is to suggest products, their specifications, and indicate if they are Not Recommended for New Design (NRND). You should also suggest multiple products if available.

    Below is a question from a user who wants to understand more about product recommendations based on the analysis of products available. Your role is to provide a factual, unbiased answer based on the provided data. If the information is not available in the data, it's okay to say you don't know. Keep your answers precise and do not answer questions not related to product recommendations.

    Context: {context}
    Query: {question}
    Answer:
    """


    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

