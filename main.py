import os
import tempfile
import streamlit as st
import pickle
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.title("Article Search Engine")
st.sidebar.title("Article Links")

urls = []

for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_clicked = st.sidebar.button("Process")
file_path = "vector_store_openai.pkl"
openai_api_key = "OPENAI_API_KEY"
llm = OpenAI(temperature=0.85, max_tokens=500, openai_api_key=openai_api_key)

if process_clicked:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            loader = UnstructuredURLLoader(urls=urls, temp_dir=tmpdirname)
            data = loader.load()

            # Debug: Print raw loaded data
            st.write("Loaded data:", data)

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )

            docs = text_splitter.split_documents(data)

            # Debug: Print split documents
            st.write("Split documents:", docs)

            if not docs:
                st.error("No documents found. Please check your input data.")
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorStore = FAISS.from_documents(docs, embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorStore, f)

        except Exception as e:
            st.error(f"Error fetching or processing data: {e}")

question = st.text_input("Question:")

if question:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": question}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])
    else:
        st.error("Vector store not found. Please process the documents first.")
