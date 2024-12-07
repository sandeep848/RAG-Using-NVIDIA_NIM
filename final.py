import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class MultimodalAIApp:
    def __init__(self):
        load_dotenv()
        self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
        if not self.NVIDIA_API_KEY:
            st.error("NVIDIA API Key not found. Please set it in your .env file.")
        self.CHUNK_SIZE = 700
        self.CHUNK_OVERLAP = 50
        self.MAX_DOCUMENTS = 50

    def load_and_process_document(self, uploaded_file):
        try:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(uploaded_file.name)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(uploaded_file.name)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                return None
            documents = loader.load()
            embeddings = NVIDIAEmbeddings()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE, 
                chunk_overlap=self.CHUNK_OVERLAP
            )
            split_documents = text_splitter.split_documents(
                documents[:self.MAX_DOCUMENTS]
            )
            vector_store = FAISS.from_documents(split_documents, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return None
        finally:
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)

    def create_retrieval_chain(self):
        try:
            llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful AI assistant. Answer the questions based strictly on the provided context.
            If the answer is not in the context, say "I cannot find the answer in the provided documents."

            Context:
            {context}

            Question: {input}
            
            Helpful Answer:
            """)
            document_chain = create_stuff_documents_chain(llm, prompt)
            return document_chain
        except Exception as e:
            st.error(f"Error creating retrieval chain: {e}")
            return None

    def run(self):
        st.set_page_config(
            page_title="NVIDIA Document Q&A", 
            page_icon="📄"
        )
        st.title("🚀 Document Question Answering using NVIDIA_NIM")
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Upload a document (PDF or TXT)", 
            type=['pdf', 'txt']
        )
        if uploaded_file:
            with st.spinner("Processing document..."):
                vector_store = self.load_and_process_document(uploaded_file)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.success("Document processed successfully!")
                query = st.text_input("Enter your question about the document:")
                if query and st.button("Get Answer"):
                    try:
                        document_chain = self.create_retrieval_chain()
                        if document_chain:
                            retriever = st.session_state.vector_store.as_retriever(
                                search_kwargs={"k": 5}
                            )
                            retrieval_chain = create_retrieval_chain(retriever, document_chain)
                            start_time = time.time()
                            response = retrieval_chain.invoke({'input': query})
                            response_time = time.time() - start_time
                            st.subheader("Answer")
                            st.write(response['answer'])
                            st.caption(f"Response generated in {response_time:.2f} seconds")
                            with st.expander("Relevant Document Chunks"):
                                for i, doc in enumerate(response["context"], 1):
                                    st.text_area(
                                        f"Chunk {i}", 
                                        value=doc.page_content, 
                                        height=100
                                    )
                    except Exception as e:
                        st.error(f"Error processing question: {e}")

def main():
    app = MultimodalAIApp()
    app.run()

if __name__ == "__main__":
    main()
