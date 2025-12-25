"""
RAG Document QA System using LangChain
A production-ready document question-answering system with vector embeddings
"""

import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st


class RAGDocumentQA:
    """RAG-based Document QA System"""
    
    def __init__(self, model_name: str = "llama2", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
    def load_documents(self, directory_path: str, file_types: List[str] = ['.pdf', '.txt']) -> List:
        """Load documents from directory"""
        documents = []
        
        for file_type in file_types:
            if file_type == '.pdf':
                loader = DirectoryLoader(
                    directory_path,
                    glob=f"**/*{file_type}",
                    loader_cls=PyPDFLoader
                )
            elif file_type == '.txt':
                loader = DirectoryLoader(
                    directory_path,
                    glob=f"**/*{file_type}",
                    loader_cls=TextLoader
                )
            
            documents.extend(loader.load())
        
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, chunks: List, persist_directory: str = "./vectorstore"):
        """Create and persist vector store"""
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save vectorstore
        self.vectorstore.save_local(persist_directory)
        return self.vectorstore
    
    def load_vectorstore(self, persist_directory: str = "./vectorstore"):
        """Load existing vectorstore"""
        self.vectorstore = FAISS.load_local(
            persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore
    
    def setup_qa_chain(self, temperature: float = 0.7, top_k: int = 4):
        """Setup QA chain with custom prompt"""
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Initialize LLM
        llm = Ollama(
            model=self.model_name,
            temperature=temperature,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain
    
    def ask_question(self, question: str):
        """Ask a question and get answer with sources"""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Run setup_qa_chain() first.")
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }


def main():
    """Streamlit UI for RAG QA System"""
    st.set_page_config(page_title="RAG Document QA", page_icon="📚", layout="wide")
    
    st.title("📚 RAG Document Question Answering System")
    st.markdown("Upload documents and ask questions using Retrieval-Augmented Generation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_name = st.selectbox(
            "Select LLM Model",
            ["llama2", "mistral", "codellama", "phi"]
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        top_k = st.slider("Top K Documents", 1, 10, 4)
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        
        st.markdown("---")
        st.markdown("### 📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        process_docs = st.button("Process Documents", type="primary")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGDocumentQA(model_name=model_name)
    
    if 'vectorstore_ready' not in st.session_state:
        st.session_state.vectorstore_ready = False
    
    # Process documents
    if process_docs and uploaded_files:
        with st.spinner("Processing documents..."):
            # Save uploaded files
            docs_dir = "./uploaded_docs"
            os.makedirs(docs_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(docs_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Load and process documents
            documents = st.session_state.rag_system.load_documents(docs_dir)
            st.info(f"Loaded {len(documents)} documents")
            
            chunks = st.session_state.rag_system.split_documents(documents, chunk_size=chunk_size)
            st.info(f"Created {len(chunks)} chunks")
            
            # Create vectorstore
            st.session_state.rag_system.create_vectorstore(chunks)
            st.info("Vector store created successfully")
            
            # Setup QA chain
            st.session_state.rag_system.setup_qa_chain(temperature=temperature, top_k=top_k)
            st.session_state.vectorstore_ready = True
            
            st.success("✅ Documents processed successfully! Ready to answer questions.")
    
    # Question answering interface
    st.markdown("---")
    st.header("💬 Ask Questions")
    
    if st.session_state.vectorstore_ready:
        question = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.rag_system.ask_question(question)
                    
                    # Display answer
                    st.markdown("### 🎯 Answer")
                    st.write(result["answer"])
                    
                    # Display sources
                    st.markdown("### 📄 Sources")
                    for idx, doc in enumerate(result["source_documents"], 1):
                        with st.expander(f"Source {idx}"):
                            st.write(doc.page_content)
                            st.caption(f"Metadata: {doc.metadata}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Please upload and process documents first using the sidebar.")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What are the main topics covered in the document?
        - Can you summarize the key findings?
        - What methodology was used?
        - What are the conclusions?
        """)


if __name__ == "__main__":
    main()
