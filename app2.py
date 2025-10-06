import os
from getpass import getpass
from dotenv import load_dotenv

# LangChain Core/Chains
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Integrations (Groq, Loaders, Embeddings, Vector Store)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# Ensure GROQ_API_KEY is set in your environment
os.environ["GROQ_API_KEY"] = "" 

# Ensure your .env file has the GROQ_API_KEY for embeddings
load_dotenv()

#one file should be uploaded atleast
# Define the paths for your documents
pdf_file_path = ""  #place the name of the pdf file if uploaded
docx_file_path = "" #place the name of the doc file if uploaded


# List to hold all documents loaded from any source
all_documents = []

# ---  Load PDF Document (if available) ---
if os.path.exists(pdf_file_path):
    print(f"Loading PDF file: {pdf_file_path}...")
    try:
        pdf_loader = PyPDFLoader(pdf_file_path)
        pdf_documents = pdf_loader.load()
        all_documents.extend(pdf_documents)
        print(f"Successfully loaded {len(pdf_documents)} pages from PDF.")
    except Exception as e:
        print(f"Error loading PDF file: {e}")
else:
    print(f"PDF file not found at: {pdf_file_path}. Skipping.")

# ---  Load DOCX Document (if available) ---
if os.path.exists(docx_file_path):
    print(f"Loading DOCX file: {docx_file_path}...")
    try:
        # Using UnstructuredWordDocumentLoader to handle DOCX/DOC and complex elements
        word_loader = UnstructuredWordDocumentLoader(docx_file_path, mode="elements")
        word_documents = word_loader.load()
        all_documents.extend(word_documents)
        print(f"Successfully loaded {len(word_documents)} elements from DOCX.")
    except ImportError:
        print("Unstructured dependencies not installed. Please run: pip install 'unstructured[docx]'")
    except Exception as e:
        print(f"Error loading DOCX file: {e}")
else:
    print(f"DOCX file not found at: {docx_file_path}. Skipping.")

# ---  Final Check ---
if all_documents:
    print(f"\nTotal documents loaded for RAG processing: {len(all_documents)}")
else:
    print("\nNo documents were successfully loaded. Please check file paths and permissions.")

# The 'all_documents' list now contains data from all available files.
# You can proceed with chunking and embedding this list.

#  Split the Document into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # Overlap helps maintain context between chunks
    separators=["\n\n", "\n", " ", ""]
)

texts = text_splitter.split_documents(all_documents)

print(f"Split document into {len(texts)} chunks.")

# Using 'BAAI/bge-small-en-v1.5' which is a common and good performing model
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 'faiss_pdf_index' is the directory name you mentioned in your initial output
FAISS_INDEX_PATH = "faiss_pdf_index"

# NOTE: allow_dangerous_deserialization is required if your embeddings
# Create the FAISS vector store from the documents and embeddings
vector_store = FAISS.from_documents(texts, embeddings)

vector_store.save_local(FAISS_INDEX_PATH)

print(f"FAISS index created and saved locally as '{FAISS_INDEX_PATH}'.")
print("You can now load this index for RAG using FAISS.load_local().")

# Re-initialize the SAME embedding model used to save the index
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load the saved FAISS index
print(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
try:
    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embeddings,
        # Required if your embedding model uses an arbitrary Python class
        allow_dangerous_deserialization=True 
    )
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAlectionAISS index: {e}")
    print("Make sure the directory exists and was created with the same embedding model.")
    exit()




#  Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", # Use llama3-8b for faster performance, or llama3-70b-8192 for better quality
    temperature=0
)

#  Create a Prompt Template
system_prompt = (
    "You are an expert Q&A assistant for the provided documents. "
    "Use the following retrieved context to answer the user's question. "
    "If you don't know the answer, just say that you don't know. "
    "\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

#  Create the Document Combination Chain (Stuffing)
# This chain takes the retrieved documents and the user's question (input) 
# and formats them into the prompt for the LLM.
document_chain = create_stuff_documents_chain(llm, prompt)

#  Convert the vector store into a Retriever
# A retriever is what performs the similarity search on the vector store.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

#  Create the final RAG Chain
# This chain orchestrates the RAG process: Retriever -> Document Chain
rag_chain = create_retrieval_chain(retriever, document_chain)

print("\n--- Starting RAG Query Interface ---")
print("Enter 'quit' or 'exit' to stop the program.\n")

while True:
    # Get user input
    question = input("Your question : ")

    # Check for exit command
    if question.lower() in ['quit', 'exit']:
        print("\nExiting RAG Query Interface. Goodbye!")
        break

    if not question.strip():
        print("Please enter a question.\n")
        continue

    print(f"\nQuerying: {question}")
        
    try:
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": question})

        # Print the answer
        print("\n--- Answer ---")
        print(response["answer"])
        print("-" * 20 + "\n")

    except Exception as e:
        print(f"\nAn error occurred during RAG chain invocation: {e}")
        print("-" * 20 + "\n")
