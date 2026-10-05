import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Data folder
DATA_FOLDER = "data"

documents = []


# 2. Read all .txt files
for root, dirs, files in os.walk(DATA_FOLDER):

    for file in files:

        if file.endswith(".txt"):

            file_path = os.path.join(root, file)

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            documents.extend(loader.load())


print("Number of documents:", len(documents))


# 3. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 4. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


print("Data successfully stored in Chroma DB.")