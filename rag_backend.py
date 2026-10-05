import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


class RAGBackend:

    def __init__(self):

        # Project directory
        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        # Chroma database location
        self.chroma_dir = os.path.join(
            self.base_dir,
            "chroma_db"
        )

        # Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Chroma vector database
        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings
        )

        # Local LLM
        self.llm = OllamaLLM(
            model="mistral:latest",
            temperature=0.2
        )


    def ask(self, question):

        # Search relevant documents
        documents = self.vectorstore.similarity_search(
            question,
            k=4
        )

        # No relevant documents
        if not documents:

            return (
                "I could not find this information "
                "in the available enterprise documents."
            )


        # Create context
        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )


        # RAG prompt
        prompt = f"""
You are an Enterprise RAG Assistant.

Answer the user's question using ONLY
the information provided in the context.

Do not invent information.

If the answer is not available in the context,
say:

"I could not find this information in the
available enterprise documents."

Keep the answer clear and professional.

================ CONTEXT ================

{context}

================ QUESTION ================

{question}

================ ANSWER ================
"""


        # Generate answer
        answer = self.llm.invoke(prompt)

        return answer