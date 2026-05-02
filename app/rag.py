from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from app.config import PINECONE_API_KEY, GROQ_API_KEY, INDEX_NAME


# 🔹 Initialize once (IMPORTANT) 
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are an expert in Ayurveda with deep knowledge of traditional principles, herbs, treatments, and lifestyle practices.

Instructions:
1. First, try to answer the question strictly using the provided context.
2. If the answer is clearly available in the context, base your response only on it.
3. If the context is incomplete or does not contain the answer:
   - Use your own Ayurvedic knowledge to provide a helpful answer.
   - Clearly state: "This answer is based on general Ayurvedic knowledge, not the provided context."
4. If you are unsure or the question is outside Ayurveda, say: "ask question based on Ayurveda only."

Guidelines:
- Keep answers accurate, concise, and practical.
- Avoid making unsafe medical claims or giving harmful advice.
- When relevant, mention Ayurvedic concepts (doshas, prakriti, etc.).

Context:s
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# 🔹 RAG Function
def ask_ayurveda(question: str):

    # Retrieve
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # LLM 
    response = llm.invoke(final_prompt)

    return {
        "answer": response.content,
        "sources": [
            {
                "chapter": doc.metadata.get("chapter_title"),
                "verse": doc.metadata.get("verse_range"),
                "topic": doc.metadata.get("topic")
            }
            for doc in docs
        ]
    }