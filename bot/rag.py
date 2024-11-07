import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_6vKoPN_29eWAQfJpnxi7X8ZMsPYhKmuzeGTknYZVzrJqTLoBNQWg16dZnamobd16E71FwS")
index_name = "factcheck-local"
pinecone_index = pc.Index(index_name)
print('Pinecone index initialized')

# Set up embeddings
model_name = "BAAI/bge-m3"
encode_kwargs = {'normalize_embeddings': True, "show_progress_bar": False}
query_instruction = "Represent this sentence for searching relevant passages: "
embeddings_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction=query_instruction,
)
print('Embedding model created')

# Create a vector store using Pinecone
vector_store = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings_function,
    text_key="text"
)
print('Vector store created')

# Set up the retriever
retriever = vector_store.as_retriever()
print('Retriever created')

# Set up the language model
llm = ChatOpenAI(model_name="gpt-4o")

# Create the prompt template
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

async def generate_factcheck(query: str):
    return await chain.ainvoke({"input": query})
