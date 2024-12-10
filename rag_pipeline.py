from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

model_path = "/Users/tobe/Downloads/llama-3.2-transformers-3b-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Import and create embedding model AFTER loading model/tokenizer
from local_llama_embedding import LocalLlamaEmbedding
embedding_model = LocalLlamaEmbedding(tokenizer=tokenizer, model=model, device=device)

# Now you can safely set Settings.embed_model
Settings.embed_model = embedding_model

# Create Pinecone vector store
vector_store = PineconeVectorStore(
    pinecone_index=pc.Index(host=os.environ["PINECONE_HOST"]),
    embedding=embedding_model
)

# Set up LLM
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.environ["OPENAI_API_KEY"]
)
Settings.llm = llm

# Create VectorIndex from vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Create query engine and query
query_engine = index.as_query_engine(embed_model=embedding_model, similarity_top_k=10, response_mode="tree_summarize")
question = "What hydrogen safety standards are recommended for Oklahoma facilities?"
response = query_engine.query(question)
print(response)

def answer_question(question: str) -> str:
    response = query_engine.query(question)
    return str(response)
