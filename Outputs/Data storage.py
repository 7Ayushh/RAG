from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json

client = MongoClient('mongodb://localhost:27017/')
db = client['ChatBot_db']
collection = db['document_chunks']

with open(r'C:\Users\divya\OneDrive\Documents\Vector Database DAV project\chunks.json', 'r') as file:
    chunks = json.load(file)

model = SentenceTransformer('all-MiniLM-L6-v2')

def vectorize_chunks(chunks):
    texts = [chunk['content'] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings

embeddings = vectorize_chunks(chunks)

embeddings_list = embeddings.tolist()

chunk_documents = []
for i, chunk in enumerate(chunks):
    chunk_document = {
        "chunk_id": f"chunk_{i+1}",
        "text": chunk['content'],
        "metadata": chunk['metadata'],
        "vector": embeddings_list[i]
    }
    chunk_documents.append(chunk_document)

collection.insert_many(chunk_documents)

print(f"Inserted {len(chunk_documents)} chunks into MongoDB") 