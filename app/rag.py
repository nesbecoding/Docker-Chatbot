from pathlib import Path
import os

data_dir=Path(os.environ.get("data_dir","../raw data"))
content=[]
for path in data_dir.rglob("*.md"):
    text=path.read_text(encoding="utf-8")
    content.append(text)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

data=[]

for c in content:
    for piece in splitter.split_text(c):
        data.append(piece)


import ollama
import numpy as np

def embed(texts,batch_size=32):
    vectors=[]
    for i in range(0,len(texts),batch_size):
        response=ollama.embed(
            model="nomic-embed-text",
            input=texts[i:i+batch_size],
        )
        vectors.extend(response["embeddings"])

    return np.array(vectors,dtype="float32")

embeddings=embed(data)

from faiss import normalize_L2,IndexFlatIP

normalize_L2(embeddings)
index=IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


def retrieve(query,k):
    embedded_query=embed([query])
    normalize_L2(embedded_query)
    D, I =index.search(embedded_query,k)
    return [data[i] for i in I[0]]


def generate(prompt):

    response=ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
        options={"temperature":0}
    )

    return response['message']['content']


def generate_answer(query,memory):

    context="\n".join(retrieve(query,3))

    history=""
    for q,a in memory[-3:]:
        history+=f"User : {q} \nAssistant : {a}\n"

    prompt=f"""
    You are a helpful assistant.

    Rules : 
    -when the user asks something that doesn't exist in the data you have, say exactly "I don't have enough data".
    - If the context contains no relevant information, say exactly "I don't have enough data".
    - If you don't have an answer, say you don't have enough data.
    - Be concise and friendly.
    - Answer in one sentence.

    Conversation history:
    {history}

    Context from knowledge base:
    {context}

    Question :
    {query}
    """

    answer= generate(prompt)
    memory.append((query,answer))
    return answer



