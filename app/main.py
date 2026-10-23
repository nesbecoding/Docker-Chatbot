from fastapi import FastAPI
from rag import generate_answer

app=FastAPI()
sessions={}

@app.get("/return_answer")
def return_answer(question : str,session_id : str):
    if session_id in sessions : m =sessions[session_id]
    else : m=[]
    return {"answer": generate_answer(question,m)}
