import os

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from chat_agent import ChatAgent

app = FastAPI()
chat_agent = ChatAgent()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message
    response = chat_agent.process_message(
        session_id=session_id, user_message=message)
    return {"agent": response}

if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=12345)
