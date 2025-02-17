import os

from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from agents.agent_manager import AgentManager


class UserRequest(BaseModel):
    session_id: str
    user_id: str
    user_message: str


app = FastAPI(title="Multi-Agent System API")
agent_manager = AgentManager()


@app.get("/")
async def read_root():
    return {"hello": "world"}


@app.post("/chat")
async def chat(user_request: UserRequest):
    session_id = user_request.session_id
    user_id = user_request.user_id
    user_message = user_request.user_message
    ai_message = agent_manager.dispatch(session_id=session_id,
                                        user_id=user_id,
                                        user_message=user_message)
    return {'response': ai_message}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
