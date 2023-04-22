import sys
import os

import uvicorn

from fastapi import Depends, FastAPI
from fastapi import responses, status
from pathlib import Path
from logbook import Logger
from fastapi.middleware.cors import CORSMiddleware


import views.LP_Detect_and_Rec_Main_Router as Plates_Router
from config import settings

BASEPATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASEPATH)

logger = Logger(__name__)
app = FastAPI()

# app.add_middleware(
#     CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
#     allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
#     allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
#     allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
#     allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
# )

app.include_router(Plates_Router.router)


@app.get("/")
def root_page():
    return {"message": "Welcome to Test API"}

if __name__ == "__main__":
    port = 8000
    # ngrok_tunnel = ngrok.connect(port)
    # print('Public URL:', ngrok_tunnel.public_url)
    # nest_asyncio.apply()
    uvicorn.run(app, port=port)
