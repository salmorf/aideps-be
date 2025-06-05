import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    root_path="/api",
    title="Il Mio API",
    description="",
    version="1.0.0",
    contact={
        "name": "Kazaam Lab",
        "email": "",
    },
    openapi_url="/openapi.json",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)
from app.routes import ml, server, user

print("AVVIANDO SERVER...")
RAW_LABELS_DIR = Path("./app/raw_labels")
RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
app.include_router(server.router)
app.include_router(ml.router)
app.include_router(user.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://135.181.217.246:3000", "http://aideps.kazaamlab.com:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
