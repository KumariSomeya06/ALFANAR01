# backend/main.py
import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
import uvicorn
from app import ocr_app as orchestrator_app

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Blob connection
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "quotationdocuments"
FOLDER_NAME = "documents"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# -----------------------------
# Session memory
# -----------------------------
session_memory = {}  # { session_id: last OrchestratorState }

@app.post("/process/")
async def process_file(
    files: Optional[List[UploadFile]] = File(None),
    query: Optional[str] = Form(None),
    session_id: Optional[str] = Form("default")
):
    try:
        extracted_text = None

        # Case 1: User uploaded new files → upload to blob storage
        if files:
            for file in files:
                blob_client = blob_service_client.get_blob_client(
                    container=CONTAINER_NAME, blob=f"{FOLDER_NAME}/{file.filename}"
                )
                contents = await file.read()
                blob_client.upload_blob(contents, overwrite=True)

        # Case 2: No new file → use memory
        else:
            prev_state = session_memory.get(session_id)
            if prev_state:
                extracted_text = prev_state.get("raw_text")

        # Build orchestrator input
        orchestrator_input = {
            "user_input": query or "",
            "container": CONTAINER_NAME,
            "intents": [],
            "routes": [],
            "result": session_memory.get(session_id, {}).get("result", {}),
            "raw_text": extracted_text or "",
            "final_response": "",
            "task_breakdown": ""
        }

        # Run orchestrator
        orchestrator_result = orchestrator_app.invoke(orchestrator_input)

        # Save session memory
        session_memory[session_id] = {
            **orchestrator_result,
            "raw_text": orchestrator_result.get("raw_text", extracted_text)
        }

        return {
            "message": "Processed successfully",
            "query": query,
            "session_id": session_id,
            "result": orchestrator_result
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
