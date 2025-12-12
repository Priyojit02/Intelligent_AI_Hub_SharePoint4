# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime

from config import get_settings
from models import *
from state import app_state
from parallel import ParallelProcessor
from services.sharepoint import SharePointClient
from services.file_extractor import FileExtractor
from services.vector_store import VectorStoreManager
from services.qa_engine import QAEngineBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Initialize services
sharepoint_client = SharePointClient(
    tenant_id=settings.TENANT_ID,
    client_id=settings.CLIENT_ID,
    client_secret=settings.CLIENT_SECRET
)

file_extractor = FileExtractor()

vector_store_manager = VectorStoreManager(
    persist_dir=settings.PERSIST_DIR,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_API_BASE,
    embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)

qa_engine_builder = QAEngineBuilder(
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_API_BASE,
    model_name=settings.OPENAI_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    retriever_k=settings.RETRIEVER_K
)

parallel_processor = ParallelProcessor(
    max_workers=settings.MAX_WORKERS,
    batch_size=settings.BATCH_SIZE
)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Intelligent Assistant API")
    yield
    # Shutdown
    logger.info("Shutting down Intelligent Assistant API")
    app_state.clear_all()


# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Assistant API",
    description="Backend API for document Q&A with SharePoint integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js development server
        "http://localhost:3001",  # Alternative dev port
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://127.0.0.1:3001",  # Alternative localhost
        "*",  # Allow all for development - REMOVE IN PRODUCTION
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ENDPOINTS ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(status="healthy")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(status="healthy")


# ==================== ENDPOINT 1: List all hubs ====================

@app.get("/hubs", response_model=HubListResponse)
async def list_hubs():
    """
    List all available hubs with their metadata
    """
    try:
        hub_names = vector_store_manager.list_hubs()
        hubs = []
        
        for hub_name in hub_names:
            metadata = vector_store_manager.load_metadata(hub_name) or {}
            manifest = vector_store_manager.load_manifest(hub_name) or {}
            
            hubs.append(HubInfo(
                hub_name=hub_name,
                status=HubStatus.READY,
                file_count=manifest.get("count", 0),
                created_at=metadata.get("created_at"),
                last_synced=metadata.get("last_synced"),
                sharepoint_linked=bool(metadata.get("sharepoint_link")),
                auto_sync_enabled=metadata.get("auto_sync", False)
            ))
        
        return HubListResponse(hubs=hubs, total=len(hubs))
    
    except Exception as e:
        logger.error(f"Error listing hubs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENDPOINT 2: Get hub details ====================

@app.get("/hubs/{hub_name}", response_model=HubDetailResponse)
async def get_hub_details(hub_name: str):
    """
    Get detailed information about a specific hub including file manifest
    """
    try:
        # Check if hub exists
        if hub_name not in vector_store_manager.list_hubs():
            raise HTTPException(status_code=404, detail=f"Hub '{hub_name}' not found")
        
        manifest = vector_store_manager.load_manifest(hub_name) or {}
        metadata = vector_store_manager.load_metadata(hub_name) or {}
        
        files = [
            FileMetadata(**f) for f in manifest.get("files", [])
        ]
        
        return HubDetailResponse(
            hub_name=hub_name,
            status=HubStatus.READY,
            files=files,
            manifest=manifest,
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hub details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENDPOINT 3: Ingest from SharePoint (with parallel processing) ====================

@app.post("/hubs/from-sharepoint", response_model=IngestResponse)
async def ingest_from_sharepoint(
    request: SharePointIngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest documents from SharePoint folder/file link with parallel processing
    Optimized for handling 100+ files efficiently
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting SharePoint ingestion for hub: {request.hub_name}")
        
        # 1. Get SharePoint item metadata
        item_json = sharepoint_client.share_link_to_drive_item(request.sharepoint_link)
        
        # 2. Collect all files recursively
        logger.info("Collecting files from SharePoint...")
        files = sharepoint_client.collect_files_recursively(item_json)
        
        if not files:
            raise HTTPException(status_code=400, detail="No files found in SharePoint link")
        
        logger.info(f"Found {len(files)} files to process")
        
        # 3. Process files in parallel
        logger.info("Processing files in parallel...")
        all_text = parallel_processor.process_files_parallel(
            files=files,
            download_func=sharepoint_client.download_file,
            extract_func=file_extractor.extract
        )
        
        if not all_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from files")
        
        # 4. Create vector store
        logger.info("Creating vector store...")
        vectorstore = vector_store_manager.create_vectorstore(all_text, request.hub_name)
        
        # 5. Save manifest and metadata
        vector_store_manager.save_manifest(request.hub_name, files)
        vector_store_manager.save_metadata(request.hub_name, {
            "sharepoint_link": request.sharepoint_link,
            "auto_sync": request.auto_sync,
            "created_at": datetime.utcnow().isoformat(),
            "last_synced": datetime.utcnow().isoformat()
        })
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully ingested {len(files)} files in {processing_time:.2f}s")
        
        return IngestResponse(
            hub_name=request.hub_name,
            status="success",
            message=f"Successfully processed {len(files)} files from SharePoint",
            files_processed=len(files),
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SharePoint ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENDPOINT 4: Ingest from uploaded files ====================

@app.post("/hubs/from-upload", response_model=IngestResponse)
async def ingest_from_upload(
    hub_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Ingest documents from uploaded files
    Supports multiple file uploads with parallel processing
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting file upload ingestion for hub: {hub_name}")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Process files in parallel
        all_text = []
        file_metadata = []
        
        async def process_uploaded_file(file: UploadFile):
            content = await file.read()
            text, success = file_extractor.extract(content, file.filename)
            if success:
                all_text.append(text)
                file_metadata.append({
                    "id": file.filename,
                    "name": file.filename,
                    "etag": str(hash(content))[:32],
                    "size": len(content),
                    "lastModifiedDateTime": datetime.utcnow().isoformat()
                })
        
        # Process all files
        import asyncio
        await asyncio.gather(*[process_uploaded_file(f) for f in files])
        
        combined_text = "\n\n".join(all_text)
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from uploaded files")
        
        # Create vector store
        logger.info("Creating vector store...")
        vectorstore = vector_store_manager.create_vectorstore(combined_text, hub_name)
        
        # Save manifest and metadata
        vector_store_manager.save_manifest(hub_name, file_metadata)
        vector_store_manager.save_metadata(hub_name, {
            "created_at": datetime.utcnow().isoformat(),
            "source": "upload"
        })
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully ingested {len(files)} uploaded files in {processing_time:.2f}s")
        
        return IngestResponse(
            hub_name=hub_name,
            status="success",
            message=f"Successfully processed {len(files)} uploaded files",
            files_processed=len(files),
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENDPOINT 5: Load hub into memory ====================

@app.post("/hubs/{hub_name}/load")
async def load_hub(hub_name: str):
    """
    Load a hub's vector store into memory for querying
    If auto_sync is enabled, checks SharePoint for updates first
    """
    try:
        # Check if already loaded
        if app_state.get_hub(hub_name):
            return {
                "status": "success",
                "message": f"Hub '{hub_name}' is already loaded",
                "hub_name": hub_name,
                "synced": False
            }
        
        # Check if hub exists
        if hub_name not in vector_store_manager.list_hubs():
            raise HTTPException(status_code=404, detail=f"Hub '{hub_name}' not found")
        
        # Load metadata to check auto_sync
        metadata = vector_store_manager.load_metadata(hub_name)
        synced = False
        
        # AUTO-SYNC CHECK: If auto_sync enabled, check SharePoint first
        if metadata and metadata.get("auto_sync") and metadata.get("sharepoint_link"):
            logger.info(f"🔍 Auto-sync enabled, checking SharePoint for updates: {hub_name}")
            try:
                sharepoint_link = metadata["sharepoint_link"]
                
                # Get current files from SharePoint
                item_json = sharepoint_client.share_link_to_drive_item(sharepoint_link)
                current_files = sharepoint_client.collect_files_recursively(item_json)
                
                # Compare with existing manifest
                old_manifest = vector_store_manager.load_manifest(hub_name) or {}
                old_map = old_manifest.get("map", {})
                new_map = {f["id"]: f["etag"] for f in current_files}
                
                # If changes detected, sync
                if old_map != new_map:
                    logger.info(f"📝 Changes detected in SharePoint, syncing {hub_name}...")
                    
                    # Calculate changes
                    added = len(set(new_map.keys()) - set(old_map.keys()))
                    removed = len(set(old_map.keys()) - set(new_map.keys()))
                    modified = len({k for k in old_map.keys() & new_map.keys() if old_map[k] != new_map[k]})
                    
                    logger.info(f"  📊 Changes: +{added} files, ~{modified} modified, -{removed} removed")
                    
                    # Re-process files
                    all_text = parallel_processor.process_files_parallel(
                        files=current_files,
                        download_func=sharepoint_client.download_file,
                        extract_func=file_extractor.extract
                    )
                    
                    if all_text.strip():
                        # Rebuild vector store
                        vectorstore = vector_store_manager.create_vectorstore(all_text, hub_name)
                        
                        # Update manifest and metadata
                        vector_store_manager.save_manifest(hub_name, current_files)
                        metadata["last_synced"] = datetime.utcnow().isoformat()
                        metadata["sync_count"] = metadata.get("sync_count", 0) + 1
                        vector_store_manager.save_metadata(hub_name, metadata)
                        
                        synced = True
                        logger.info(f"✅ Synced {hub_name} with SharePoint")
                    else:
                        logger.warning(f"⚠️ No text extracted, using existing vector store")
                else:
                    logger.info(f"✓ No changes detected in SharePoint for {hub_name}")
            
            except Exception as sync_error:
                logger.error(f"❌ Sync error (will load existing data): {sync_error}")
                # Continue to load existing data even if sync fails
        
        # Load vector store (either updated or existing)
        logger.info(f"Loading hub into memory: {hub_name}")
        vectorstore = vector_store_manager.load_vectorstore(hub_name)
        
        if not vectorstore:
            raise HTTPException(status_code=404, detail=f"Hub '{hub_name}' vector store not found")
        
        # Build QA chain
        qa_chain = qa_engine_builder.build_qa_chain(vectorstore)
        
        # Store in application state
        app_state.set_hub(hub_name, qa_chain, vectorstore)
        
        logger.info(f"✅ Successfully loaded hub: {hub_name}")
        
        message = f"Hub '{hub_name}' loaded successfully"
        if synced:
            message += " (synced with SharePoint)"
        
        return {
            "status": "success",
            "message": message,
            "hub_name": hub_name,
            "synced": synced,
            "last_synced": metadata.get("last_synced") if metadata else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading hub: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENDPOINT 6: Chat/Query documents ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Query documents in a loaded hub
    """
    try:
        # Check if hub is loaded
        hub_data = app_state.get_hub(request.hub_name)
        
        if not hub_data:
            # Try to load it automatically
            logger.info(f"Hub '{request.hub_name}' not loaded, loading now...")
            vectorstore = vector_store_manager.load_vectorstore(request.hub_name)
            
            if not vectorstore:
                raise HTTPException(
                    status_code=404,
                    detail=f"Hub '{request.hub_name}' not found. Please load or create it first."
                )
            
            qa_chain = qa_engine_builder.build_qa_chain(vectorstore)
            app_state.set_hub(request.hub_name, qa_chain, vectorstore)
            hub_data = app_state.get_hub(request.hub_name)
        
        qa_chain = hub_data["qa"]
        
        # Execute query
        logger.info(f"Processing query for hub '{request.hub_name}': {request.query[:50]}...")
        result = qa_chain({"query": request.query})
        
        # Format response
        sources = None
        if request.include_sources and result.get("source_documents"):
            sources = [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        
        return ChatResponse(
            query=request.query,
            answer=result.get("result", ""),
            sources=sources,
            hub_name=request.hub_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENDPOINT 7: Sync with SharePoint ====================

@app.post("/hubs/{hub_name}/sync", response_model=SyncResponse)
async def sync_hub(hub_name: str, force: bool = False):
    """
    Sync hub with SharePoint to check for updates
    Re-processes files if changes detected
    """
    try:
        # Load metadata
        metadata = vector_store_manager.load_metadata(hub_name)
        if not metadata or not metadata.get("sharepoint_link"):
            raise HTTPException(
                status_code=400,
                detail=f"Hub '{hub_name}' is not linked to SharePoint"
            )
        
        sharepoint_link = metadata["sharepoint_link"]
        
        # Get current files from SharePoint
        logger.info(f"Checking SharePoint for updates: {hub_name}")
        item_json = sharepoint_client.share_link_to_drive_item(sharepoint_link)
        current_files = sharepoint_client.collect_files_recursively(item_json)
        
        # Load existing manifest
        old_manifest = vector_store_manager.load_manifest(hub_name) or {}
        old_map = old_manifest.get("map", {})
        
        # Build new manifest
        new_map = {f["id"]: f["etag"] for f in current_files}
        
        # Detect changes
        changes_detected = (old_map != new_map) or force
        
        if not changes_detected:
            return SyncResponse(
                hub_name=hub_name,
                status="success",
                changes_detected=False,
                files_updated=0,
                message="No changes detected in SharePoint"
            )
        
        # Re-process files
        logger.info(f"Changes detected, re-processing files for hub: {hub_name}")
        all_text = parallel_processor.process_files_parallel(
            files=current_files,
            download_func=sharepoint_client.download_file,
            extract_func=file_extractor.extract
        )
        
        # Rebuild vector store
        vectorstore = vector_store_manager.create_vectorstore(all_text, hub_name)
        
        # Update manifest and metadata
        vector_store_manager.save_manifest(hub_name, current_files)
        metadata["last_synced"] = datetime.utcnow().isoformat()
        vector_store_manager.save_metadata(hub_name, metadata)
        
        # Update in-memory hub if loaded
        if app_state.get_hub(hub_name):
            qa_chain = qa_engine_builder.build_qa_chain(vectorstore)
            app_state.set_hub(hub_name, qa_chain, vectorstore)
        
        return SyncResponse(
            hub_name=hub_name,
            status="success",
            changes_detected=True,
            files_updated=len(current_files),
            message=f"Successfully synced {len(current_files)} files from SharePoint"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Additional utility endpoints ====================

@app.delete("/hubs/{hub_name}")
async def delete_hub(hub_name: str):
    """Delete a hub and all its data"""
    try:
        if hub_name not in vector_store_manager.list_hubs():
            raise HTTPException(status_code=404, detail=f"Hub '{hub_name}' not found")
        
        # Remove from memory
        app_state.remove_hub(hub_name)
        
        # Delete from disk
        vector_store_manager.delete_hub(hub_name)
        
        return {
            "status": "success",
            "message": f"Hub '{hub_name}' deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hubs/{hub_name}/unload")
async def unload_hub(hub_name: str):
    """Unload a hub from memory"""
    try:
        if not app_state.get_hub(hub_name):
            return {
                "status": "success",
                "message": f"Hub '{hub_name}' was not loaded"
            }
        
        app_state.remove_hub(hub_name)
        
        return {
            "status": "success",
            "message": f"Hub '{hub_name}' unloaded from memory"
        }
    
    except Exception as e:
        logger.error(f"Unload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hubs/loaded/list")
async def list_loaded_hubs():
    """List all currently loaded hubs"""
    return {
        "loaded_hubs": app_state.list_loaded_hubs(),
        "count": len(app_state.list_loaded_hubs())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
