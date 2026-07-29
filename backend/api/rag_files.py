import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.auth import get_current_user
from core.database import get_db
from core.models import RagChunk, RagFile, User
from services.rag import (
    delete_rag_file_record,
    normalize_pdf_filename,
    process_rag_file,
    retrieve_rag_chunks,
    validate_public_http_url,
)
from core.rag_config import RAG_MAX_UPLOAD_MB


router = APIRouter(prefix="/api/files", tags=["files"])


class RagFileResponse(BaseModel):
    id: int
    filename: str
    source_type: str
    mime_type: str
    url: str | None
    final_url: str | None
    title: str | None
    site_name: str | None
    size_bytes: int
    status: str
    error: str | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class RagLinkCreate(BaseModel):
    url: HttpUrl


class RagSearchResult(BaseModel):
    chunk_id: int
    chunk_index: int | None
    file_id: int
    filename: str
    page_start: int | None
    page_end: int | None
    heading_path: str | None
    content: str
    score: float
    vector_similarity: float | None
    text_rank: float | None
    source_types: list[str]


class RagChunkResponse(BaseModel):
    id: int
    chunk_index: int
    page_start: int | None
    page_end: int | None
    heading_path: str | None
    content: str
    content_chars: int
    embedding_stored: bool
    embedding_dimension: int
    embedding_preview: list[float]
    search_indexed: bool
    created_at: datetime
    updated_at: datetime


class RagChunkPageResponse(BaseModel):
    file_id: int
    filename: str
    source_type: str
    status: str
    total: int
    offset: int
    limit: int
    items: list[RagChunkResponse]


def _file_response(rag_file: RagFile, chunk_count: int = 0) -> RagFileResponse:
    return RagFileResponse(
        id=rag_file.id,
        filename=rag_file.filename,
        source_type=rag_file.source_type or "pdf",
        mime_type=rag_file.mime_type,
        url=rag_file.url,
        final_url=rag_file.final_url,
        title=rag_file.title,
        site_name=rag_file.site_name,
        size_bytes=rag_file.size_bytes,
        status=rag_file.status,
        error=rag_file.error,
        chunk_count=chunk_count,
        created_at=rag_file.created_at,
        updated_at=rag_file.updated_at,
    )


def _is_pdf_upload(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    return filename.endswith(".pdf") and content_type in {
        "application/pdf",
        "application/octet-stream",
        "binary/octet-stream",
        "",
    }


def _chunk_response(chunk: RagChunk) -> RagChunkResponse:
    embedding = list(chunk.embedding) if chunk.embedding is not None else []
    return RagChunkResponse(
        id=chunk.id,
        chunk_index=chunk.chunk_index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        heading_path=chunk.heading_path,
        content=chunk.content,
        content_chars=len(chunk.content),
        embedding_stored=bool(embedding),
        embedding_dimension=len(embedding),
        embedding_preview=[float(value) for value in embedding[:8]],
        search_indexed=chunk.search_vector is not None,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
    )


@router.get("", response_model=List[RagFileResponse])
async def list_files(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(RagFile, func.count(RagChunk.id).label("chunk_count"))
        .outerjoin(RagChunk, RagChunk.file_id == RagFile.id)
        .where(RagFile.user_id == current_user.id)
        .group_by(RagFile.id)
        .order_by(RagFile.updated_at.desc(), RagFile.created_at.desc())
    )
    return [_file_response(rag_file, chunk_count) for rag_file, chunk_count in result.all()]


@router.post("", response_model=RagFileResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _is_pdf_upload(file):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    data = await file.read()
    max_bytes = RAG_MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"PDF must be {RAG_MAX_UPLOAD_MB}MB or smaller")

    rag_file = RagFile(
        user_id=current_user.id,
        filename=normalize_pdf_filename(file.filename or "document.pdf"),
        storage_path="",
        mime_type=file.content_type or "application/pdf",
        source_type="pdf",
        size_bytes=len(data),
        status="processing",
    )
    db.add(rag_file)
    await db.flush()

    from core.storage import storage_client
    object_name = f"{current_user.id}/{rag_file.id}.pdf"
    storage_path = await storage_client.upload_file(data, object_name)
    rag_file.storage_path = storage_path

    await db.commit()
    await db.refresh(rag_file)

    from core.task_queue import task_queue
    task_queue.enqueue(process_rag_file, rag_file.id)
    return _file_response(rag_file, 0)


@router.post("/link", response_model=RagFileResponse)
async def add_link(
    request: RagLinkCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        url = await validate_public_http_url(str(request.url))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rag_file = RagFile(
        user_id=current_user.id,
        filename=url,
        storage_path="",
        mime_type="text/markdown",
        source_type="link",
        url=url,
        final_url=url,
        site_name=url,
        size_bytes=0,
        status="processing",
    )
    db.add(rag_file)
    await db.commit()
    await db.refresh(rag_file)

    from core.task_queue import task_queue
    task_queue.enqueue(process_rag_file, rag_file.id)
    return _file_response(rag_file, 0)


@router.get("/{file_id}/chunks", response_model=RagChunkPageResponse)
async def list_file_chunks(
    file_id: int,
    offset: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if offset < 0:
        raise HTTPException(status_code=422, detail="Offset must be zero or greater")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 100")

    file_result = await db.execute(
        select(RagFile).where(RagFile.id == file_id, RagFile.user_id == current_user.id)
    )
    rag_file = file_result.scalars().first()
    if not rag_file:
        raise HTTPException(status_code=404, detail="File not found")

    total_result = await db.execute(
        select(func.count(RagChunk.id)).where(
            RagChunk.file_id == file_id,
            RagChunk.user_id == current_user.id,
        )
    )
    total = total_result.scalar_one()
    chunks_result = await db.execute(
        select(RagChunk)
        .where(RagChunk.file_id == file_id, RagChunk.user_id == current_user.id)
        .order_by(RagChunk.chunk_index)
        .offset(offset)
        .limit(limit)
    )
    return RagChunkPageResponse(
        file_id=rag_file.id,
        filename=rag_file.filename,
        source_type=rag_file.source_type or "pdf",
        status=rag_file.status,
        total=total,
        offset=offset,
        limit=limit,
        items=[_chunk_response(chunk) for chunk in chunks_result.scalars().all()],
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(RagFile).where(RagFile.id == file_id, RagFile.user_id == current_user.id)
    )
    rag_file = result.scalars().first()
    if not rag_file:
        raise HTTPException(status_code=404, detail="File not found")

    await delete_rag_file_record(rag_file, db)
    return {"message": "Deleted successfully"}


@router.post("/search", response_model=List[RagSearchResult])
async def search_files(
    request: RagSearchRequest,
    current_user: User = Depends(get_current_user),
):
    chunks = await retrieve_rag_chunks(current_user.id, request.query, force=True)
    return [
        RagSearchResult(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            file_id=chunk.file_id,
            filename=chunk.filename,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            heading_path=chunk.heading_path,
            content=chunk.content,
            score=chunk.score,
            vector_similarity=chunk.vector_similarity,
            text_rank=chunk.text_rank,
            source_types=list(chunk.source_types),
        )
        for chunk in chunks
    ]
