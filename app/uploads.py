"""Bound multipart input before parsing and keep uploaded files in memory."""

from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ddpm_derm.deploy import MAX_UPLOAD_BYTES

MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024


class MemoryMultipartParser(MultiPartParser):
    spool_max_size = MAX_REQUEST_BYTES
    max_file_size = MAX_REQUEST_BYTES  # Starlette versions before 0.46


async def image_upload(request: Request):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            length = int(length)
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length.") from exc
        if length < 0:
            raise HTTPException(400, "Invalid Content-Length.")
        if length > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Upload request is too large.")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Upload request is too large.")
        body.extend(chunk)

    async def stream():
        yield bytes(body)

    parser = MemoryMultipartParser(
        request.headers, stream(), max_files=1, max_fields=0,
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise HTTPException(400, "Invalid multipart image upload.") from exc
    try:
        image = form.get("image")
        if not isinstance(image, UploadFile):
            raise HTTPException(422, "An image file is required.")
        yield image
    finally:
        await form.close()
