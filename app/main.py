from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ddpm_derm.deploy import (
    DISCLAIMER,
    MAX_UPLOAD_BYTES,
    ClassifierService,
    DeploymentSettings,
    UploadValidationError,
    decode_uploaded_image,
)

INDEX_PATH = Path(__file__).with_name("index.html")


class Probability(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: list[Probability]
    disclaimer: str
    model: dict[str, Any]


class GalleryItemResponse(BaseModel):
    image_id: str
    url: str


class GalleryResponse(BaseModel):
    version: str
    selection: str
    items: list[GalleryItemResponse]
    disclaimer: str


def create_app(service=None, settings: DeploymentSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or ClassifierService(settings)
        yield

    app = FastAPI(
        title="HAM10000 classifier portfolio demo",
        version="1.0.0",
        description=DISCLAIMER,
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        if not INDEX_PATH.is_file():
            raise HTTPException(status_code=500, detail="demo UI file is missing")
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return request.app.state.service.health()

    @app.get("/api/gallery", response_model=GalleryResponse)
    async def gallery(request: Request) -> GalleryResponse:
        runtime = request.app.state.service
        health_info = runtime.health()
        items = [
            GalleryItemResponse(
                image_id=item.image_id,
                url=f"/gallery/{item.image_id}",
            )
            for item in runtime.gallery(limit=24)
        ]
        return GalleryResponse(
            version=health_info["gallery_version"],
            selection="First 24 manifest rows; no manual quality selection.",
            items=items,
            disclaimer=DISCLAIMER,
        )

    @app.get("/gallery/{image_id}", include_in_schema=False)
    async def gallery_image(image_id: str, request: Request) -> FileResponse:
        try:
            path = request.app.state.service.gallery_path(image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="gallery image not found") from exc
        return FileResponse(path)

    @app.post("/api/predict", response_model=PredictionResponse)
    async def predict(request: Request, image: UploadFile = File(...)) -> PredictionResponse:
        try:
            data = await image.read(MAX_UPLOAD_BYTES + 1)
            decoded = decode_uploaded_image(image.filename, image.content_type, data)
        except UploadValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        finally:
            await image.close()

        prediction = request.app.state.service.predict(decoded)
        return PredictionResponse(
            **prediction,
            disclaimer=DISCLAIMER,
            model=request.app.state.service.health(),
        )

    return app


app = create_app()
