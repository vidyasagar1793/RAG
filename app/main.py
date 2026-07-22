from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.db.qdrant import init_qdrant_collections, qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_qdrant_collections()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Verify the API and databases are running."""
    health_status = {
        "api": "healthy",
        "postgres": "unhealthy",
        "qdrant": "unhealthy",
    }

    try:
        await db.execute(text("SELECT 1"))
        health_status["postgres"] = "healthy"
    except Exception:
        pass

    try:
        await qdrant_client.get_collections()
        health_status["qdrant"] = "healthy"
    except Exception:
        pass

    return health_status