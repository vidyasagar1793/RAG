from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.db.postgres import get_db
from app.db.qdrant import qdrant

# ... keep your existing create_app() setup ...
app = FastAPI()

@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Verify the API and databases are running."""
    health_status = {
        "api": "healthy",
        "postgres": "unhealthy",
        "qdrant": "unhealthy"
    }

    # 1. Test PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        health_status["postgres"] = "healthy"
    except Exception:
        pass

    # 2. Test Qdrant
    try:
        collections = await qdrant.get_collections()
        health_status["qdrant"] = "healthy"
    except Exception:
        pass

    return health_status