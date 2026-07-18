from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings

# asyncpg requires a slightly different connection string prefix
async_db_url = settings.postgres_uri.replace("postgresql://", "postgresql+asyncpg://")

# The engine manages the connection pool
engine = create_async_engine(async_db_url, echo=False)

# The sessionmaker generates individual sessions for each API request
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db():
    """FastAPI Dependency to yield a database session per request."""
    async with AsyncSessionLocal() as session:
        yield session