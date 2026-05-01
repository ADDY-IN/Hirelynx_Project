from fastapi import FastAPI
from app.core.config import settings
from app.api.v1 import health, parser, scoring, admin, candidate, recruiter, employer

app = FastAPI(
    title=settings.PROJECT_NAME, 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Include Routers ---

# V1 API
app.include_router(health.router, tags=["Health"])
app.include_router(parser.router, prefix="/v1")
app.include_router(scoring.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")
app.include_router(candidate.router, prefix="/v1")
app.include_router(recruiter.router, prefix="/v1")
app.include_router(employer.router, prefix="/v1")

@app.on_event("startup")
async def startup_event():
    print(f"Starting {settings.PROJECT_NAME}...")

@app.on_event("shutdown")
async def shutdown_event():
    print(f"Shutting down {settings.PROJECT_NAME}...")
