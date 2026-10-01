from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="RealSaS Platform", version="0.1.0")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/contracts/product-invariants")
    async def product_invariants() -> dict[str, object]:
        return {
            "render_is_compile": False,
            "product_revision_immutable": True,
            "attempt_may_mutate_product_current": False,
            "promotion_requires_transaction": True,
            "artifact_bytes_immutable": True,
        }

    return app


app = create_app()
