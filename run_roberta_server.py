from __future__ import annotations

import uvicorn

from config import settings


def main() -> None:
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
