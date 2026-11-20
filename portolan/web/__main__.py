"""`python -m portolan.web`: run the web app under uvicorn on `$PORT`
(default 8000).

Binds `0.0.0.0` when `PORTOLAN_ROLE` is set in the environment, else
`127.0.0.1` (loopback, the safe default on a dev machine).
`docker/entrypoint.sh` exports `PORTOLAN_ROLE=web` when the container
leaves it unset, so a container always binds all interfaces.
"""

from __future__ import annotations

import os

import uvicorn

from portolan.config import get_settings
from portolan.web.app import create_app

app = create_app(get_settings())


def main() -> None:
    host = "0.0.0.0" if os.environ.get("PORTOLAN_ROLE") else "127.0.0.1"
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
