"""Entry point for running face module as standalone.

Usage:
    python -m src.face detect --image test.jpg
    python -m src.face detect --camera
    python -m src.face recognize --image test.jpg
    python -m src.face register --name "John" --images path/to/images/
    python -m src.face list
    python -m src.face api --port 8000
"""

from .cli import main

if __name__ == "__main__":
    main()
