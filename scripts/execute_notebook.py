"""Ejecuta un notebook desde la raíz del repositorio y lo guarda solo si termina."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = (root / args.notebook).resolve()
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"Ejecutado y guardado: {path.relative_to(root)}")


if __name__ == "__main__":
    main()
