"""Convert a '# %%' cell-marked Python file to a notebook and execute it top to bottom.

    python tools/py2nb.py exp01-signals/exp01.py

The notebook is written next to the source (exp01.ipynb) and executed on a
fresh kernel with the notebook's folder as working directory, so a notebook
that runs here has no hidden state and no dead cells.
"""
import os
import sys

import nbformat
from nbclient import NotebookClient


def split_cells(src):
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if kind is not None and text:
            if kind == "md":
                text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
                cells.append(nbformat.v4.new_markdown_cell(text))
            else:
                cells.append(nbformat.v4.new_code_cell(text))

    for line in src.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(path):
    with open(path, encoding="utf-8") as fh:
        nb = nbformat.v4.new_notebook(cells=split_cells(fh.read()))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    folder = os.path.dirname(os.path.abspath(path))
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": folder}}).execute()
    out = os.path.splitext(path)[0] + ".ipynb"
    nbformat.write(nb, out)
    print("executed", out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
