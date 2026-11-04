"""Execute every lab notebook top to bottom (used by CI). Fails on the first error."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
failed = []
for nb_path in sorted((root / "labs").glob("*/*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    t = time.time()
    try:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}}).execute()
        print(f"ok   {nb_path.relative_to(root)} ({time.time() - t:.1f} s)")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {nb_path.relative_to(root)}: {e}")
        failed.append(nb_path)
sys.exit(1 if failed else 0)
