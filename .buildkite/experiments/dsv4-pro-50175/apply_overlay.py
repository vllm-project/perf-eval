import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


here = Path(__file__).resolve().parent
arm = sys.argv[1]
manifest = json.loads((here / "manifest.json").read_text())
root = Path(importlib.util.find_spec("vllm").origin).parent.parent
if arm != "client":
    archive = zipfile.ZipFile(here / f"overlay-{arm}.zip")
    for change in manifest["arms"][arm]["files"]:
        target = root / change["path"]
        actual = sha256(target) if target.exists() else None
        if actual != change["base_sha256"]:
            raise RuntimeError(f"Base source mismatch: {target}: {actual}")
    for change in manifest["arms"][arm]["files"]:
        target = root / change["path"]
        if change["target_sha256"] is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(change["path"]))
            assert sha256(target) == change["target_sha256"]
        for cached in (target.parent / "__pycache__").glob(target.stem + ".*.pyc"):
            cached.unlink()

packages = sorted(
    (d.metadata["Name"], d.version)
    for d in importlib.metadata.distributions()
    if d.metadata["Name"]
)
native = {str(p.relative_to(root)): sha256(p) for p in (root / "vllm").rglob("*.so")}
result = {
    "arm": arm,
    "target_python_revision": manifest["arms"][arm]["commit"] if arm != "client" else manifest["base_commit"],
    "base_digest": manifest["base_digest"],
    "packages": packages,
    "native_extensions": native,
}
(here / "identity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("packages", "native_extensions")}))
