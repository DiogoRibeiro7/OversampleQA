import shutil
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "examples"
    dst_dir = repo_root / "docs" / "gallery_examples"
    dst_dir.mkdir(parents=True, exist_ok=True)

    keep_names = set()

    # Sync top-level example scripts
    for path in src_dir.glob("*.py"):
        shutil.copy2(path, dst_dir / path.name)
        keep_names.add(path.name)

    header = src_dir / "GALLERY_HEADER.rst"
    if header.exists():
        shutil.copy2(header, dst_dir / header.name)
        keep_names.add(header.name)

    # Remove stale files
    for path in dst_dir.iterdir():
        if path.is_file() and path.name not in keep_names:
            path.unlink()

    print(f"Synced gallery examples to {dst_dir}")


if __name__ == "__main__":
    main()
