from __future__ import annotations
# scripts/add_future_annotations.py
from pathlib import Path

root = Path("/home/workspace/projects/transformer/UPT/src/the_well")
for p in root.rglob("*.py"):
    print(p)
    txt = p.read_text(encoding="utf-8")
    if "from __future__ import annotations" in txt.splitlines()[:5]:
        continue
    p.write_text("from __future__ import annotations\n" + txt, encoding="utf-8")
print("Patched future import into files.")

