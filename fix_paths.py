import glob, re, os

py_files = [f for f in glob.glob("**/*.py", recursive=True)
            if ".venv" not in f and "fix_" not in f]

DATA_DIR_HEADER = (
    "import os\n\n"
    "os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)\n"
    "DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')\n"
)

changed = 0
for path in py_files:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Only touch files that use './data/' paths
    if "'./data/" not in content and '"./data/' not in content:
        continue

    new_content = content

    # Add DATA_DIR header if not already present
    if "DATA_DIR" not in new_content:
        # Insert after first block of imports (find first non-import, non-comment line)
        lines = new_content.splitlines(keepends=True)
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("#") or stripped == "":
                insert_idx = i + 1
            else:
                break
        # If 'import os' already exists, don't add it again
        os_import = "" if "import os\n" in new_content else "import os\n\n"
        header = (
            os_import +
            "os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)\n"
            "DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')\n"
        )
        lines.insert(insert_idx, header)
        new_content = "".join(lines)

    # Replace all './data/...' path strings with os.path.join(DATA_DIR, '...')
    def replace_path(m):
        quote = m.group(1)
        filename = m.group(2)
        return f"os.path.join(DATA_DIR, {quote}{filename}{quote})"

    new_content = re.sub(r"(['\"])\.\/data\/([^'\"]+)\1", replace_path, new_content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed: {path}")
        changed += 1

print(f"\n{changed} files updated.")
