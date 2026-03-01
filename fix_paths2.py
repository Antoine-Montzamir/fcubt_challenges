import glob, re

py_files = [f for f in glob.glob("**/*.py", recursive=True)
            if ".venv" not in f and "fix_" not in f]

changed = 0
for path in py_files:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if "'./figures/" not in content and '"./figures/' not in content:
        continue

    new_content = content

    # Add FIGURES_DIR if not already present
    if "FIGURES_DIR" not in new_content:
        lines = new_content.splitlines(keepends=True)
        # Find the DATA_DIR line to insert after it, or after imports
        insert_idx = 0
        for i, line in enumerate(lines):
            if "DATA_DIR" in line or line.strip().startswith("import ") or line.strip().startswith("from ") or line.strip().startswith("#") or line.strip() == "":
                insert_idx = i + 1
            else:
                break

        header = (
            "os.makedirs(os.path.join(os.path.dirname(__file__), 'figures'), exist_ok=True)\n"
            "FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')\n"
        )
        lines.insert(insert_idx, header)
        new_content = "".join(lines)

    # Replace './figures/...' with os.path.join(FIGURES_DIR, '...')
    def replace_fig_path(m):
        quote = m.group(1)
        filename = m.group(2)
        return f"os.path.join(FIGURES_DIR, {quote}{filename}{quote})"

    new_content = re.sub(r"(['\"])\.\/figures\/([^'\"]+)\1", replace_fig_path, new_content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed: {path}")
        changed += 1

print(f"\n{changed} files updated.")
