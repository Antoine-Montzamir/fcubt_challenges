import glob, re

py_files = [f for f in glob.glob("**/*.py", recursive=True) if ".venv" not in f and "fix_fdapy" not in f]

pattern_dense_dict = re.compile(r"DenseFunctionalData\(\s*\{([^}]+)\}\s*,\s*([^)]+)\)")
pattern_dense_obj = re.compile(r"DenseFunctionalData\(\s*([\w.]+\.argvals)\s*,\s*([^)]+)\)")

changed_files = []

for path in py_files:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if "DenseFunctionalData" not in content:
        continue

    needs_denseargvals = [False]
    needs_densevalues = [False]

    def replace_dict(m):
        needs_denseargvals[0] = True
        needs_densevalues[0] = True
        dict_content = m.group(1)
        values = m.group(2).strip()
        return f"DenseFunctionalData(DenseArgvals({{{dict_content}}}), DenseValues({values}))"

    def replace_obj(m):
        needs_densevalues[0] = True
        argvals = m.group(1)
        values = m.group(2).strip()
        return f"DenseFunctionalData({argvals}, DenseValues({values}))"

    new_content = pattern_dense_dict.sub(replace_dict, content)
    new_content = pattern_dense_obj.sub(replace_obj, new_content)

    import_line = ""
    if needs_denseargvals[0] and "DenseArgvals" not in content:
        import_line += "from FDApy.representation.argvals import DenseArgvals\n"
    if needs_densevalues[0] and "DenseValues" not in content:
        import_line += "from FDApy.representation.values import DenseValues\n"

    if import_line:
        lines = new_content.splitlines(keepends=True)
        last_fdapy_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("from FDApy") or line.startswith("import FDApy"):
                last_fdapy_idx = i
        if last_fdapy_idx >= 0:
            lines.insert(last_fdapy_idx + 1, import_line)
            new_content = "".join(lines)
        else:
            new_content = import_line + new_content

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed: {path}")
        changed_files.append(path)

print(f"\n{len(changed_files)} files updated.")
