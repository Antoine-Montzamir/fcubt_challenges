import glob, re

py_files = [f for f in glob.glob("**/*.py", recursive=True)
            if ".venv" not in f and "fix_fdapy" not in f]

changed = 0
for path in py_files:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    new_content = content

    # 1. Remove DenseArgvals and DenseValues imports
    new_content = re.sub(r"from FDApy\.representation\.argvals import DenseArgvals\n", "", new_content)
    new_content = re.sub(r"from FDApy\.representation\.values import DenseValues\n", "", new_content)

    # 2. KarhunenLoeve(n_functions=N, basis_name='B', argvals=DenseArgvals({...}))
    #    -> KarhunenLoeve('B', n_functions=N, argvals={...})
    def fix_karhunen(m):
        n_func = m.group(1)
        basis = m.group(2)
        dict_content = m.group(3)
        return f"KarhunenLoeve('{basis}', n_functions={n_func}, argvals={{{dict_content}}})"
    new_content = re.sub(
        r"KarhunenLoeve\(n_functions=(\S+?),\s*basis_name='(\w+)',\s*argvals=DenseArgvals\(\{([^}]+)\}\)\)",
        fix_karhunen, new_content
    )

    # 3. Also fix without argvals: KarhunenLoeve(n_functions=N, basis_name='B')
    #    -> KarhunenLoeve('B', n_functions=N)  (only if not already fixed)
    new_content = re.sub(
        r"KarhunenLoeve\(n_functions=(\S+?),\s*basis_name='(\w+)'\)",
        r"KarhunenLoeve('\2', n_functions=\1)",
        new_content
    )

    # 4. DenseFunctionalData(DenseArgvals({...}), DenseValues(x))
    #    -> DenseFunctionalData({...}, x)
    def fix_dense_dict(m):
        dict_content = m.group(1)
        values = m.group(2).strip()
        return f"DenseFunctionalData({{{dict_content}}}, {values})"
    new_content = re.sub(
        r"DenseFunctionalData\(DenseArgvals\(\{([^}]+)\}\)\s*,\s*DenseValues\(([^)]+)\)\)",
        fix_dense_dict, new_content
    )

    # 5. DenseFunctionalData(obj.argvals, DenseValues(x)) -> DenseFunctionalData(obj.argvals, x)
    new_content = re.sub(
        r"DenseFunctionalData\(([\w.]+\.argvals),\s*DenseValues\(([^)]+)\)\)",
        r"DenseFunctionalData(\1, \2)",
        new_content
    )

    # 6. Any remaining DenseValues(...) wrapper that snuck through
    new_content = re.sub(r"DenseValues\(([^)]+)\)", r"\1", new_content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed: {path}")
        changed += 1

print(f"\n{changed} files updated.")
