"""Fix Unicode symbols in main.py that break cp1250 terminals."""
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = "        print(f\"  {r['task_id'][:30]:<30} {r['score']:>6.2f} {'\u2713' if r['matched'] else '\u2717':>6}  {r['cause']}\")"
new = (
    "        matched_str = 'Y' if r['matched'] else 'N'\n"
    "        print(f\"  {r['task_id'][:30]:<30} {r['score']:>6.2f} {matched_str:>6}  {r['cause']}\")"
)

if old in content:
    content = content.replace(old, new)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed: Unicode symbols replaced with ASCII Y/N")
else:
    print("Pattern not found, trying fallback...")
    # Fallback: find the line by unicode char presence
    lines = content.splitlines(keepends=True)
    fixed = False
    for i, line in enumerate(lines):
        if "\u2713" in line or "\u2717" in line:
            lines[i] = (
                "        matched_str = 'Y' if r['matched'] else 'N'\n"
                "        print(f\"  {r['task_id'][:30]:<30} {r['score']:>6.2f} {matched_str:>6}  {r['cause']}\")\n"
            )
            fixed = True
            print(f"Fixed line {i+1}")
    if fixed:
        with open("main.py", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("Done")
    else:
        print("No unicode found - already clean")
