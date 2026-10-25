import os
import glob

sql_section = """
## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.
"""

skip_dirs = ['cgo', 'lua']

for readme in glob.glob("*/README.md"):
    sdk_name = readme.split('/')[0]
    if sdk_name in skip_dirs:
        continue
        
    with open(readme, 'r') as f:
        content = f.read()
        
    if "Unified SQL Engine" in content:
        continue
        
    lines = content.split('\n')
    out_lines = []
    inserted = False
    
    for line in lines:
        if line.startswith("## ") and not inserted and "Features" not in line and "Unified" not in line:
            out_lines.append(sql_section.strip())
            out_lines.append("")
            inserted = True
            
        out_lines.append(line)
        
    with open(readme, 'w') as f:
        f.write('\n'.join(out_lines))
        
print("Updated all READMEs!")
