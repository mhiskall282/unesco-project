import json
import re

# Load metrics
with open('logs/bwoa_v2_metrics.json', 'r') as f:
    bwoa_metrics = json.load(f)

cm = bwoa_metrics['confusion_matrix']
classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Calculate per-class metrics
per_class_results = []
for i, class_name in enumerate(classes):
    tp = cm[i][i]
    fp = sum(cm[k][i] for k in range(len(classes)) if k != i)
    fn = sum(cm[i][k] for k in range(len(classes)) if k != i)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    per_class_results.append({
        'class': class_name,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Format markdown table
table_rows = [
    "| Class | Precision | Recall | F1 |",
    "| :--- | :---: | :---: | :---: |"
]
for res in per_class_results:
    table_rows.append(f"| {res['class']} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1']:.4f} |")

new_table = "\n".join(table_rows)
print("Computed Per-Class Table:")
print(new_table)

# Update docs/results.md
with open('docs/results.md', 'r') as f:
    results_content = f.read()

# Find the section "## 4. Per-Class Performance"
pattern = r"(## 4\. Per-Class Performance \(BWOA Optimized Model\)\s*\n[^\n]*\n\n\| Class \| Precision \| Recall \| F1 \|\n\| :--- \| :---: \| :---: \| :---: \|\n)(?:\|[^\n]*\n)*"

match = re.search(pattern, results_content)
if match:
    prefix = match.group(1)
    # Reconstruct the block
    body = "\n".join(table_rows[2:]) + "\n"
    updated_results = results_content[:match.start()] + prefix + body + results_content[match.end():]
    with open('docs/results.md', 'w') as f:
        f.write(updated_results)
    print("Successfully updated docs/results.md")
else:
    print("Pattern not found in docs/results.md")
