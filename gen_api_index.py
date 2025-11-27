import os
from pathlib import Path

api_dir = Path('docs/api')
output_file = api_dir / 'index.md'

# Get all .md files except index.md, sorted alphabetically
modules = sorted(f for f in os.listdir(api_dir) if f.endswith('.md') and f != 'index.md')

# Generate Markdown content
content = '# API Reference\n\nThe API is organized by module. Select a module below to view its documentation.\n\n'
for module in modules:
    module_name = module.replace('.md', '')
    content += f'- [{module_name}]({module})\n'

# Write to index.md
with open(output_file, 'w') as f:
    f.write(content)