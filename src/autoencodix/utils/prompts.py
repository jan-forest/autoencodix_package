PROMPT = """
You are a bioinformatics expert.
I will provide a latent dimension name and a list of genes identified as top contributors to that dimension (derived using Captum). Your task is to analyze only the genes provided and derive biologically plausible interpretations of what the latent dimension may represent.
Please produce the following:
1. A concise explanation of the dominant biological themes represented by the provided gene set.
2. One to three mechanistic hypotheses describing potential biological processes, regulatory programs, or cellular states captured by this latent dimension.
3. A summary of pathways, processes, or molecular functions that may be implicated (using only GO, KEGG, or other standard pathway resources if accessible).
4. A TLDR summarizing the key biological insight.
Output Format:
Output ONLY a valid JSON object with the following keys. Do not include any other text, explanations, or markdown outside the JSON. Use double quotes around text blocks (strings) but never inside a text block. Ensure no trailing commas.
- "TLDR": a one-sentence high-level summary (string without quotes).
- "DETAILS": an object with these exact keys:
  - "dominant_themes": concise explanation of the dominant biological themes (string without quotes).
  - "hypotheses": array of 1-3 mechanistic hypotheses (array of strings without quotes).
  - "pathways_summary": summary of implicated pathways/processes/functions (string without quotes).
Constraints:
- Do not invent gene names or functions. Use only the genes provided.
- Base all interpretations strictly on the supplied gene list and standard biological knowledge resources.
Input genes:
{gene_block}
"""
