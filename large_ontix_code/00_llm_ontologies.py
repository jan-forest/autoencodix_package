# ### Define prompt ###
# # number of latent dimensions
# latent_dims = [10, 24]
# second_level_range = "5-20"
# # Ontology sources to consider
# ontology_sources = ["KEGG", "Reactome", "Gene Ontology"]
# # Organism
# organism = "human"
# # Example json format
# example_json_file = "./data/llm_ontologies/example.json"
# # Read example json format
# with open(example_json_file, "r") as f:
# 	example_json = f.read()

# prompt_combinations = []

# for dims in latent_dims:	

# 	prompt_template = f"""Define a list of {dims} dimensions which describe the state, functions, metabolism and role of cells in {organism}.
# 	The dimensions should be distinct and comprise all relevant aspects of cellular biology to discriminate all cells in the {organism} body along those dimensions.
# 	The dimensions should be useful to classify and compare cells within an organism like {organism}.
# 	Phrase and adjust the dimensions such that they measurable by gene expression data.
# 	Finally, assign each of the ten dimension the {second_level_range} most relevant and characterizing pathways or ontology terms with their respective identifier from databases like {" or ".join(ontology_sources)}.
# 	Please, avoid largely overlapping and duplicated terms from different sources. For example, use only TCA cycle from either KEGG or Reactome but not both as terms or pathways linked to a dimension.
# 	Try to identify the {second_level_range} most important but distinct subterms regardless of their database source. Give your answer in the following JSON format:
# 	{example_json}
# 	Make sure to follow the JSON format exactly without any additional text outside the JSON structure.
# 	"""
# 	prompt_combinations.append(prompt_template)

# ### Save prompts to txt-file ###
# import os

# output_dir = "./data/llm_ontologies/prompts"
# os.makedirs(output_dir, exist_ok=True)

# for i, prompt in enumerate(prompt_combinations):
#     with open(os.path.join(output_dir, f"prompt_{latent_dims[i]}.txt"), "w") as f:
#         f.write(prompt)
# #############################################

### Retrieve Gene IDs for Ontology Terms ###
import os
import json
import mygene

def get_genes_for_id(identifier, scope):
    """
    Retrieves HUGO gene symbols for a given pathway or GO identifier.

    Args:
        identifier (str): The ID to query (e.g., 'GO:0007173').
        scope (str): The type of identifier (e.g., 'go', 'pathway.reactome').

    Returns:
        list: A list of gene symbols, or an empty list if none are found.
    """
    mg = mygene.MyGeneInfo()
    # We query for the identifier and ask for only the 'symbol' field back.
    # The 'hits' key contains the list of matching genes.
    if isinstance(identifier, str):
        result = mg.query(identifier, scopes=scope, fields=['symbol','ensembl.gene'], species='human', fetch_all=True, verbose=False)
 
    genes_symbols = []
    genes_ensembl = []
    for hit in result:
        # print(hit)
        if 'symbol' in hit:
            genes_symbols.append(hit['symbol'])
        if 'ensembl' in hit:
            ensembl_data = hit['ensembl']
            if isinstance(ensembl_data, list):
                for ens in ensembl_data:
                    if 'gene' in ens:
                        genes_ensembl.append(ens['gene'])
            elif isinstance(ensembl_data, dict) and 'gene' in ensembl_data:
                genes_ensembl.append(ensembl_data['gene'])
    return genes_symbols, genes_ensembl

ontology_folder = "./data/llm_ontologies/output_llms"
ontology_files = [f for f in os.listdir(ontology_folder) if f.endswith('ontology.json')] # exclude "_with_genes" files

ontologies = []
for filename in ontology_files:
	print(f"Processing ontology file: {filename}")
	filepath = os.path.join(ontology_folder, filename)
	with open(filepath, 'r', encoding='utf-8') as f:
		ontologies.append(json.load(f))

# Mapping from ontology source to mygene scope
source_to_scope = {
	'go': 'go',
	'reactome': 'pathway.reactome',
	'kegg': 'pathway.kegg'
}
i_ont = 0
for ontology in ontologies:
	# Each ontology dict has a single key, e.g., 'gemini_ontology'
	ontology_name = list(ontology.keys())[0]
	print(f"Processing ontology: {ontology_name}")
	dimensions = ontology[ontology_name].keys()
	print(ontology_name)
	for dim in dimensions:
		print(dim)
		t_int = 0
		for term in ontology[ontology_name][dim]['ontology_terms']:
			
			# Each term is a dict with 'id' and 'source' keys
			# We will add a 'genes' key to each term with the list of gene symbols
			term_id = term['id']
			# print(term_id)
			source = term['source'].lower()
			scope = source_to_scope.get(source)
			if term_id and scope:
				genes_symbols, genes_ensembl = get_genes_for_id(term_id, scope)
				term['genes_hugo'] = genes_symbols  # Add the list of gene symbols to the term
				term['ensembl'] = genes_ensembl  # Add the list of Ensembl gene IDs to the term

			ontologies[i_ont][ontology_name][dim]['ontology_terms'][t_int] = term  # Update the terms with genes
			t_int += 1
	
	i_ont += 1

# ontologies now contains gene lists for each ontology term
# Save updated ontologies with genes to new JSON files
for i, ontology_dict in enumerate(ontologies):
	# Each ontology_dict has a single key, e.g., 'gemini_ontology'
	ontology_name = list(ontology_dict.keys())[0]
	filename = f"{ontology_name}_with_genes.json"
	filepath = os.path.join(ontology_folder, filename)
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(ontology_dict, f, ensure_ascii=False, indent=2)


### Infos about ontologies ###
# Calculate the number of unique genes for each ontology
ontology_unique_gene_counts = {}
ontology_gene_sets = {}

for ontology_dict in ontologies:
	ontology_name = list(ontology_dict.keys())[0]
	all_genes = set()
	for dim in ontology_dict[ontology_name]:
		for term in ontology_dict[ontology_name][dim]['ontology_terms']:
			if 'genes_hugo' in term:
				all_genes.update(term['genes_hugo'])
	ontology_unique_gene_counts[ontology_name] = len(all_genes)
	ontology_gene_sets[ontology_name] = all_genes

print(ontology_unique_gene_counts)

# Calculate the number of overlapping genes between all ontology_name
ontology_names = list(ontology_gene_sets.keys())
if len(ontology_names) > 1:
	overlap_genes = set.intersection(*(ontology_gene_sets[name] for name in ontology_names))
	print(f"Number of overlapping genes among all ontologies: {len(overlap_genes)}")
else:
	print("Only one ontology present, no overlap calculation.")


final_folder = "./data/llm_ontologies/final_ontologies"
os.makedirs(final_folder, exist_ok=True)
## Save ontology in AUTOENCODIX format
for ontology_dict in ontologies:
	ontology_name = list(ontology_dict.keys())[0]
	level1_rows = []
	level2_rows = []
	for dim in ontology_dict[ontology_name]:
		for term in ontology_dict[ontology_name][dim]['ontology_terms']:
			term_id = term.get('id', '')
			# Level 1: term_id <tab> dimension
			level1_rows.append(f"{term_id}\t{dim}")
			# Level 2: ensembl_gene <tab> term_id
			for ensembl_id in term.get('ensembl', []):
				level2_rows.append(f"{ensembl_id}\t{term_id}")

	# Write level1 file
	level1_filename = f"{ontology_name}__ensembl_level1.tsv"
	with open(os.path.join(final_folder, level1_filename), 'w', encoding='utf-8') as f1:
		f1.write('\n'.join(level1_rows))

	# Write level2 file
	level2_filename = f"{ontology_name}__ensembl_level2.tsv"
	with open(os.path.join(final_folder, level2_filename), 'w', encoding='utf-8') as f2:
		f2.write('\n'.join(level2_rows))