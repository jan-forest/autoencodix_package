##### STEP 0 - Definitions #####
import sys

data_folder = "./data/census_chunks/"
llm_ontology_folder = "./data/llm_ontologies/"
max_chunk_size = 1000 # Downsample chunks larger than this size
perc_genes = 0.05 # Filter cells with lower than this percentage of genes expressed

step_from_cli = sys.argv[1]  # "step1", "step2", "..."

test_mode = sys.argv[3] if len(sys.argv) > 3 else "no"

# Create folder if it doesn't exist
import os
if not os.path.exists(data_folder):
	os.makedirs(data_folder)

##### STEP 1 - Determine census chunks #####
## Check if human_meta_counts_healthy-diseased.pkl already exists
if step_from_cli == "step1":
	print("STEP 1 - Determine census chunks")

	import cellxgene_census
	census = cellxgene_census.open_soma()

	value_filter="is_primary_data == True" # Only original data publications to avoid duplicates
	species = "homo_sapiens"

	column_levels = ["tissue_general", "cell_type_ontology_term_id", "sex", "disease"] # Will define chunks

	print("Retrieving metadata")
	## Retrieve metadata
	human_meta = (
		census["census_data"][species]
		.obs.read(column_names=column_levels, value_filter=value_filter)
		# .obs.read(column_names=["tissue_general", "cell_type", "sex"], value_filter=value_filter)
		.concat()
		.to_pandas()
	)
	census.close()

	## Combine all non-healthy disease states into a single category
	human_meta["healthy"] = human_meta["disease"].apply(lambda x: "healthy" if x == "normal" else "diseased")
	human_meta = human_meta[["tissue_general", "cell_type_ontology_term_id", "sex", "healthy"]]

	print("Sample counts for top chunks:")
	print(human_meta.value_counts().head(10))
	print("Number of chunks:", human_meta.value_counts().shape[0])

	## Keep only where value count is larger than 1000
	human_meta_counts = human_meta.value_counts()
	human_meta_counts = human_meta_counts[human_meta_counts > 1000]
	print("Number of chunks with more than 1000 samples:", human_meta_counts.shape[0])
	print("Total number of samples in these chunks:", human_meta_counts.sum())

	## Save human_meta_counts 
	human_meta_counts.to_pickle(os.path.join(data_folder, "human_meta_counts_healthy-diseased.pkl"))

##### STEP 2 - Get census data chunks #####
if step_from_cli == "step2":
	import math
	import pandas as pd
	import glob
	import cellxgene_census
	import scanpy

	print("STEP 2 - Get census data chunks")

	## Read human_meta_counts_healthy-diseased.pkl if not already in memory
	human_meta_counts = pd.read_pickle(os.path.join(data_folder, "human_meta_counts_healthy-diseased.pkl"))

	## Get all unique gene names from LLM ontologies
	print("Retrieving gene names from LLM ontologies")
	tsv_files = glob.glob(os.path.join(llm_ontology_folder, "*_level2.tsv"))

	gene_names = set()
	for file in tsv_files:
		df = pd.read_csv(file, sep='\t', usecols=[0])
		gene_names.update(df.iloc[:, 0].dropna().unique())

	print(f"Total unique gene names: {len(gene_names)}")
	# gene_names = list(gene_names)[0:100]  # Limit to first 100 for testing

	## Define filter to get data only for these genes
	var_value_filter="feature_id in [" + ", ".join(f'"{gene}"' for gene in gene_names) + "]"

	## Handle command line arguments for parallel processing
	if len(sys.argv) > 2:
		part = int(sys.argv[2])
	else:
		part = 0
	
	total_parts = 12
	
	if test_mode == "test":
		print("Test mode: only process last 24 chunks")
		all_chunks = human_meta_counts.index[::-1][0:24] # Last 24 chunks for testing  
	else:
		all_chunks = human_meta_counts.index[:]
	num_chunks = len(all_chunks)
	chunks_per_part = math.ceil(num_chunks / total_parts)
	start_idx = part * chunks_per_part
	end_idx = min((part + 1) * chunks_per_part, num_chunks)
	selected_chunks = all_chunks[start_idx:end_idx]

	print("Processing part info")
	print(len(selected_chunks), part, start_idx, end_idx, chunks_per_part, num_chunks)

	for chunk_name in selected_chunks:	
		print(pd.Timestamp.now())
		print(f"Processing chunk: {chunk_name}")
		print(f"Number of cells in chunk: {human_meta_counts.loc[chunk_name[0],chunk_name[1],chunk_name[2],chunk_name[3]]}")
		if chunk_name[3] == "healthy":
			obs_value_filter = f"tissue_general in ['{chunk_name[0]}'] and cell_type_ontology_term_id in ['{chunk_name[1]}'] and sex in ['{chunk_name[2]}'] and disease in ['normal'] and is_primary_data == True"
		elif chunk_name[3] == "diseased":
			obs_value_filter = f"tissue_general in ['{chunk_name[0]}'] and cell_type_ontology_term_id in ['{chunk_name[1]}'] and sex in ['{chunk_name[2]}'] and disease not in ['normal'] and is_primary_data == True"

		with cellxgene_census.open_soma() as census:
			adata_chunk = cellxgene_census.get_anndata(
				census=census, 
				organism="Homo sapiens",
				var_value_filter=var_value_filter,
				obs_value_filter=obs_value_filter,  
			)
		census.close()

		print("Filter and downsample")
		scanpy.pp.filter_cells(adata_chunk, min_genes=int(perc_genes * len(gene_names)), inplace=True)
		if adata_chunk.shape[0] > max_chunk_size:
			adata_chunk = scanpy.pp.subsample(adata_chunk, n_obs=max_chunk_size, copy=True)
		
		print("Write chunk to h5ad file")
		scanpy.write(data_folder + f"{chunk_name[0].replace(' ', '-')}_{chunk_name[1].replace(':','-').replace(' ', '-')}_{chunk_name[2].replace(' ', '-')}_{chunk_name[3]}_chunk.h5ad", adata_chunk)

		del adata_chunk

