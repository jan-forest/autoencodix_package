import os
import pandas as pd
import glob
import cellxgene_census
import scanpy
import sys
import math

folder = "./notebooks/census_chunks/"
# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

## Read human_meta_counts.pkl
# human_meta_counts = pd.read_pickle(folder + "human_meta_counts.pkl")
human_meta_counts = pd.read_pickle(folder + "human_meta_counts_healthy-diseased.pkl")


max_chunk_size = 1000
perc_genes = 0.05

# Find all files ending with "_level1.tsv" in the folder
tsv_files = glob.glob("./notebooks/llm_ontologies/*_level2.tsv")

# Read only the first column (gene names) from each file and collect unique gene names
gene_names = set()
for file in tsv_files:
	df = pd.read_csv(file, sep='\t', usecols=[0])
	gene_names.update(df.iloc[:, 0].dropna().unique())

print(f"Total unique gene names: {len(gene_names)}")

# gene_names = list(gene_names)[0:100]  # Limit to first 100 for testing
var_value_filter="feature_id in [" + ", ".join(f'"{gene}"' for gene in gene_names) + "]"


# Get part number from command line argument (default to 0 if not provided)
if len(sys.argv) > 1:
	part = int(sys.argv[1])
else:
	part = 0

total_parts = 12
# all_chunks = human_meta_counts.index[::-1][:]  # Change [0:5] to [:] for all chunks in production
all_chunks = human_meta_counts.index[0:11]  # Change [0:5] to [:] for all chunks in production
num_chunks = len(all_chunks)
chunks_per_part = math.ceil(num_chunks / total_parts)
start_idx = part * chunks_per_part
end_idx = min((part + 1) * chunks_per_part, num_chunks)
selected_chunks = all_chunks[start_idx:end_idx]

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
	scanpy.write(folder + f"{chunk_name[0].replace(' ', '-')}_{chunk_name[1].replace(':','-').replace(' ', '-')}_{chunk_name[2].replace(' ', '-')}_{chunk_name[3]}_chunk.h5ad", adata_chunk)

	del adata_chunk
