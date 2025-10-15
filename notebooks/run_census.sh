#!/bin/bash
for i in {0..12};do
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=census_download
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=1
	#SBATCH --cpus-per-task=6
	#SBATCH --partition=paul
	#SBATCH --time=24:00:00
	#SBATCH --mem=128G
	if [ -z "$VIRTUAL_ENV" ];
	then 
		source .venv/bin/activate
		echo $VIRTUAL_ENV
	fi
	python notebooks/get_census_chunks.py $i
	exit 0
	EOT
done