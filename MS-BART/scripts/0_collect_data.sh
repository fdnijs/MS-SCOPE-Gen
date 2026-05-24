### MIST
mkdir -p data/MassSpecGym/mist

wget https://zenodo.org/records/11580401/files/MassSpecGym_mist_data.zip -O data/MassSpecGym/mist/mist.zip
unzip data/MassSpecGym/mist/mist.zip -d data/MassSpecGym/mist/tmp

mv data/MassSpecGym/mist/tmp/*/* data/MassSpecGym/mist/
rm -r data/MassSpecGym/mist/tmp
rm data/MassSpecGym/mist/mist.zip

# Get mist checkpoint (original MassSpecGym folds used)
mkdir -p data/MassSpecGym/mist/ckpts
wget -O data/MassSpecGym/mist/ckpts/original.ckpt https://zenodo.org/records/11580401/files/mist_fp.ckpt


### MS-BART data

# Pretrain Data
mkdir -p data/MassSpecGym/molecules
wget -O data/MassSpecGym/molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv

# MassSpecGym 
wget -O data/MassSpecGym/MassSpecGym.tsv https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym.tsv
