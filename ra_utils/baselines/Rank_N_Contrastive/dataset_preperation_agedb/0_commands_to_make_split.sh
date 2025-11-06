find    /home/cwatzenboeck/data/public/agedb/AgeDB/ -name *.jpg  > /home/cwatzenboeck/data/public/agedb/tabular/01_agedb_paths_no_header.csv  

# Get some extra infos from file name (age, sex, id)
python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/extract_infos_from_name.py  \
   -i /home/cwatzenboeck/data/public/agedb/tabular/01_agedb_paths_no_header.csv \
   -o /home/cwatzenboeck/data/public/agedb/tabular/02_agedb_paths_with_infos.csv



# Split:   AgeDB-STRAT
python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/create_stratified_split.py \
  -i /home/cwatzenboeck/data/public/agedb/tabular/02_agedb_paths_with_infos.csv \
  -o /home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv  

# Check for leakage (are people exclusively in seperate splits?)
python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/check_splits.py \
  -i /home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv  



# Split:   AgeDB-DIR
# +  Comparible to work by others. Use original splits of published papers. 
# +  Test set has uniform distribution of ages (at least for range ~20 - ~65); Training set does not. Good for testing disbalance.
# -  Leakage
## Merge with old split -> Comparible with LDS, FDS, RankSIM, ... ; Comparible with file names on linux file system 
python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/merge_with_agedbDIR_splits.py \
  --old_splits_file_to_use /home/cwatzenboeck/data/public/agedb/tabular/agedb_splits_RnC_paper.csv  \
  --metadata_file /home/cwatzenboeck/data/public/agedb/tabular/02_agedb_paths_with_infos.csv
  --output /home/cwatzenboeck/data/public/agedb/tabular/03B_agedb_splits_DIR.csv


