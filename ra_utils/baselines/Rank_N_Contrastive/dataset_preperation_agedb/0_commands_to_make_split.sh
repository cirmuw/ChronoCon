find    /home/cwatzenboeck/data/public/agedb/AgeDB/ -name *.jpg  > /home/cwatzenboeck/data/public/agedb/tabular/01_agedb_paths_no_header.csv  

python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/extract_infos_from_name.py  \
   -i /home/cwatzenboeck/data/public/agedb/tabular/01_agedb_paths_no_header.csv \
   -o /home/cwatzenboeck/data/public/agedb/tabular/02_agedb_paths_with_infos.csv



python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/create_stratified_split.py \
  -i /home/cwatzenboeck/data/public/agedb/tabular/02_agedb_paths_with_infos.csv \
  -o /home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv  



python /home/cwatzenboeck/code/RA/ra_utils/ra_utils/baselines/Rank_N_Contrastive/dataset_preperation_agedb/check_splits.py \
  -i /home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv  






