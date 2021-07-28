#$ -j y
#$ -l h_rss=30g,h=lgpbddgx04
#$ -q GPUDGXML
#$ -P gpudgx-prj
#$ -M dmitry.efimov@aexp.com -m bea
#$ -cwd
python -u train_cdragan.py \
    --settings_file settings.json \
    --version ver3 \
    --device /device:GPU:0

