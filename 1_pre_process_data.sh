#$ -j y
#$ -l h_rss=100g
#$ -q goldml_normal
#$ -P dsm-prj
#$ -M dmitry.efimov@aexp.com -m bea
#$ -cwd
/opt/python/python35/bin/python -u preprocess_data.py \
    --settings_file settings.json \
    --fit_boxcox 0


