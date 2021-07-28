export CPATH=${CPATH}:/usr/local/cuda/include
export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

#$ -j y
#$ -l h_rss=40g,h=lgpbd1671
#$ -q gpuml
#$ -P gpu-prj
#$ -M dmitry.efimov@aexp.com -m bea
#$ -cwd
/opt/python/python35/bin/python -u generate_cdragan.py \
    --settings_file settings.json \
    --version ver3 \
    --device /device:GPU:0 \
    --segment as_train \
    --num_sample_sim 4000000 \
    --using_iter 0

