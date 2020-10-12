export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ls /mnt/default/
nvidia-smi
python3 -c 'import platform; import torch; print("torch version:{}".format(torch.__version__)); print("python version:{}".format(platform.python_version()))'
TEXT=/mnt/default/RetrieveNMT/data/MD/retrieve-en2de-top2/concat/data-bin/
MODEL=/hdfs/resrchvc/t-jianya/RetrieveNMT/data/MD/retrieve-en2de-top2/concat/model/en2de-TM-base2
python3 train.py $TEXT \
    --source-lang en --target-lang de \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler my_inverse_sqrt --warmup-updates 4000 \
    --max-update 1000000 --max-epoch 100 --log-interval 100 --log-format "simple"  \
    --save-interval-updates 4000 --save-interval 100 --keep-interval-updates 5 --max-tokens 10000 \
    --arch my_transformer --save-dir $MODEL \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 0.1 --min-lr 1e-16 \
    --update-freq 2 --ddp-backend=no_c10d --fp16 --share-all-embeddings --max-source-positions 40960 