nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3
TEXT=/mnt/default/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/
MODEL=/hdfs/msrmt/t-jianya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/model/CopyNet-de2en
python3 train.py $TEXT \
    --source-lang de --target-lang en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler my_inverse_sqrt --warmup-updates 4000 \
    --max-update 1000000 --max-epoch 100 --log-interval 100 --log-format "simple"  \
    --save-interval-updates 5000 --save-interval 50 --keep-interval-updates 5 --max-tokens 6000 \
    --arch my_transformer --save-dir $MODEL \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 0.1 --min-lr 1e-16 \
    --update-freq 2 --ddp-backend=no_c10d --share-all-embeddings