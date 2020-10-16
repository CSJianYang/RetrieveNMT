nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3
TEXT=/mnt/default/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-es/es2en/top2/data-bin/
MODEL=/hdfs/msrmt/t-jianya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-es/es2en/top2/model/CSTM-es2en
python3 train.py $TEXT \
    --source-lang es --target-lang en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler my_inverse_sqrt --warmup-updates 4000 \
    --max-update 1000000 --max-epoch 50 --log-interval 100 --log-format "simple"  \
    --save-interval-updates 5000 --save-interval 100 --keep-interval-updates 5 --max-tokens 6000 \
    --arch my_transformer --save-dir $MODEL \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 0.1 --min-lr 1e-16 \
    --update-freq 2 --ddp-backend=no_c10d --share-all-embeddings