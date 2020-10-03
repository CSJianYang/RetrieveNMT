nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEXT=/hdfs/msrmt/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-es/data-bin/
MODEL=/hdfs/msrmt/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-es/model/en2es-base
python3 /hdfs/msrmt/v-jiaya/RetriveNMT/fairseq-baseline/train.py $TEXT \
    --source-lang en --target-lang es \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler my_inverse_sqrt --warmup-updates 4000 \
    --max-update 1000000 --max-epoch 100 --log-interval 100 --log-format "simple"  \
    --save-interval-updates 5000 --save-interval 100 --keep-interval-updates 5 --max-tokens 6000 \
    --arch my_transformer --save-dir $MODEL \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 0.1 --min-lr 1e-16 \
    --update-freq 1 --ddp-backend=no_c10d