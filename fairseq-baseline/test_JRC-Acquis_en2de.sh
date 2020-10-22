export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_DIR=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/baseline/train/en-de/model/en2de-base/
TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/baseline/train/en-de/data-bin/
/home/v-jiaya/anaconda3/bin/python /home/v-jiaya/RetrieveNMT/fairseq-baseline/generate.py $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 64 --beam 6 --source-lang en --target-lang de --remove-bpe --lenpen 0.5 --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt


