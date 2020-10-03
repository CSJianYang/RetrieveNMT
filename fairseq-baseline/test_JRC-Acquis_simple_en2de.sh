export CUDA_VISIBLE_DEVICES=1
MODEL_DIR=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/model/de2en-simple-update1/
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/data-bin/
/home/v-jiaya/anaconda3/bin/python /home/v-jiaya/pivot/fairseq-baseline/generate.py $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 64 --beam 8 --source-lang en --target-lang de --remove-bpe --lenpen 0.5 --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt


