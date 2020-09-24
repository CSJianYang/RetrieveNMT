export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/v-jiaya/anaconda3/bin/python
GENERATE=/home/v-jiaya/RetriveNMT/RetriveNMT/generate.py
MODEL_DIR=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/model/en2de-reset-selectlayer/
TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/data-bin/
$PYTHON $GENERATE $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 64 --beam 8 --source-lang en --target-lang de --remove-bpe --lenpen 0.5 --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt


