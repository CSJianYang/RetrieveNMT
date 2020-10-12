export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/v-jiaya/anaconda3/bin/python
GENERATE=/home/v-jiaya/RetrieveNMT/fairseq-baseline/generate.py
TEXT=/home/v-jiaya/RetrieveNMT/data/MD/en-de/concat/test-data-bin/
src=$1
tgt=$2
testset=$3
if [ "$src" == "en" ]; then
    MODEL_DIR=/home/v-jiaya/RetrieveNMT/data/MD/en-de/concat/model/en2de-base1/
elif [ "$src" == "de" ]; then
    MODEL_DIR=/home/v-jiaya/RetrieveNMT/data/MD/en-de/concat/model/de2en-base1/
else
    echo "Error Language !"
    exit
fi
   
if [ "$testset" == "JRC" ]; then
    gen_subset=test0
    lenpen=1.0
elif [ "$testset" == "newstest2014" ]; then
    gen_subset=test1
    lenpen=0
elif [ "$testset" == "Opensubtitles" ]; then
    gen_subset=test2
    lenpen=1.0
elif [ "$testset" == "tst2014" ]; then
    gen_subset=test3
    lenpen=1.0
else
    echo "Error testset !"
    exit
fi    
    
$PYTHON $GENERATE $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 64 --beam 8 --gen-subset $gen_subset --source-lang $src --target-lang $tgt --remove-bpe --lenpen $lenpen --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt