export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/v-jiaya/anaconda3/bin/python
GENERATE=/home/v-jiaya/RetrieveNMT/CSTM/generate.py
src=$1
tgt=$2
testset=$3
if [ "$src" == "en" -a "$tgt" == "de" ]; then
    MODEL_DIR=/home/v-jiaya/RetrieveNMT/data/MD/retrieve-en2de/concat/model/en2de-CSTM1/
    TEXT=/home/v-jiaya/RetrieveNMT/data/MD/retrieve-en2de/concat/test-data-bin/
elif [ "$src" == "de" -a "$tgt" == "en" ]; then
    MODEL_DIR=/home/v-jiaya/RetrieveNMT/data/MD/retrieve-de2en/concat/model/de2en-CSTM1/
    TEXT=/home/v-jiaya/RetrieveNMT/data/MD/retrieve-de2en/concat/test-data-bin/
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
    
$PYTHON $GENERATE $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 32 --beam 8 --gen-subset $gen_subset --source-lang $src --target-lang $tgt --remove-bpe --lenpen $lenpen --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt