export CUDA_VISIBLE_DEVICES=2
PYTHON=/home/v-jiaya/anaconda3/bin/python
GENERATE=/home/v-jiaya/RetrieveNMT/RetrieveNMT/generate.py
src=$1
tgt=$2
if [ "$src" == "en" -a "$tgt" == "de" ]; then
    MODEL=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/model/en2de-addseg/checkpoint_best.pt
    TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/data-bin/
    lenpen=1.3
elif [ "$src" == "de" -a "$tgt" == "en" ]; then
    MODEL=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/model/de2en-addseg/checkpoint_best.pt
    TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/
    lenpen=1.0
elif [ "$src" == "en" -a "$tgt" == "fr" ]; then
    MODEL=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-fr/en2fr/top2/model/en2fr-addseg/checkpoint_best.pt
    TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-fr/en2fr/top2/data-bin/
    lenpen=1.0
elif [ "$src" == "fr" -a "$tgt" == "en" ]; then
    MODEL=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-fr/fr2en/top2/model/fr2en-addseg/checkpoint_best.pt
    TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-fr/fr2en/top2/data-bin/
    lenpen=1.0
elif [ "$src" == "en" -a "$tgt" == "es" ]; then
    MODEL=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-es/en2es/top2/model/en2es-addseg/checkpoint_best.pt
    TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-es/en2es/top2/data-bin/
    lenpen=1.5
elif [ "$src" == "es" -a "$tgt" == "en" ]; then
    MODEL=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-es/es2en/top2/model/es2en-addseg/checkpoint_best.pt
    TEXT=/home/v-jiaya/RetrieveNMT/data/JRC-Acquis/RetriveNMT/en-es/es2en/top2/data-bin/
    lenpen=1.0
else
    echo "Error Language !"
    exit
fi

$PYTHON $GENERATE $TEXT/ --path $MODEL --batch-size 32 --beam 8 --source-lang $src --target-lang $tgt --remove-bpe --lenpen $lenpen --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt