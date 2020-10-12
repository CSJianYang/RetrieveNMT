CONCAT=/home/v-jiaya/RetrieveNMT/RetrieveNMT/concat_source_retrive.py
src=de
tgt=en
for dataset in wmt14-en-de iwslt14-en-de JRC-Acquis Opensubtitles; do
    echo "concat $dataset..."
    for split in test valid train; do
        OLD_DIR=/home/v-jiaya/RetrieveNMT/data/MD/en-de/$dataset/train/
        NEW_DIR=/home/v-jiaya/RetrieveNMT/data/MD/retrieve-de2en-top2/$dataset/train/
        mkdir -p $NEW_DIR
        src_set=$OLD_DIR/$split.$src
        tgt_set=$OLD_DIR/$split.$tgt
        retrieve_set=$OLD_DIR/$split.$src.retrive.20
        dest_set=$NEW_DIR/$split.$src
        new_tgt_set=$NEW_DIR/$split.$tgt
        python3 $CONCAT -src-set $src_set -tgt-set $tgt_set -retrieve-set $retrieve_set -dest-set $dest_set -new-tgt-set $new_tgt_set -topK 2
    done
done