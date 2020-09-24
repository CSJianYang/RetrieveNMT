PYTHON=/home/v-jiaya/anaconda3/bin/python
CONCAT_SOURCE_RETRIVE=/home/v-jiaya/RetriveNMT/RetriveNMT/concat_source_retrive.py
#en-de
#en2de

#src=en
#tgt=de
#SOURCE=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
#DEST=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/train/
#mkdir $DEST
#for splt in test dev train; do
#    $PYTHON $CONCAT_SOURCE_RETRIVE -src-set  $SOURCE/$splt.$src -retrive-set $SOURCE/$splt.$src.retrive.20 -dest-set $DEST/$splt.$src
#done
#cp $SOURCE/*.$tgt $DEST/

#en-es
#en2es
src=en
tgt=es
lg=en-es
SOURCE=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/$lg/train/
DEST=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/$lg/${src}2${tgt}/top2/train/
mkdir -p $DEST
for splt in test dev train; do
    $PYTHON $CONCAT_SOURCE_RETRIVE -src-set  $SOURCE/$splt.$src -retrive-set $SOURCE/$splt.$src.retrive.20 -dest-set $DEST/$splt.$src
done
cp $SOURCE/*.$tgt $DEST/

#es2en
src=es
tgt=en
lg=en-es
SOURCE=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/$lg/train/
DEST=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/$lg/${src}2${tgt}/top2/train/
mkdir -p $DEST
for splt in test dev train; do
    $PYTHON $CONCAT_SOURCE_RETRIVE -src-set  $SOURCE/$splt.$src -retrive-set $SOURCE/$splt.$src.retrive.20 -dest-set $DEST/$splt.$src
done
cp $SOURCE/*.$tgt $DEST/

#en-fr
#en2fr
src=en
tgt=fr
lg=en-fr
SOURCE=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/$lg/train/
DEST=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/$lg/${src}2${tgt}/top2/train/
mkdir -p $DEST
for splt in test dev train; do
    $PYTHON $CONCAT_SOURCE_RETRIVE -src-set  $SOURCE/$splt.$src -retrive-set $SOURCE/$splt.$src.retrive.20 -dest-set $DEST/$splt.$src
done
cp $SOURCE/*.$tgt $DEST/

#fr2en
src=fr
tgt=en
lg=en-fr
SOURCE=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/$lg/train/
DEST=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/$lg/${src}2${tgt}/top2/train/
mkdir -p $DEST
for splt in test dev train; do
    $PYTHON $CONCAT_SOURCE_RETRIVE -src-set  $SOURCE/$splt.$src -retrive-set $SOURCE/$splt.$src.retrive.20 -dest-set $DEST/$splt.$src
done
cp $SOURCE/*.$tgt $DEST/