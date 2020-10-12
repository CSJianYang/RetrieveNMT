javac SearchFiles.java
TRAIN=/home/v-jiaya/RetrieveNMT/data/MD/en-de/wmt14-en-de/train/
lang=$1
field=$2
java SearchFiles -testPath $TRAIN/test.$lang -index $TRAIN/index -search $field -topN 20
java SearchFiles -testPath $TRAIN/valid.$lang -index $TRAIN/index -search $field -topN 20
java SearchFiles -testPath $TRAIN/train.$lang -index $TRAIN/index -search $field -topN 20