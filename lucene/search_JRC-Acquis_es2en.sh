javac SearchFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-es/train/
#java SearchFiles -testPath $TRAIN/dev.es -index $TRAIN/index -search target -topN 20
#java SearchFiles -testPath $TRAIN/test.es -index $TRAIN/index -search target -topN 20
java SearchFiles -testPath $TRAIN/train.es -index $TRAIN/index -search target -topN 20