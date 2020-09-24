javac SearchFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-fr/train/
java SearchFiles -testPath $TRAIN/dev.en -index $TRAIN/index -search source -topN 20
java SearchFiles -testPath $TRAIN/test.en -index $TRAIN/index -search source -topN 20
java SearchFiles -testPath $TRAIN/train.en -index $TRAIN/index -search source -topN 20