javac SearchFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java SearchFiles -testPath $TRAIN/test.en -index $TRAIN/index -search source -topN 20
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java SearchFiles -testPath $TRAIN/dev.en -index $TRAIN/index -search source -topN 20
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java SearchFiles -testPath $TRAIN/train.en -index $TRAIN/index -search source -topN 20