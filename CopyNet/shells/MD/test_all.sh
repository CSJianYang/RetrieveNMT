TEST_SHELL=/home/v-jiaya/RetrieveNMT/CopyNet/shells/MD/test.sh
echo "translate en->de"
src=en
tgt=de
for testset in JRC newstest2014 Opensubtitles tst2014; do
    bash $TEST_SHELL $src $tgt $testset
done

echo "translate de->en"
src=de
tgt=en
for testset in JRC newstest2014 Opensubtitles tst2014; do
    bash $TEST_SHELL $src $tgt $testset
done