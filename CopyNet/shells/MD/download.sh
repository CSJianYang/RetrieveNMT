sudo bash philly-fs.bash -rm -r //philly/rr1/resrchvc/t-jianya/RetrieveNMT/data/MD/retrieve-en2de-top2/concat/model/en2de-CopyNet1/
sudo bash philly-fs.bash -rm -r //philly/rr1/resrchvc/t-jianya/RetrieveNMT/data/MD/retrieve-de2en-top2/concat/model/de2en-CopyNet1/

bash philly-fs.bash -cp -r //philly/rr1/resrchvc/t-jianya/RetrieveNMT/data/MD/retrieve-en2de-top2/concat/model/en2de-CopyNet1/checkpoint_best.pt /home/v-jiaya/RetrieveNMT/data/MD/retrieve-en2de-top2/concat/model/en2de-CopyNet1/
bash philly-fs.bash -cp -r //philly/rr1/resrchvc/t-jianya/RetrieveNMT/data/MD/retrieve-de2en-top2/concat/model/de2en-CopyNet1/checkpoint_best.pt /home/v-jiaya/RetrieveNMT/data/MD/retrieve-de2en-top2/concat/model/de2en-CopyNet1/
