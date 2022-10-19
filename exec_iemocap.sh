#!bin/bash
# Var assignment
LR=5e-4 # 2.5e-4 for bert embeddings
GPU='0'
TP=MVN
du=150
dc=100
w1=40
dataset="IEMOCAP6"
data_path="Data/IEMOCAP6/IEMOCAP6_data.pt"
vocab_path="Data/IEMOCAP6/IEMOCAP6_vocab.pt"
emodict_path="Data/IEMOCAP6/IEMOCAP6_emodict.pt"
tr_emodict_path="Data/IEMOCAP6/IEMOCAP6_tr_emodict.pt"
embedding="Data/IEMOCAP6/IEMOCAP6_embedding.pt"
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $Enc - $Dec $iter ---
python -u EmoMain.py -epochs 60 -lr $LR -gpu $GPU -type $TP -wind1 $w1 -d_h1 $du -d_h2 $dc -report_loss 96 -dataset $dataset -data_path $data_path -vocab_path $vocab_path -emodict_path $emodict_path -tr_emodict_path $tr_emodict_path -embedding $embedding
done > iemocap_mvn.txt 2>&1 &
