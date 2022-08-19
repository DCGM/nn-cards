
SCRIPT=./train_layout.py
DATA=./data/scriptOutput.csv 

for gcn_layers in 1 2 4
do
for gcn_repetitions in 1 2 4
do
for layer_type in GatedGraphConv GraphConv
do
for activation in None ReLU
do

python -u $SCRIPT --name test -d $DATA \
    --batch-size 24 --learning-rate 0.0002 --max-iteration 50000 \
    --net-config "{\"type\":\"gcn\", \"gcn_layers\":${gcn_layers}, \"gcn_repetitions\":${gcn_repetitions}, \"hidden_dim\":64, \"layer_type\":\"${layer_type}\", \"activation\":\"${activation}\"}" \
    | tee -a log_GCN_${layer_type}_${activation}_d64_l${gcn_layers}_r${gcn_repetitions}.log

done
done
done
done

exit
for l in $(seq 1 8)
do
python -u $SCRIPT --name test -d $DATA \
    --batch-size 16 --learning-rate 0.0002 --max-iteration 50000 \
    --net-config "{\"type\":\"mlp\", \"depth\":${l}}" | tee -a log_MLP_d64_l${l}.log
done


