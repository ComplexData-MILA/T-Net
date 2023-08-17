# T-Net
Code for T-Net for combatting Human Trafficking

## Installation
`pip install -r requirements.txt`

## Unzip files
Unzip the `data.zip` and `results/synthetic_asw.zip` folder for running rest of the code.


## Running T-Net
`python3 main.py --data_file data/synthetic_asw/synthetic_labelled_graph.pkl --epochs 100 --save_dir results/synthetic_asw --save_filename tnet_cl_results.pkl`

## Running baselines
`python3 main.py --data_file ht_datasets/synthetic_asw/synthetic_labelled_graph.pkl --epochs 100 --save_dir results/synthetic_asw --save_filename mlp_results.pkl --baseline --baseline_method mlp`

Choose a baseline method name from `mlp, gcn, nrgnn, pignn`. For `NRGNN` and `PIGNN` install their code from their official github repository to run them or use the saved model from our results folder

1. `NRGNN` - https://github.com/EnyanDai/NRGNN
2. `PIGNN` - https://github.com/TianBian95/pi-gnn


## For Misclassifciation results (Figure 2 in paper)
`python3 main.py --data_file ht_datasets/synthetic_asw/synthetic_labelled_graph.pkl --save_dir results/synthetic_asw --get_misclassification`

## For tabular results (Table 5 in paper)
`python3 main.py --save_dir results/synthetic_asw --print_results`


