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

1. `NRGNN` - (https://github.com/EnyanDai/NRGNN)
2. `PIGNN` - (https://github.com/TianBian95/pi-gnn)


## For Misclassifciation results (Figure 2 in paper)
`python3 main.py --data_file ht_datasets/synthetic_asw/synthetic_labelled_graph.pkl --save_dir results/synthetic_asw --get_misclassification`

## For tabular results (Table 5 in paper)
`python3 main.py --save_dir results/synthetic_asw --print_results`

## For accessing ASW-Synth
If you want to get access to the synthetically generated dataset, send an email with a short description of why you need the data to pratheeksha.nair@mail.mcgill.ca 

## Labeling Functions
The labeling functions used in the paper are specified in `labeling_functions.py` and the code for obtaining weak labels are also included. The code for building the graph from the ads is in `build_graph.py`

## Cite
Please consider citing our work if you find it useful,

```
@inproceedings{nair2024t,
  title={T-NET: Weakly Supervised Graph Learning for Combatting Human Trafficking},
  author={Nair, Pratheeksha and Liu, Javin and Vajiac, Catalina and Olligschlaeger, Andreas and Chau, Duen Horng and Cazzolato, Mirela and Jones, Cara and Faloutsos,    Christos and Rabbany, Reihaneh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={20},
  pages={22276--22284},
  year={2024}
}
```
