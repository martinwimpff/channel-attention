# EEG motor imagery decoding: A framework for comparative analysis with channel attention mechanisms
This is the official repository to the paper [EEG motor imagery decoding: A framework for comparative analysis with channel attention mechanisms](https://arxiv.org/abs/2310.11198).
The model is also integrated in [braindecode](https://github.com/braindecode/braindecode/blob/master/braindecode/models/attentionbasenet.py).

## Usage
### Data
All data will be downloaded automatically except for the BCIC III dataset.
Download the BCIC III dataset and put all files in the directory defined in [load_bcic3](channel_attention/utils/load_bcic3.py).
### Installation
- clone this repository
- run `pip install .` to install the `channel-attention` package

_Note: you can also use poetry for the installation_
### Model training
- run [run.py](channel_attention/run.py) with the `--config` of your choice

## Citation
If you find this repository useful, please cite our work
```
@article{wimpff2023eeg,
  title={EEG motor imagery decoding: A framework for comparative analysis with channel attention mechanisms},
  author={Wimpff, Martin and Gizzi, Leonardo and Zerfowski, Jan and Yang, Bin},
  journal={arXiv preprint arXiv:2310.11198},
  year={2023}
}
```
