# Federated Learning via Meta-Variational Dropout (Official Code)

This is a predisclosure version of github repo. The full code will be updated soon before the conference.

This repository contains the official PyTorch code for the paper: [Federated Learning via Meta-Variational Dropout published in NeurIPS 2023](link-to-paper).

- Python >= 3.7.4
- CUDA >= 10.0 supported GPU

## :rocket: Getting Started

**Setup Environment**

   ```bash
   pip install -r environment.yml
   ```


## :bar_chart: Training and Testing

### Running Experiments

- **Cifar10 Experiment**

  ```bash
  python main.py --dataset cifar10 <other-options>
  ```

- **Cifar100 Experiment**

  ```bash
  python main.py --dataset cifar100 <other-options>
  ```


### Visualization

- **Tensorboard Setup**

  ```bash
  cd runs
  tensorboard --logdir=./ --port=7770 --samples_per_plugin image=100 --reload_multifile=True --reload_interval 30 --host=0.0.0.0
  ```

  Access visualizations at [localhost:7770](http://localhost:7770).



## :page_facing_up: Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{jeon2023,
  title={Federated LEarning via Meta-Variational Dropout},
  author={Jeon, Insu and Hong, Minui and Yun, Junhyeog and Kim, Gunhee},
  booktitle={NeurIPS},
  year={2023}
}
```

## Reference
- pFedHN
- FedBABU
- pFedBench
- Leaf
- NVDP
