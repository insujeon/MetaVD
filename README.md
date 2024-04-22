# Federated Learning via Meta-Variational Dropout (Official Code)

This repository contains the official PyTorch code for the paper: [Federated Learning via Meta-Variational Dropout published in NeurIPS 2023](https://openreview.net/forum?id=VNyKBipt91).


## Requirement
- Python >= 3.7.4
- CUDA >= 10.0 supported GPU
- Anaconda

## :rocket: Getting Started

**Setup Environment**

   ```bash
   pip install -r environment.yml
   conda activate metavd
   ```

## :bar_chart: Training and Testing

### Running Experiments


  ```bash
  python main.py --model <model-name> --dataset <dataset-name> <other-options>
  ```


**EX) Run Cifar10 Experiment with MetaVD**

  ```bash
  python main.py --model nvdpgaus --dataset cifar10
  ```

**EX) Run Cifar100 Experiment with MetaVD and heterogenity level of $\alpha = 5.0$**

  ```bash
  python main.py --model nvdpgaus --dataset cifar100 --alpha 5.0
  ```

## :gear: Options
We currently support various models and datasets options.

### Supported Models
| Model Name | Flag | Description |
| --- | --- | --- |
| [FedAvg](https://arxiv.org/abs/1602.05629) | `fedavg` | Federated Averaging |
| [FedAvg + Finetuning](https://arxiv.org/abs/1602.05629) | `fedavgper` | Personalized Federated Learning |
| [FedAvg + MetaVD](https://openreview.net/forum?id=VNyKBipt91) | `fedavgnvdpgausq` | Federated Averaging with MetaVD (proposed in this work) |
| [FedAvg + SNIP](https://arxiv.org/abs/1810.02340) | `fedavgsnip` | Federated Averaging with SNIP |
| [FedProx](https://arxiv.org/abs/1812.06127) | `fedprox` | Federated Proximal Optimization |
| [FedBE](https://arxiv.org/abs/2009.01974) | `fedbe` | Federated Learning with Bayesian Ensemble |
| [Reptile](https://arxiv.org/abs/1909.12488) | `reptile` | Federated Learning with Reptile |
| [Reptile + VD](https://arxiv.org/abs/1506.02557) | `vdgausq` | Reptile with VD |
| [Reptile + EnsembleVD](https://openreview.net/forum?id=BkeAf2CqY7) | `vdgausemq` | Reptile with EnsembleVD |
| [Reptile + MetaVD](https://openreview.net/forum?id=VNyKBipt91) | `nvdpgausq` | Reptile with MetaVD (proposed in this work)  |
| [Reptile + SNIP](https://arxiv.org/abs/1810.02340) | `reptilesnip` | Reptile with SNIP |
| [MAML](https://arxiv.org/abs/1802.07876) | `maml` | Federated Learning with Model-Agnostic Meta-Learning |
| [MAML + MetaVD](https://openreview.net/forum?id=VNyKBipt91) | `mamlgausq` | MAML with MetaVD (proposed in this work)  |
| [MAML + SNIP](https://arxiv.org/abs/1810.02340) | `mamlsnip` | MAML with SNIP |
| [PerFedAvg](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) | `perfedavg` | HF-MAML with SNIP |
| [PerFedAvg + MetaVD](https://openreview.net/forum?id=VNyKBipt91) | `perfedavgnvdpgausq` | HF-MAML with MetaVD (proposed in this work)  |
| [PerFedAvg + SNIP](https://arxiv.org/abs/1810.02340) | `perfedavgsnip` | HF-MAML with SNIP |


### Supported Datasets
| Dataset Name | Flag | Description |
| --- | --- | --- |
| [Femnist](https://leaf.cmu.edu/) | `femnist` | Federated EMNIST dataset |
| [Celeba](https://leaf.cmu.edu/) | `celeba` | CelebA dataset |
| [MNIST](https://paperswithcode.com/dataset/mnist) | `mnist` | MNIST dataset |
| [Cifar10](https://github.com/KarhouTam/FL-bench/tree/master/data) | `cifar10` | CIFAR10 dataset |
| [Cifar100](https://github.com/KarhouTam/FL-bench/tree/master/data) | `cifar100` | CIFAR100 dataset |
| [EMNIST](https://paperswithcode.com/dataset/emnist) | `emnist` | Extended MNIST dataset |
| [FMNIST](https://github.com/KarhouTam/FL-bench/tree/master/data) | `fmnist` | Fashion MNIST dataset |


> Please see the arg parser in main.py file to enable other options.

### Other details

For all datasets, we set the number of rounds (`num_rounds`) to 1000 to ensure sufficient convergence following conventions. The batch size (`local_bs`) was set to 64, and local steps (`local_epochs`) was set to 5. Personalization was executed with a batch size (`adaptation_bs`) of 64 and a 1-step update. 

For all methods, we investigated the server learning rate and local SGD learning rate within identical
ranges. The server learning rate η was explored within the range of [0.6, 0.7, 0.8, 0.9, 1.0]. The local
SGD learning rate was investigated within the range of [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]. In MAML
and PerFedAvg, an additional client learning rate γ is required, for which we searched within the range of
[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]. For MetaVD, an additional KL divergence parameter
β is needed, and we sought its optimal value within the range of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
We follow the hyperparameter setting outlined in pFedGP, except for adjusting the batch size to 64 or 320 and
investigating the learning rate within the range of [0.03 to 0.1]. To ensure the reproducibility of the experiments,
we will release all code, including baselines, on our GitHub repository.




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
@article{jeon2024federated,
  title={Federated Learning via Meta-Variational Dropout},
  author={Jeon, Insu and Hong, Minui and Yun, Junhyeog and Kim, Gunhee},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## Reference
- [pFedHN](https://github.com/AvivSham/pFedHN)
- [FedBABU](https://github.com/jhoon-oh/FedBABU)
- [pFed-Bench](https://github.com/alibaba/FederatedScope)
- [FL-bench](https://github.com/KarhouTam/FL-bench)
- [Leaf](https://github.com/TalwalkarLab/leaf)
- [NVDP](https://github.com/insujeon/NVDPs)
