# MNIST Classification with Mutual Information Analysis

This repository contains code for training simple fully connected and convolutional neural networks on the MNIST dataset. Additionally, it provides tools for calculating mutual information using different methods to analyze the layers' information retention during training.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Simple MLP](#simple-mlp)
- [Simple CNN](#simple-cnn)
- [Mutual Information Calculation](#mutual-information-calculation)
- [References](#references)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aidinattar/info-bottleneck.git
    cd your-repo-name
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Simple MLP

To train a simple fully connected network on the MNIST dataset:

1. Navigate to the directory:
    ```bash
    cd info_bottleneck
    ```

2. Run the training script:
    ```bash
    python main.py --model mlp
    ```

### Simple CNN

To train a simple convolutional neural network on the MNIST dataset:

1. Navigate to the directory:
    ```bash
    cd info_bottleneck
    ```

2. Run the training script:
    ```bash
    python main.py --model cnn
    ```

## Mutual Information Calculation

The mutual information is calculated using different methods (`binning`, `kde`, `kraskov`). The mutual information calculation is integrated within the `NetworkTrainer` class and can be specified during initialization.

### Example Usage

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# Initialize the trainer and train the model
trainer = SimpleCNNTrainer(train_loader, val_loader, activation='relu', optimizer=optim.Adam, epochs=10, device='cuda')
trainer.train()
```

## References

This code is based on the Information Bottleneck principle for neural networks. For more information, refer to the following resources:
- [On the Information Bottleneck Theory of Deep Learning](https://openreview.net/pdf?id=ry_WPG-A-)
- [Information Bottleneck for Deep Learning](https://arxiv.org/abs/1612.00410)
- [Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062)
- [Information Bottleneck](https://en.wikipedia.org/wiki/Information_bottleneck_method)

## Future Work

The current implementation focuses on fully connected and convolutional neural networks. Future work could include:
- Extending the mutual information analysis to other network architectures such as recurrent neural networks (RNNs) or transformers.
- Applying the mutual information analysis to more complex datasets beyond MNIST.
- Investigating the impact of different mutual information estimation methods on the performance and interpretability of neural networks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the authors of the referenced papers and the open-source community for providing valuable resources and tools that made this project possible.





