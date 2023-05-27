# nn-from-scratch

Basic implementation of simple neural network model (MNIST classifier)

To run model,
```bash
source build.sh
```

## Implementation detail
Typically, a single forward path would be denoted as $W\mathbf{x} + b$, $W \in \mathbb{R}^{10 \times (784)}$, $\mathbf{x} \in \mathbb{R}^{784}$, with bias $\mathbf{b} \in \mathbb{R}^{10}$ for $28 \times 28$ pixel data. However, here I sacrificed some memory for a simplified notation.

Let $W \in \mathbb{R}^{10 \times (785)}$ denote the weight matrix, $\mathbf{x} \in \mathbb{R}^{785}$, where the first element $x_0$ is always 1. The remaining 784 elements represent the grayscaled pixel data. That is, the first column of the $W$ matrix will act as bias.

$$
a = \text{softmax}(W\mathbf{x})
$$

$W$ is updated using stochastic gradient descent in usual manner to minimize the cross entropy loss.

The naive implementation only uses C++ Standard Library. `naive_blas` is a single-threaded, sequential implementation of basic Linear Algebra operations for this implementation. It is not optimized for performance and may be used for comparison with various acceleration options in the future. 

The identical python implementation using numpy can be found in `ref.py`.

