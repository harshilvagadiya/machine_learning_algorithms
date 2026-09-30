# 🧠 Neural Networks & Deep Learning Architectures — Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Neural Networks represent parametric function approximators formed by compositions of affine transformations and non-linear activation functions:
$$f(\mathbf{x}) = \sigma_L\left(\mathbf{W}_L \dots \sigma_1(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \dots + \mathbf{b}_L\right)$$

According to the **Cybenko Universal Approximation Theorem (1989)**, a feedforward network with a single hidden layer and non-linear activation can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision.

---

## 2. Classical Tabular vs Deep Learning Architectures

| Architecture Family | Primary Data Modality | Fundamental Mechanism | Core Computational Operation | Key Limitation / Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Perceptron (1958)** | Tabular linearly separable | Binary step function $\operatorname{sgn}(\mathbf{w}^T \mathbf{x} + b)$ | Outer product weight update | Cannot solve XOR (Minsky & Papert) |
| **MLP (Feedforward)** | Tabular / Dense vectors | Fully connected dense layers + ReLU/GELU | Matrix multiplication $\mathbf{W} \mathbf{x} + \mathbf{b}$ | High parameter count, loses spatial/temporal invariance |
| **CNN (ConvNet)** | Spatial grid (Images, Spectrograms) | Convolutional kernels with weight sharing | 2D/3D Cross-correlation $\mathbf{K} * \mathbf{X}$ | Translation-invariant, but fails on unordered sets |
| **RNN (Recurrent)** | Sequential time-series / NLP | Hidden state recurrence $\mathbf{h}_t = \tanh(\mathbf{W} \mathbf{x}_t + \mathbf{U} \mathbf{h}_{t-1})$ | Sequential temporal step | Exploding & vanishing gradients ($BPTT$) |
| **LSTM (Long Short-Term Memory)** | Long-range time-series / Audio | Cell state $C_t$ with Forget, Input, Output gates | Additive gradient highway $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$ | Sequential processing prevents massive parallelization |
| **GRU (Gated Recurrent Unit)** | Sequence forecasting | Merged state with Reset $r_t$ and Update $z_t$ gates | Interpolated state update $\mathbf{h}_t = (1-z_t)\mathbf{h}_{t-1} + z_t \tilde{\mathbf{h}}_t$ | Slightly lower expressivity than 3-gate LSTM |
| **Transformer (Attention)** | Sequences, Multimodal, Text | Scaled Dot-Product Self-Attention | $\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$ | Quadratic memory $\mathcal{O}(L^2)$ without flash attention |

---

## 3. Mathematical Formulations

### A. Convolutional Neural Networks (CNN)
For a 2D image $I$ and kernel $K$ of size $(2k+1) \times (2k+1)$:
$$S(i, j) = (I * K)(i, j) = \sum_{m=-k}^k \sum_{n=-k}^k I(i - m, j - n) K(m, n)$$
- **Receptive Field:** Grows linearly with network depth: $RF_{l} = RF_{l-1} + (k_l - 1) \cdot \prod_{i=1}^{l-1} s_i$.

### B. Long Short-Term Memory (LSTM - Hochreiter & Schmidhuber, 1997)
$$\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \quad &&\text{(Forget Gate: what to erase)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \quad &&\text{(Input Gate: what to write)} \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \quad &&\text{(Candidate Cell State)} \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad &&\text{(Cell State Highway: Constant Error Carousel)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \quad &&\text{(Output Gate: what to reveal)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad &&\text{(Hidden State)}
\end{aligned}$$

### C. Self-Attention & Transformers (Vaswani et al., 2017)
$$\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$$
$$\text{where} \quad \text{head}_i = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{W}_i^Q (\mathbf{K}\mathbf{W}_i^K)^T}{\sqrt{d_k}}\right)(\mathbf{V}\mathbf{W}_i^V)$$
- **Why Self-Attention displaced RNNs:** Constant path length $\mathcal{O}(1)$ between any two sequence positions vs $\mathcal{O}(L)$ in RNNs, enabling 100% parallel training over sequence lengths on modern tensor cores.

---

## 4. Production Engineering & Serving Latency
- **Classical Tabular (MLP / Perceptron):** $\le 0.4 \text{ ms}$ single-sample CPU inference.
- **CNN (ResNet-18 / MobileNet):** $\approx 1.5 - 4.0 \text{ ms}$ on GPU, $12 - 35 \text{ ms}$ on CPU with ONNX Runtime.
- **Transformer Encoder (DistilBERT / MiniLM):** $\approx 3.0 - 8.0 \text{ ms}$ on GPU, sub-50ms CPU with INT8 quantization.
