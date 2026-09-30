# ⚡ Passive-Aggressive Algorithms (PAClassifier & PARegressor) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
The **Passive-Aggressive (PA)** family (Crammer et al., 2006) represents the gold standard in **first-order online learning and streaming machine learning**.
Unlike batch gradient descent which loops over historical datasets, Passive-Aggressive operates sequentially on incoming streaming instances $(\mathbf{x}_t, y_t)$:
- **Passive:** If the incoming instance satisfies the margin constraint (loss $\ell_t = 0$), the weight vector is left untouched: $\mathbf{w}_{t+1} = \mathbf{w}_t$.
- **Aggressive:** If the instance violates the margin ($\ell_t > 0$), the algorithm projects $\mathbf{w}_t$ minimally into the feasible halfspace so that the current instance achieves zero loss, while retaining maximum memory of past instances.

---

## 2. Mathematical Optimization Problem

At step $t$, the Passive-Aggressive update solves the constrained convex optimization problem:
$$\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w} - \mathbf{w}_t\|_2^2 + C \xi_t \quad \text{subject to } \ell(\mathbf{w}; (\mathbf{x}_t, y_t)) \le \xi_t, \quad \xi_t \ge 0$$

### The Hinge Loss Formulation:
For classification ($y_t \in \{-1, +1\}$):
$$\ell_t = \max(0, 1 - y_t (\mathbf{w}_t^T \mathbf{x}_t))$$

For $\epsilon$-insensitive regression:
$$\ell_t = \max(0, |y_t - \mathbf{w}_t^T \mathbf{x}_t| - \epsilon)$$

### Exact Analytical Closed-Form Updates:
Rather than taking a heuristic step with an empirical learning rate $\eta$, PA has an **exact closed-form Lagrange multiplier solution**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \tau_t y_t \mathbf{x}_t$$

| Variant | Step Size $\tau_t$ Formula | Operational Behavior |
| :--- | :--- | :--- |
| **PA (Hard Margin)** | $\tau_t = \frac{\ell_t}{\|\mathbf{x}_t\|_2^2}$ | Enforces exact zero loss immediately (Noise-sensitive) |
| **PA-I (Linear Slack)** | $\tau_t = \min\left(C, \frac{\ell_t}{\|\mathbf{x}_t\|_2^2}\right)$ | Bounds maximum single-step parameter movement by $C$ |
| **PA-II (Quadratic Slack)** | $\tau_t = \frac{\ell_t}{\|\mathbf{x}_t\|_2^2 + \frac{1}{2C}}$ | Smooth non-linear regularization damping |

---

## 3. Production Engineering & Serving Latency
- **No Learning Rate Decay:** In contrast to SGD (`SGDClassifier` / `SGDRegressor`) which requires careful tuning of learning schedules (`eta0`, `power_t`, `learning_rate='optimal'`), PA has **zero learning rate tuning**. The step size $\tau_t$ is computed analytically per sample.
- **Microsecond Latency SLA:**
  $$\text{Serving Latency} \le 0.05 \text{ ms (P99 < 0.15 ms)}$$
- **High-Velocity Streaming Pipelines:** Ideal for ad-click click-through-rate (CTR), network intrusion detection, algorithmic trading tick feeds, and real-time concept drift adaptation via `partial_fit`.
