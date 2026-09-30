# ⏳ Survival Analysis (Cox Proportional Hazards & AFT) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Traditional regression and classification fail on time-to-event datasets because of **right-censoring**:
- In clinical trials: A patient moves away or the study concludes after 5 years while the patient is still alive ($T > 5$ years).
- In customer retention: A customer is active today with tenure of 18 months ($T > 18$ months).

Treating censored individuals as non-events causes severe underestimation of event rates; discarding them induces fatal survivorship bias.
**Survival Analysis** explicitly handles censoring by modeling the duration distribution $T$ and the instant hazard rate $h(t)$.

---

## 2. Fundamental Survival Functions

1. **Survival Function $S(t)$:** Probability that an individual survives beyond time $t$:
   $$S(t) = P(T > t) = 1 - F(t) = \exp\left( -\int_0^t h(u) \, du \right)$$
2. **Hazard Function $h(t)$:** Instantaneous rate of event occurrence given survival up to time $t$:
   $$h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t} = \frac{f(t)}{S(t)} = -\frac{d}{dt} \ln S(t)$$
3. **Cumulative Hazard $H(t)$:**
   $$H(t) = \int_0^t h(u) \, du = -\ln S(t)$$

---

## 3. Semi-Parametric vs Parametric Models

| Modeling Dimension | Cox Proportional Hazards (Cox, 1972) | Accelerated Failure Time (AFT / Weibull) |
| :--- | :--- | :--- |
| **Model Type** | Semi-parametric (Baseline hazard $h_0(t)$ is unspecified) | Fully parametric (Weibull, Log-Normal, Log-Logistic) |
| **Hazard Formulation** | $h(t \mid \mathbf{x}) = h_0(t) \exp(\mathbf{x}^T \boldsymbol{\beta})$ | $\ln T = \mathbf{x}^T \boldsymbol{\beta} + \sigma W$ (Accelerates/decelerates time) |
| **Proportionality** | Covariates act multiplicatively on the hazard rate | Covariates act multiplicatively on survival time directly |
| **Estimation** | Maximizes Partial Likelihood (Independent of $h_0(t)$) | Maximizes Full Parametric Likelihood via Newton-Raphson |
| **Extrapolation** | Cannot extrapolate beyond maximum observed study time | Can extrapolate expected survival times into the infinite future |

### The Cox Partial Likelihood:
$$L(\boldsymbol{\beta}) = \prod_{i=1}^E \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta})}{\sum_{j \in \mathcal{R}(t_i)} \exp(\mathbf{x}_j^T \boldsymbol{\beta})}$$
where $\mathcal{R}(t_i)$ is the risk set of individuals who remain under observation and event-free immediately prior to event time $t_i$.

---

## 4. Evaluation: Harrell's Concordance Index (C-Index)
$$\text{C-index} = \frac{\sum_{i, j} \mathbb{I}(T_i < T_j) \cdot \mathbb{I}(\hat{h}_i > \hat{h}_j) \cdot E_i}{\sum_{i, j} \mathbb{I}(T_i < T_j) \cdot E_i}$$
- $\text{C-index} = 0.50$: Random guessing.
- $\text{C-index} = 1.00$: Perfect risk discrimination.
- Industry benchmark in clinical and churn modeling: $0.68 - 0.92$.

---

## 5. Production Engineering & Serving Latency
- **Real-Time Risk Scoring:** Once $\boldsymbol{\beta}$ is fitted, computing the partial hazard risk score for a customer is a single dot product:
  $$\text{Risk Score} = \exp(\mathbf{x}_*^T \boldsymbol{\beta}) \implies \text{Serving Latency} \le 0.15 \text{ ms}$$
- **Full Curve Materialization:** Generating full dynamic survival curves $S(t \mid \mathbf{x}_*)$ across all discrete timeline intervals takes $< 2.0 \text{ ms}$.
