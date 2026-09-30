# ⚖️ The Zero-Frequency Trap & Laplace Smoothing

If a word $w$ appears in the test set that was never observed in training documents of class $c$, $N_{wc} = 0$:
$$p_{wc} = 0 \implies P(X \mid Y = c) = \prod_{j} p_{jc} = 0$$
A single unseen word completely zeroes out the entire class probability!

### Laplace Smoothing (Additive Smoothing)
Setting $\alpha = 1.0$ (Laplace) or $\alpha < 1.0$ (Lidstone) adds a pseudo-count to every term, guaranteeing $p_{jc} > 0$:
$$p_{jc} = \frac{N_{jc} + \alpha}{N_c + \alpha d}$$
