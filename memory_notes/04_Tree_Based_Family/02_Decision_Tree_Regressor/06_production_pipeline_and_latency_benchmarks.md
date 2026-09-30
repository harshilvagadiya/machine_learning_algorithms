# 🚀 Decision Tree Regressor: Production Latency

Evaluating a single tree of depth $d$ requires $d$ scalar comparisons:
$$\tau_{\text{inference}} < 0.02 \text{ ms}$$
Sub-millisecond inference guaranteed. Model can be converted to nested C `if-else` blocks for hardware FPGA/microcontroller deployment!
