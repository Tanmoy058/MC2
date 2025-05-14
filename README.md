# MC2: Mitigating KV Cache Competition Implementation Guide

This guide explains how to implement and use **MC2 (Mitigating KV Cache Competition)** to enhance user experience in LLM serving by reducing tail latency for both **Time-to-First-Token (TTFT)** and **Time-Between-Tokens (TBT)**.

## Overview

MC2 is designed to solve the KV-cache bottleneck in LLM serving systems, which can significantly impact user experience in time-sensitive applications. The system consists of four key components:

- **Confidence-based Padding**: Predicts output length with statistical bounds and adjusts padding based on arrival rate.
- **SLO-aware Batching and KVC Allocation**: Efficiently allocates KV cache with focus on meeting SLOs.
- **Preemption Policy**: Intelligently selects requests for preemption based on SLOs, remaining time, and KVC usage.
- **Preemption Strategy Selection**: Dynamically chooses between swapping and recomputation based on sequence length.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- vLLM codebase
- Matplotlib, NumPy, and Pandas (for evaluation)

### Installation

Download the MC2 implementation code and install required dependencies:

```bash
pip install torch numpy pandas matplotlib tqdm
If integrating with vLLM, make sure vLLM is properly installed:
```

```bash
pip install vllm
Using MC2
Standalone Example
To run a basic example of MC2 in action:
```

```bash
python mc2_implementation.py --mode example
This will simulate a basic LLM serving scenario using MC2.
```

Integration with vLLM
To patch vLLM with MC2 features:

```bash
python mc2_implementation.py --vllm_path /path/to/vllm
```
This will patch the vLLM codebase with MC2 components.


### Performance Evaluation
To evaluate MC2's performance against baseline methods:

```bash
python mc2_implementation.py --mode evaluate --output_dir ./evaluation_results
```
This will run performance tests comparing MC2 with vLLM, RLP, S3, and Sarathi across multiple datasets and arrival rates.

Ablation Study
To analyze the contribution of individual MC2 components:

```bash
python mc2_implementation.py --mode ablation --output_dir ./ablation_results
```
This will generate reports showing how much each component contributes to overall performance.

Sensitivity Analysis
To analyze MC2's sensitivity to different parameter values:

```bash
python mc2_implementation.py --mode sensitivity --output_dir ./sensitivity_results
```
This will test various parameter configurations and identify optimal values.

### Implementation Details
Core Components
1. Confidence-based Padding
The implementation extends the base LLM model with additional layers for predicting output length and deviation direction:


```python
class OutputLengthPredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.config.hidden_size

        # Length predictor
        self.length_predictor = nn.Linear(hidden_size, 1)

        # Deviation classifier
        self.deviation_classifier = nn.Linear(hidden_size + 1, 2)
```
Padding is calculated using Hoeffding's inequality to bound deviations with a specified confidence:

```python
def calculate_padding(self, predicted_length, deviation_direction, arrival_rate):
    confidence = self.alpha / (1 + self.beta * arrival_rate)
    range_value = max_range - min_range
    t = math.sqrt(-((range_value ** 2) / 2) * math.log(1 - confidence))

    if deviation_direction == 0:  # Underprediction
        return int(t)
    else:  # Overprediction
        return -int(t)
```

2. SLO-aware Batching and KVC Allocation
The implementation includes an embedding method to reuse allocated but unused KV cache:

```python
def embed_request(self, source_req_id, target_req_id, buffer_size=8):
    # Check if there's enough space with buffer
    if target_allocated - target_used - source_req_size - buffer_size >= 0:
        # Embed the request
        return True
    return False
```

3. Preemption Policy
The preemption policy considers SLO, remaining time, and KVC occupancy:

```python
def select_request_for_preemption(self):
    # Step 1: Order by TBT SLO (descending)
    candidates = sorted(self.running_requests, key=lambda r: r.tbt_slo, reverse=True)

    # Step 2: Find requests with similar TBT SLOs and order by remaining time
    # Step 3: Find requests with similar remaining time and order by KVC occupancy
    ...
```

4. Preemption Strategy Selection
The implementation dynamically selects between swapping and recomputation based on sequence length:

```python
def select_preemption_strategy(self, request):
    sequence_length = request.completion_tokens

    if sequence_length > self.sweet_spot_sequence_length:
        return "swap"
    else:
        return "recompute"
```

Evaluation Metrics
Tail TTFT: 95th percentile Time-to-First-Token latency.

TTFT SLO Attainment: Fraction of requests meeting their TTFT SLOs.

Tail TBT: 95th percentile Time-Between-Tokens latency.

TBT SLO Attainment: Fraction of requests meeting their TBT SLOs.

Normalized Latency: Average latency per token.



### Customization

You can modify MC2's parameters to suit different environments:

| Parameter               | Description                                         | Default |
|-------------------------|-----------------------------------------------------|---------|
| `block_size`             | Size of KV cache blocks                             | 8       |
| `reserved_blocks`        | Number of globally reserved blocks                  | 8       |
| `alpha`                  | Parameter for confidence calculation                | 0.95    |
| `beta`                   | Parameter for confidence sensitivity to arrival rate| 0.001   |
| `preallocate_iterations` | Number of iterations to preallocate KVC             | 2       |


### References
This implementation is based on the research paper:

"Mitigating KV Cache Competition to Enhance User Experience in LLM Serving"
