# Federated Learning Assignment - Tasks 2 & 3

Implementation of FedAvg and heterogeneity experiments for EE5102/CS6302.

## Quick Start (Kaggle P100)

### Setup
```bash
# Clone the repo
!git clone https://github.com/YOUR_USERNAME/federated-learning-assignment.git
%cd federated-learning-assignment

# Install dependencies
!pip install -q torch torchvision numpy matplotlib

# Create directories
!mkdir -p logs plots data
```

### Task 2: FedAvg Communication Efficiency

**Experiment 1: Varying Local Epochs (K)**
```bash
!python task2_fedavg.py --experiment vary_k --k 1 --run_id k1
!python task2_fedavg.py --experiment vary_k --k 5 --run_id k5
!python task2_fedavg.py --experiment vary_k --k 10 --run_id k10
!python task2_fedavg.py --experiment vary_k --k 20 --run_id k20
```

**Experiment 2: Varying Client Sampling**
```bash
!python task2_fedavg.py --experiment vary_sampling --sampling_frac 1.0 --run_id samp100
!python task2_fedavg.py --experiment vary_sampling --sampling_frac 0.5 --run_id samp50
!python task2_fedavg.py --experiment vary_sampling --sampling_frac 0.2 --run_id samp20
```

### Task 3: Data Heterogeneity Impact
```bash
!python task3_heterogeneity.py --alpha 100 --run_id iid
!python task3_heterogeneity.py --alpha 1.0 --run_id moderate
!python task3_heterogeneity.py --alpha 0.1 --run_id high
!python task3_heterogeneity.py --alpha 0.05 --run_id extreme
```

## Parallel Execution

✅ **Tasks 2 and 3 are INDEPENDENT** - Run them on different Kaggle notebooks simultaneously!

## Output Files

Results saved to `logs/`:
- Task 2: `task2_{experiment}_{run_id}.json`
- Task 3: `task3_alpha{alpha}_{run_id}.json`

## Configuration

Default settings in `config.py`:
- Clients: 5
- Rounds: 50
- Batch size: 64 (adjust based on GPU)
- Seed: 42

## Notes

- All experiments use CIFAR-10
- Batch size can be adjusted in `config.py` if memory issues occur
- Logs include per-round accuracy, loss, training time, and weight divergence
