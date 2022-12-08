# DMVB: Deep Reinforcement Learning via Multi-View Bisimulation

Pytorch implementation of DMVB based on DRIBO and DBC code bases.

## Usage
Prepare conda environment:
```bash
conda env create -f dmvb.yml
```

Train the agent:
```bash
python dmvb.py --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk/dmvc/1
```

