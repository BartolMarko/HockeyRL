# TD-MPC TODO

### Definitely should be implemented:

- action repeat
- self play frequency adaptation
- iCEM
- Pink noise exploration
- Opponent adaptation
    - since episode start statistics
    - action prediction

### Maybe implement:

- MPPI planning as in TDMPC2
- Parallelization of evaluation
- Parallelization of training
- Time measur
- RNN or some fancier opponent modelling

### Explore what is and decide:

- CrossQ paper: Batch normalization
- TDMPC2 fancy network symlog smth

## DONE:
- implement opponent evaluation statistics based on first puck possession
- episode mirroring for training speedup and data efficiency
