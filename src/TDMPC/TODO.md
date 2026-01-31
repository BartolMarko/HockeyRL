# TD-MPC TODO

### Definitely should be implemented:

- action repeat
- iCEM
- Pink noise exploration
- Opponent adaptation
    - since episode start statistics
    - action prediction

### Maybe implement:

- Parallelization of evaluation (run evaluation episodes in background while training)
- Time measuring
- RNN or some fancier opponent modelling

## DONE:
- implement opponent evaluation statistics based on first puck possession
- episode mirroring for training speedup and data efficiency
- TDMPC2 fancy network symlog smth - fixed exploding losses
- opponent pool with removable opponents
- logging episode statistics filtered by first puck possession

## NOT NEEDED:
- CrossQ paper: Batch normalization

