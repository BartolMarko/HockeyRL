# TD-MPC TODO

### Definitely should be implemented:

- smart action seeding:
    - seed with paths to goal positions
    - seed with paths to future ball positions
- Pink noise exploration
- Opponent adaptation
    - since episode start statistics
    - action prediction

### Maybe implement:

- RNN or some fancier opponent modelling

## DONE:
- iCEM
- action repeat
- implement opponent evaluation statistics based on first puck possession
- episode mirroring for training speedup and data efficiency
- TDMPC2 fancy network symlog smth - fixed exploding losses
- opponent pool with removable opponents
- logging episode statistics filtered by first puck possession

## NOT NEEDED:
- CrossQ paper: Batch normalization

