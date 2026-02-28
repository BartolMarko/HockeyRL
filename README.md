# WayneGradientzky

## Requirements

All dependencies and environment specifications are defined in `container.def`.

## Models & Checkpoints

Pretrained model checkpoints are available in the `models/` directory.

Agents can be instantiated using the `agent_factory` utility located in `src/agent_factory`.

Examples:

```python
from src.agent_factory import agent_factory

td3 = agent_factory(
    'td3',
    {
        "type": "TD3",
        "weights_path": "path/to/weights",
        "config_path": "path/to/config"
    }
)

sac = agent_factory(
    'sac',
    {
        "type": "SAC",
        "config_path": "path/to/config",
        "weights_folder": "path/to/weights"
    }
)

tdmpc = agent_factory(
    "tdmpc",
    {
        "type": "TDMPC",
        "load_dir": "path/to/model"
    }
)
```

## Team Bot

The team bot used during the competition is implemented in:

`src/TD3/if_else_bot.py`