from pathlib import Path
from typing import Optional
import numpy as np
import torch

from src.named_agent import NamedAgent
from src.TDMPC.tdmpc import TDMPC


class TDMPCAgent(NamedAgent):
    def __init__(
        self,
        load_dir: Optional[Path],
        tdmpc: Optional[TDMPC],
        step: int,
        eval_mode: bool = True,
        name_suffix: str = "",
    ) -> None:
        super().__init__(name=f"TDMPCAgent{name_suffix}")
        if load_dir is None and tdmpc is None:
            raise ValueError("Either load_dir or tdmpc must be provided.")
        if load_dir is not None and tdmpc is not None:
            raise ValueError("Only one of load_dir or tdmpc should be provided.")

        self.t0 = False
        self.tdmpc = tdmpc
        self.eval_mode = eval_mode
        if tdmpc is None:
            self.tdmpc = TDMPC(cfg=None, load_dir=load_dir)
        self.step = step if step is not None else self.tdmpc.cfg.train_steps

    def on_start_game(self, game_id):
        self.t0 = True

    def get_step(self, obs: np.ndarray) -> np.ndarray:
        action = (
            self.tdmpc.plan(obs, eval_mode=self.eval_mode, step=self.step, t0=self.t0)
            .cpu()
            .numpy()
        )
        self.t0 = False
        return action

    def QValues(self, obs: np.ndarray, action: np.ndarray) -> list[float]:
        obs = torch.from_numpy(obs).float().to(self.tdmpc.device).unsqueeze(0)
        action = torch.from_numpy(action).float().to(self.tdmpc.device).unsqueeze(0)
        with torch.no_grad():
            z = self.tdmpc.model.encode(obs)
            Q_1, Q_2 = self.tdmpc.model.QValues(z, action)
        return [float(Q_1.item()), float(Q_2.item())]
