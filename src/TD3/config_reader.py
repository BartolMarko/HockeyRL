from __future__ import annotations

import yaml
from typing import Union

class Config:
    def __init__(self, source : Union[dict, str]):
        if type(source) is dict:
            self.cfg = source
        else:
            with open(source) as f:
                self.cfg = yaml.safe_load(f)
        
        self._build_recursive()
    
    def __getitem__(self, item):
        return self.cfg[item]
    
    def __setitem__(self, item, val):
        self.cfg[item] = val
        if isinstance(val, dict):
            new_obj = Config(val)
            setattr(self, item, new_obj)
        else:
            setattr(self, item, val)

    def __contains__(self, item):
        return (item in self.cfg)
    
    def get(self, item, alt = None):
        return self.cfg.get(item, alt)
    
    def items(self):
        return self.cfg.items()

    def keys(self):
        return self.cfg.keys()
    
    def values(self):
        return self.cfg.values()
    
    def save(self, file_path):
        with open(file_path, 'w') as f:
            yaml.dump(self.cfg, f)

    def __str__(self):
        return str(self.cfg)
    
    def __repr__(self):
        return self.cfg.__repr__()

    def _build_recursive(self):
        for k, v in self.cfg.items():
            if isinstance(v, dict):
                new_obj = Config(v)
                setattr(self, k, new_obj)
            else:
                setattr(self, k, v)

    @staticmethod
    def save_as_yaml(cfg : Config | dict, path : str):
        if isinstance(cfg, Config):
            cfg.save(path)
        elif isinstance(cfg, dict):
            with open(path, 'w') as f:
                yaml.dump(cfg, f)
        else:
            raise ValueError(f"Invalid object type of cfg {type(cfg)}, expected Config or dict.")