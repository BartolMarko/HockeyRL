import yaml
from typing import Union

class Config:
    def __init__(self, source : Union[dict, str]):
        if type(source) is dict:
            self.cfg = source
        else:
            with open(source) as f:
                self.cfg = yaml.safe_load(f)
        
        Config._build_recursive(self, self.cfg)
    
    def __getitem__(self, item):
        return self.cfg[item]
    
    def get(self, item, alt = None):
        return self.cfg.get(item, alt)
    
    def items(self):
        return self.cfg.items()

    def keys(self):
        return self.cfg.keys()
    
    def values(self):
        return self.cfg.values()

    def __str__(self):
        return str(self.cfg)
    
    def __repr__(self):
        return self.cfg.__repr__()

    @staticmethod
    def _build_recursive(obj, atts):
        for k, v in atts.items():
            if isinstance(v, dict):
                new_obj = Config(v)
                setattr(obj, k, new_obj)
                Config._build_recursive(new_obj, v)
            else:
                setattr(obj, k, v)