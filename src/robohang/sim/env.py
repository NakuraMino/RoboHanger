import taichi as ti

from typing import Dict

import omegaconf

from .sim_utils import BaseClass
from .sim import Sim
from .actor import Actor

@ti.data_oriented
class Env(BaseClass):
    def __init__(self, sim: Sim, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)
        self._sim = sim

        self._actor_map: Dict[str, Actor] = {}

    @property
    def actor_map(self) -> Dict[str, Actor]:
        return self._actor_map
    
    def _add_actor(self, actor: Actor):
        assert actor._name not in self._actor_map.keys()
        self._actor_map[actor._name] = actor

    def get_state(self) -> dict:
        state = {
            "actor": {
                k: v.get_state() for k, v in self._actor_map.items()
            },
        }
        return state
    
    def set_state(self, state: dict):
        assert isinstance(state, dict)

        for k, v in state["actor"].items():
            self._actor_map[k].set_state(v)

    def reset(self):
        """Reset actor objects."""
        for v in self._actor_map.values():
            v.reset()