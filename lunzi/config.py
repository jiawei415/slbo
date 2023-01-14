# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import yaml
import time
from lunzi.Logger import logger


_frozen = False
_initialized = False


def expand(path):
    return os.path.abspath(os.path.expanduser(path))


class MetaFLAGS(type):
    _initialized = False

    def __setattr__(self, key, value):
        assert not _frozen, "Modifying FLAGS after dumping is not allowed!"
        super().__setattr__(key, value)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        for key, value in self.__dict__.items():
            if not key.startswith("_") and not isinstance(value, classmethod):
                if isinstance(value, MetaFLAGS):
                    value = dict(value)
                yield key, value

    def as_dict(self):
        return dict(self)

    def merge(self, other: dict):
        for key in other:
            assert key in self.__dict__, f"Can't find key `{key}`"
            if isinstance(self[key], MetaFLAGS) and isinstance(other[key], dict):
                self[key].merge(other[key])
            else:
                setattr(self, key, other[key])

    def set_value(self, path, value):
        key, *rest = path
        assert key in self.__dict__, f"Can't find key `{key}`"
        if not rest:
            setattr(self, key, value)
        else:
            self[key]: MetaFLAGS
            self[key].set_value(rest, value)

    @staticmethod
    def set_frozen():
        global _frozen
        _frozen = True

    def freeze(self):
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if isinstance(value, MetaFLAGS):
                    value.freeze()
        self.finalize()

    def finalize(self):
        pass


class BaseFLAGS(metaclass=MetaFLAGS):
    pass


def parse(cls):
    global _initialized

    if _initialized:
        return
    parser = argparse.ArgumentParser(description="Stochastic Lower Bound Optimization")
    parser.add_argument("-a", "--alg", type=str, default="ppo")
    parser.add_argument("-e", "--env", type=str, default="half_cheetah")
    parser.add_argument("-s", "--seed", type=int, default=2023)
    parser.add_argument("-l", "--logdir", type=str, default="~/results/slbo")

    args, unknown = parser.parse_known_args()
    for a in unknown:
        logger.info("unknown arguments: %s", a)

    cls.config.merge(vars(args))
    for config in [f"configs/algos/{args.alg}.yml", f"configs/envs/{args.env}.yml"]:
        cls.merge(yaml.load(open(expand(config)), Loader=yaml.FullLoader))

    time_tag = time.strftime("%Y%m%d%H%M%S", time.localtime())
    run_id = f"{args.alg}_{args.env}_{args.seed}_{time_tag}"
    log_dir = os.path.join(args.logdir, args.env, run_id)
    log_dir = os.path.expanduser(log_dir)
    cls.set_value(["run_id"], run_id)
    cls.set_value(["log_dir"], log_dir)
    cls.set_value(["seed"], args.seed)

    _initialized = True
