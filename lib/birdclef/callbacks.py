import time
import re
import json
from pathlib import Path
import torch

class AutoSave:
    def __init__(self, top_k=2, metric="f1", mode="min", root=None, name="ckpt"):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root or MODEL_ROOT)
        assert self.root.exists()
        self.name = name
        self.top_models = []
        self.top_metrics = []

    def log(self, model, metrics):
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank+1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metric, rank, metrics["epoch"])

    def save(self, model, metric, rank, epoch):
        t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(self.name, epoch, self.metric, metric, t)
        name = re.sub(r"[^\w_-]", "", name) + ".pth"
        path = self.root.joinpath(name)
        old_model = None
        self.top_models.insert(rank+1, name)
        if len(self.top_models) > self.top_k:
            old_model = self.root.joinpath(self.top_models[0])
            self.top_models.pop(0)
        torch.save(model.state_dict(), path.as_posix())
        if old_model is not None:
            old_model.unlink()
        self.to_json()

    def rank(self, val):
        r = -1
        for top_val in self.top_metrics:
            if val <= top_val:
                return r
            r += 1
        return r

    def to_json(self):
        name = "{}_logs".format(self.name)
        name = re.sub(r"[^\w_-]", "", name) + ".json"
        path = self.root.joinpath(name)
        with path.open("w") as f:
            json.dump(self.logs, f, indent=2)
