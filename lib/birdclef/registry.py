from typing import List
from pathlib import Path

class Entry:
    def __init__(
        self,
        id:str,
        filepath:Path
    ):
        self.id = id
        self.filepath = filepath

def get_entries() -> List[Entry]:
    return [
        Entry(
            id          = "resnest50",
            filepath    = Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")
        ),
        Entry(
            id          = "densenet121",
            filepath    = Path("./densenet121_sr32000_d7_v1_v1/birdclef_densenet121_fold0_epoch_19_f1_val_07459_20210515233517.pth")
        ),
        Entry(
            id          = "efficientnet-b3",
            filepath    = Path("./efficientnet-b3_sr32000_d7_v1_v1/birdclef_efficientnet-b3_fold0_epoch_19_f1_val_07605_20210515074202.pth")
        ),
        Entry(
            id          = "resnext101_32x8d_wsl",
            filepath    = Path("./resnext101_32x8d_wsl_sr32000_d7_v1_v1/birdclef_resnext101_32x8d_wsl_fold0_epoch_19_f1_val_07461_20210515051857.pth")
        ),
        Entry(
            id          = "resnext101_32x16d_wsl",
            filepath    = Path("./resnext101_32x16d_wsl_sr32000_d7_v1_v1/birdclef_resnext101_32x16d_wsl_fold0_epoch_19_f1_val_07556_20210516052951.pth")
        )
    ]

def get_entry_by_id(id:str) -> Entry:
    for entry in get_entries():
        if entry.id == id:
            return entry

