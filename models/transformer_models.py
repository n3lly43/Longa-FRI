import whisper 
from omegaconf import OmegaConf
import pytorch_lightning as ptl

from nemo.core import ModelPT

class ASRModel(ModelPT):
    def __init__(self, cfg:OmegaConf, trainer: ptl.Trainer) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        if self.cfg.model_name.contains('whisper'):
            self.model = whisper.load_model(self.cfg.model_name)

    def forward(self, x):
        return self.model(x)
