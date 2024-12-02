
import evaluate
import whisper
from typing import Any, Dict, List
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
from pytorch_lightning import Trainer
from nemo.core.classes.common import PretrainedModelInfo
from models import ASRModel
from datasets import SpeechDataset, WhisperDataCollatorWhithPadding

class LongaASRModel(ASRModel):
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo] | None:
        return super().list_available_models()
    
    def setup_training_data(self, train_data_config: DictConfig | Dict):
        return super().setup_training_data(train_data_config)
    
    def setup_validation_data(self, val_data_config: DictConfig | Dict):
        return super().setup_validation_data(val_data_config)
    
    def setup_test_data(self, test_data_config: DictConfig | Dict):
        return super().setup_test_data(test_data_config)    
    

class LongaTrainer(LongaASRModel):
    def __init__(self, cfg: OmegaConf, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        
    def step_(self, batch, idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        # self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        
        key = 'loss'
        self.log(key, loss)

        return {key: loss}
    
    def training_step(self, *args: Any, **kwargs: Any):
        return self.step_(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any):
        return self.step_(*args, **kwargs)
    
    def test_step(self, *args: Any, **kwargs: Any):
        return self.step_(*args, **kwargs)
    
class LongaModel(LongaTrainer):
    def _setup_data_loader(self, data_config):
        
        self.options = whisper.DecodingOptions(data_config.tokenizer.whisper.decoding_options)
        self.tokenizer = whisper.tokenizer.get_tokenizer(data_config.tokenizer.whisper)
        
        dataset = SpeechDataset(self.data_config.audio_info, self.tokenizer, self.data_config.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.data_config.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.data_config.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
        )
    
    def setup_training_data(self, train_data_config: OmegaConf):
        self._train_dl = self._setup_data_loader(train_data_config)

    def setup_validation_data(self, val_data_config: OmegaConf):
        self._validation_dl = self._setup_data_loader(val_data_config)

    def setup_test_data(self, test_data_config: OmegaConf):
        self._test_dl = self._setup_data_loader(test_data_config)