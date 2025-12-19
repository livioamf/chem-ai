import os
import json
from typing import Optional
import pytorch_lightning as pl
import torch
from peft import PeftModel


class BestModelExporter(pl.Callback):
    def __init__(
        self,
        export_dir: str,
        scaler: Optional[object] = None,  
    ):
        super().__init__()
        self.export_dir = export_dir
        self.scaler = scaler

    @staticmethod
    def _log(msg: str):
        print(f'[Exporter] {msg}')

    def _get_best_model_path(self, trainer):
        ckpt_callback = next(
            (
                cb
                for cb in trainer.callbacks
                if isinstance(cb, pl.callbacks.ModelCheckpoint)
            ),
            None,
        )
        if ckpt_callback is None:
            self._log('Nenhum ModelCheckpoint encontrado.')
            return None
        best_model_path = ckpt_callback.best_model_path or getattr(
            ckpt_callback, 'last_model_path', None
        )
        if not best_model_path or not os.path.exists(best_model_path):
            self._log('Nenhum melhor checkpoint encontrado.')
            return None
        return best_model_path
    
    def _reload_best_model(self, pl_module, best_model_path):
        try:
            best_model = type(pl_module).load_from_checkpoint(
                best_model_path, base_model=pl_module.base_model
            )
            best_model.eval()
            return best_model
        except Exception as exc:
            self._log(f'Erro ao recarregar checkpoint: {exc}')
            return None
        
    def _export_lora_adapters(self, base_model):
        try:
            if isinstance(base_model, PeftModel):
                base_model.save_pretrained(self.export_dir)
                self._log('Apenas os adapters LoRA foram salvos.')
            else:
                self._log('Modelo não é LoRA/PEFT, não será salvo.')
        except Exception as exc:
            self._log(f'Erro ao salvar adapters LoRA: {exc}')

    def _export_mlp(self, best_model):
        try:
            torch.save(best_model.mlp, os.path.join(self.export_dir, 'mlp.pt'))
            self._log('MLP salva em mlp.pt')
        except Exception as exc:
            self._log(f'Erro ao salvar MLP: {exc}')
    
    def _export_scaler(self):
        if self.scaler is not None:
            try:
                scaler_path = os.path.join(self.export_dir, 'scaler.json')
                scaler_dict = {
                    'mean_': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                    'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                    'var_': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
                    'n_features_in_': getattr(self.scaler, 'n_features_in_', None),
                    'feature_names_in_': getattr(self.scaler, 'feature_names_in_', None),
                }
                with open(scaler_path, 'w') as f:
                    json.dump(scaler_dict, f)
                self._log('Scaler salvo em scaler.json (manual).')
            except Exception as exc:
                self._log(f'Erro ao salvar scaler: {exc}')
        else:
            self._log('Nenhum scaler fornecido')
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        os.makedirs(self.export_dir, exist_ok=True)
        best_model_path = self._get_best_model_path(trainer)
        if best_model_path is None:
            return
        self._log(f'Recarregando melhor checkpoint: {best_model_path}')
        best_model = self._reload_best_model(pl_module, best_model_path)
        if best_model is None:
            return
        base_model = best_model.base_model
        self._export_lora_adapters(base_model)
        self._export_mlp(best_model)
        self._export_scaler()
        self._log('Exportação concluída com sucesso.')