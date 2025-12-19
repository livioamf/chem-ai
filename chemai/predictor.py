import os
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler

from peft import PeftModel

class ChemBERTPredictor:
    def __init__(self, mode, model_dir, hf_model_name, max_length=128):
        self.mode = mode
        self.model_dir = model_dir
        self.hf_model_name = hf_model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = self._load_tokenizer()
        self.base_model = self._load_base_model()
        self.mlp = self._load_mlp()
        self.scaler = self._load_scaler()
    
    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.hf_model_name)
    
    def _is_lora(self):
        return os.path.exists(os.path.join(self.model_dir, "adapter_config.json"))
    
    def _load_base_model(self):
        base = AutoModel.from_pretrained(self.hf_model_name)
        if self._is_lora():
            return PeftModel.from_pretrained(base, self.model_dir).to(self.device).eval()
        return base.to(self.device).eval()
    
    def _load_mlp(self):
        mlp = torch.load(f"{self.model_dir}/mlp.pt", map_location=self.device, weights_only=False)
        return mlp.eval().to(self.device)
    
    def _load_scaler(self):
        scaler_path = os.path.join(self.model_dir, "scaler.json")
        if os.path.exists(scaler_path):
            with open(scaler_path) as f:
                scaler_dict = json.load(f)

            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_dict['mean_'])
            scaler.scale_ = np.array(scaler_dict['scale_'])
            scaler.var_ = np.array(scaler_dict['var_'])
            scaler.n_features_in_ = scaler_dict['n_features_in_']
            scaler.feature_names_in_ = scaler_dict['feature_names_in_']
            return scaler
        return None
    
    def _encode_smiles(self, smiles_list):
        tokens = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    @torch.no_grad()
    def predict(self, smiles1, smiles2=None, frac=None, temp=None):
        if smiles1 is None:
            raise ValueError("O parâmetro 'smiles1' é obrigatório em qualquer modo.")
        if isinstance(smiles1, str):
            smiles1 = [smiles1]
        if temp is not None and not isinstance(temp, (list, np.ndarray, torch.Tensor)):
            temp = [temp]
        if smiles2 is not None and isinstance(smiles2, str):
            smiles2 = [smiles2]
        if frac is not None and not isinstance(frac, (list, np.ndarray, torch.Tensor)):
            frac = [frac]
        n = len(smiles1)
        if self.mode == "pure":
            if temp is None:
                raise ValueError(
                    "Para o modo 'pure', o parâmetro 'temp' deve ser informado."
                )
            if len(temp) != n:
                raise ValueError(
                    "Os parâmetros 'smiles1' e 'temp' devem ter o mesmo tamanho."
                )
        elif self.mode == "mix":
            if None in (smiles2, frac, temp):
                raise ValueError(
                    "Para o modo 'mix', é necessário informar 'smiles1', 'smiles2', 'frac' e 'temp'."
                )
            if len(smiles2) != n or len(frac) != n or len(temp) != n:
                raise ValueError(
                    "Todos os parâmetros ('smiles1', 'smiles2', 'frac', 'temp') devem ter o mesmo tamanho."
                )
        else:
            raise ValueError("Modo inválido. Use 'pure' ou 'mix'.")
        
        tokens1 = self._encode_smiles(smiles1)
        out1 = self.base_model(**tokens1)
        cls1 = out1.last_hidden_state[:, 0, :]
        
        temp_arr = np.array(temp).reshape(-1, 1)
        if self.scaler is not None:
            temp_arr = self.scaler.transform(temp_arr)
        t = torch.tensor(temp_arr, device=self.device, dtype=torch.float32)
        
        if self.mode == "pure":
            x = torch.cat([cls1, t], dim=1)
            y_hat = self.mlp(x).squeeze(1)
        else:
            tokens2 = self._encode_smiles(smiles2)
            out2 = self.base_model(**tokens2)
            cls2 = out2.last_hidden_state[:, 0, :]
            f = torch.tensor(frac, device=self.device, dtype=torch.float32).unsqueeze(1)
            x1 = torch.cat([cls1, cls2, t, f], dim=1)
            x2 = torch.cat([cls2, cls1, t, 1 - f], dim=1)
            y_hat = 0.5 * (self.mlp(x1) + self.mlp(x2))
            y_hat = y_hat.squeeze(1)
        return y_hat.cpu().numpy()