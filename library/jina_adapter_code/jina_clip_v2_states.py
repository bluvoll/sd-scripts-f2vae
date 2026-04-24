import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

class JinaStates:
    """
    Extracts hidden states from jina-clip-v2 dynamically using PyTorch hooks.
    This safely bypasses Jina's custom wrappers without needing hardcoded model paths.
    """

    def __init__(self,
                 model_id: str,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16,
                 max_length: int = 512,
                 custom_train_jina: bool = False):
        
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.custom_train_jina = custom_train_jina
        print(f"Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=False,
        )
        
        print(f"Loading model from {model_id}...")
        self.model = AutoModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=False,
        )
        self.model.to(device)
        
        # Unload Vision Tower to save VRAM
        if hasattr(self.model, "vision_model"):
            del self.model.vision_model
            torch.cuda.empty_cache()
            print("Vision tower successfully unloaded to save VRAM.")
            
        self.model.eval()
        
        
        self.hidden_states_cache = None
        self.encoder_module = None
        
        for name, module in self.model.named_modules():
            # Exclude anything vision-related just in case
            if 'vision' in name.lower():
                continue
                
            # Transformers house their sequence layers in ModuleLists
            has_layer = hasattr(module, 'layer') and isinstance(getattr(module, 'layer'), torch.nn.ModuleList)
            has_layers = hasattr(module, 'layers') and isinstance(getattr(module, 'layers'), torch.nn.ModuleList)
            has_block = hasattr(module, 'block') and isinstance(getattr(module, 'block'), torch.nn.ModuleList)
            has_blocks = hasattr(module, 'blocks') and isinstance(getattr(module, 'blocks'), torch.nn.ModuleList)
            
            if has_layer or has_layers or has_block or has_blocks:
                layer_list = (getattr(module, 'layer', None) or getattr(module, 'layers', None) or 
                              getattr(module, 'block', None) or getattr(module, 'blocks', None))
                if layer_list is not None and len(layer_list) > 1:
                    self.encoder_module = module
                    break
                    
        if self.encoder_module is None:
            raise RuntimeError("Could not identify the text encoder module to attach a hook. Check model structure.")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        if self.custom_train_jina:
            for name, module in self.model.named_modules():
                if 'vision' in name.lower():
                    continue
                if isinstance(module, torch.nn.Embedding):
                    for param in module.parameters():
                        param.requires_grad = True
                    module.train() 

            num_layers = len(layer_list)
            for i in range(num_layers - 2, num_layers):
                for param in layer_list[i].parameters():
                    param.requires_grad = True
                layer_list[i].train() 
                
            if hasattr(self.model, 'text_projection'):
                for param in getattr(self.model, 'text_projection').parameters():
                    param.requires_grad = True

            print(f"Jina Training ENABLED: Unfrozen text embeddings and the last 2 layers.")
        else:
            print("Jina Training DISABLED: Model is completely frozen.")
        
        def forward_hook(module, args, output):
            # Output from encoder is BaseModelOutput or a tuple
            if hasattr(output, 'last_hidden_state'):
                self.hidden_states_cache = output.last_hidden_state
            elif isinstance(output, tuple):
                self.hidden_states_cache = output[0]
            else:
                self.hidden_states_cache = output
                
        self.encoder_module.register_forward_hook(forward_hook)
        print(f"Successfully attached hidden-state hook to: {self.encoder_module.__class__.__name__}")

    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Fallback mean pooling mirroring Jina-embeddings-v3."""
        hidden_states_f32 = hidden_states.to(torch.float32)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states_f32.size()).float()
        
        sum_embeddings = torch.sum(hidden_states_f32 * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        pooled = sum_embeddings / sum_mask
        return pooled.to(self.dtype)

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        import math
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        batch_size, seq_len = inputs["input_ids"].shape
        target_len = math.ceil(seq_len / 77) * 77
        pad_len = target_len - seq_len
        
        if pad_len > 0:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id or 0
                
            # Create padding tensors for the batch
            pad_ids = torch.full(
                (batch_size, pad_len), 
                pad_token_id, 
                dtype=inputs["input_ids"].dtype
            )
            pad_mask = torch.zeros(
                (batch_size, pad_len), 
                dtype=inputs["attention_mask"].dtype
            )
            
            # Concatenate along sequence dimension
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad_ids], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pad_mask], dim=1)
            
            # If tokenizer outputs token_type_ids, pad those with zeros as well
            if "token_type_ids" in inputs:
                pad_type_ids = torch.zeros(
                    (batch_size, pad_len), 
                    dtype=inputs["token_type_ids"].dtype
                )
                inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], pad_type_ids], dim=1)
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.set_grad_enabled(self.custom_train_jina):
            self.hidden_states_cache = None
            
            # Forward pass
            if hasattr(self.model, 'get_text_features'):
                pooled_state = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            else:
                out = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                if hasattr(out, 'text_embeds'):
                    pooled_state = out.text_embeds
                elif hasattr(out, 'pooler_output'):
                    pooled_state = out.pooler_output
                elif isinstance(out, tuple):
                    pooled_state = out[1] if len(out) > 1 else out[0]
                else:
                    pooled_state = out
                    
            # Ensure hook successfully captured the sequence states
            if self.hidden_states_cache is None:
                raise RuntimeError("Forward hook did not capture hidden states. The encoder module was bypassed.")
            
            # Fallback if pooled_state is somehow not a Tensor
            if not isinstance(pooled_state, torch.Tensor):
                pooled_state = self.mean_pooling(self.hidden_states_cache, inputs["attention_mask"])
                
            hidden_states = self.hidden_states_cache.to(self.dtype)
            pooled_state = pooled_state.to(self.dtype)
            
        return {
            "jina_hidden_states": hidden_states,
            "jina_mean_pooled_state": pooled_state,
            "attention_mask": inputs["attention_mask"]
        }
