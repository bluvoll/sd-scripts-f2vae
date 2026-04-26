import torch
from torch.nn import functional as F
import argparse
import math
import os
from typing import List, Dict, Optional, Tuple
import toml
from transformers import T5GemmaEncoderModel, Qwen3VLForConditionalGeneration, AutoTokenizer
from safetensors.torch import load_file


import logging
logger = logging.getLogger(__name__)

from jina_clip_v2_states import JinaStates
from jina_to_sdxl_adapter_v2 import JinaToSDXLAdapterV2

class JinaAndAdapter(torch.nn.Module):
    """
    A wrapper holding both the LLM and adapter.
    """
    def __init__(self, llm_model, tokenizer, llm_adapter, should_train_llm=False):
        super().__init__()
        self.llm_model = llm_model # Jina_states
        self.tokenizer = tokenizer # Here for vibes
        self.llm_adapter = llm_adapter # Jina_to_sdxl_adapter_v2.py
        
        self.train_llm = should_train_llm
        
    def _build_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        jina_hidden_states = batch['jina_hidden_states']
        if jina_hidden_states.dim() > 3:
            jina_hidden_states = jina_hidden_states.squeeze(0)

        jina_mean_pooled_state = batch['jina_mean_pooled_state']
        if jina_mean_pooled_state.dim() > 2:
            jina_mean_pooled_state = jina_mean_pooled_state.squeeze(0)

        attention_mask = batch['attention_mask']
        if attention_mask.dim() > 2:
            attention_mask = attention_mask.squeeze(0)

        
        target_device = next(self.llm_adapter.parameters()).device

        
        return {
            'jina_hidden_states': jina_hidden_states.float().to(target_device),
            'jina_mean_pooled_state': jina_mean_pooled_state.float().to(target_device),
            'attention_mask': attention_mask.long().to(target_device)
        }
    def forward(self, captions):
        """
        Forward pass from tokenized input to SDXL-compatible embeddings.
        """
        
        if self.train_llm:
            jina_outputs = self.llm_model(captions)
            input_data = self._build_model_inputs(jina_outputs)
        else:
            with torch.set_grad_enabled(self.train_llm): # Freeze LLM 
                jina_outputs = self.llm_model(captions)
            input_data = self._build_model_inputs(jina_outputs)
            
        prompt_embeds, pooled_embeds = self.llm_adapter(**input_data)
        
        return prompt_embeds, pooled_embeds

def load_jina_and_adapter(args, train_adapter=False):
        """
        function for loading the Jina and adapter.
        """
        logger.info("Loading Jina and adapter.")
        
        # Load Jina model via JinaStates
        train_jina_clip_certain_layers = False
        if getattr(args, "train_jina_clip_layers", False):
            train_jina_clip_certain_layers = True
            
        jina_model = JinaStates(
            model_id=args.llm_model_path,
            device="cpu",
            dtype=torch.bfloat16,
            max_length=512,
            custom_train_jina=train_jina_clip_certain_layers
        )
        print(f"Loaded Jina from: {args.llm_model_path}")
        
        # gradient checkpointing + training
        if getattr(args, "should_train_llm_encode", False) or train_jina_clip_certain_layers:
            pass    
        else:
            jina_model.model.eval()
            for param in jina_model.model.parameters():
                param.requires_grad = False
                
        if getattr(args, "gradient_checkpointing", False):
            jina_model.model.gradient_checkpointing_enable()
            if hasattr(jina_model.model, "enable_input_require_grads"):
                # jina_model.model.enable_input_require_grads()
                logger.info(f"jina_model enable_input_require_grads doesnt exist. This is fine")
            elif hasattr(jina_model.model, "get_input_embeddings"):
                # jina_model.model.get_input_embeddings().requires_grad_(True)
                logger.info(f"jina_model get_input_embeddings() doesnt exist. This is fine")
                
        ADAPTER_RESUME_PATH = args.llm_adapter_path
        
        adapter = JinaToSDXLAdapterV2(
            llm_dim=1024,
            sdxl_seq_dim=2048,
            sdxl_pooled_dim=1280,
            n_attention_blocks=4,
            num_heads=16,
            dropout=0,
            max_seq_len=539 # For multiple of 77
        )
        
        logger.info(f"Loading state_dict from {ADAPTER_RESUME_PATH}")
        old_state = load_file(ADAPTER_RESUME_PATH, device="cpu")
        
        new_state = old_state            
        adapter.load_state_dict(new_state, strict=False)
        
        if train_adapter:
            adapter.train()
        else:
            adapter.eval()
            
        should_train_llm = False
        if getattr(args, "should_train_llm_encode", False):
            should_train_llm = True
            
        text_encoder = JinaAndAdapter(
            llm_model=jina_model,
            tokenizer=jina_model.tokenizer, # from the JinaStates
            llm_adapter=adapter,
            should_train_llm=train_jina_clip_certain_layers
        )
        
        logger.info("Successfully loaded Jina and initialized adapter.")
        return text_encoder

def get_llm_text_conditioning(args, batch, text_encoder, tokenizer, accelerator, weight_dtype):
    """
    function for getting text conditioning from LLM.
    """
    # Take captions from batch pass through the LLM and adapter, and return the embeddings
    
    captions = batch.get("captions")
    if captions is None:
        raise ValueError("Batch does not contain 'captions'.")
    if args.adapter_jina:
        prompt_embeds, pooled_embeds = text_encoder(captions)
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_embeds
    }