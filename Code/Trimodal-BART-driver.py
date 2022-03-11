# Code for Trimodal BART (MAF-TAV-BART)
# File: Trimodal-BART-driver.py

# Run code-mixed file:
# nohup python -u Trimodal-BART-driver.py > ../out-files/MAF-TAV-BART.out &


import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import re
import ast
from tqdm import tqdm
from typing import Optional
from datetime import datetime

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

TARGET_COLUMN = 'code_mixed_explanation'
TEXT_INPUT_PATH = '../Data/Text/'
ACOUSTIC_INPUT_PATH = '../Data/Audio/'
VISUAL_INPUT_PATH = '../Data/Video/'

MODEL_OUTPUT_DIR = '../models/MAF-TAV-BART/'
RESULT_OUTPUT_DIR = '../results/MAF-TAV-BART/'

LOWERCASE_UTTERANCES = False
UNFOLDED_DIALOGUE = True

if UNFOLDED_DIALOGUE:
    SOURCE_COLUMN = 'dialogue'
else:
    SOURCE_COLUMN_1 = 'target'
    SOURCE_COLUMN_2 = 'context'
    


SOURCE_MAX_LEN = 480
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25

ACOUSTIC_DIM = 154
ACOUSTIC_MAX_LEN = 600

VISUAL_DIM = 2048
VISUAL_MAX_LEN = 96 

BATCH_SIZE = 4
MAX_EPOCHS = 60

BASE_LEARNING_RATE = 5e-6
NEW_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

NUM_BEAMS = 5
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3

EARLY_STOPPING_THRESHOLD = 5


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)


# -------------------------------------------------------------- MODEL -------------------------------------------------------------- #


import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    BartTokenizerFast,
    AdamW
)

from transformers.models.bart.configuration_bart import BartConfig

from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    shift_tokens_right,
    _make_causal_mask,
    _expand_mask 
)


from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)


from transformer_encoder import TransformerEncoder


# ---------------------------------------------- Multimodal Context Aware Attention ----------------------------------------------

class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()
        
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, 
                                                     num_heads=1, 
                                                     dropout=self.dropout_rate, 
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=DEVICE)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)
        




    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):
        
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output
         



# ---------------------------------------------- Modality Aware Fusion ----------------------------------------------

class MAF(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        
        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=ACOUSTIC_DIM,
                                                                dropout_rate=dropout_rate)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=VISUAL_DIM,
                                                              dropout_rate=dropout_rate)        
        self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                acoustic_context: Optional[torch.Tensor]=None,
                visual_context: Optional[torch.Tensor]=None):
                    
        # Audio as Context for Attention
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0, 2, 1)
        
        audio_out = self.acoustic_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=acoustic_context)
        
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)
        
        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input +
                                       weight_a * audio_out + 
                                       weight_v * video_out)

        return output
    
    
# ---------------------------------------------- Multimodal BartEncoder ----------------------------------------------

class MultimodalBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False
        
        # ================================ Modifications ================================ #
        self.fusion_at_layer = [4]
        self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM, 
                                                     num_layers=4,
                                                     num_heads=8, 
                                                     dim_feedforward=VISUAL_DIM)
        self.acoustic_transformer = TransformerEncoder(d_model=ACOUSTIC_DIM, 
                                                       num_layers=4,
                                                       num_heads=2, 
                                                       dim_feedforward=ACOUSTIC_DIM)
        self.MAF_layer = MAF(dim_model=embed_dim,
                             dropout_rate=0.2)
        # =============================================================================== #

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            
            # ================================ Modifications ================================ #
            if idx in self.fusion_at_layer:
                acoustic_input = self.acoustic_transformer(acoustic_input)[-1]
                visual_input = self.visual_transformer(visual_input)[-1]
                hidden_states = self.MAF_layer(text_input=hidden_states,
                                               acoustic_context=acoustic_input,
                                               visual_context=visual_input)

            # =============================================================================== #
            
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                                 

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

# -------------------------------------------------- Multimodal MultimodalBartModel --------------------------------------------------

class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                acoustic_input=acoustic_input,      # New addition of acoustic_input
                visual_input=visual_input,      # New addition of visual_input
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

# ---------------------------------------------- MultiModalBartForConditionalGeneration ----------------------------------------------

class MultimodalBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            acoustic_input=acoustic_input,      # New addition of acoustic_input
            visual_input=visual_input,      # New addition of visual_input
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past



# ----------------------------------------------------------- DATA UTILS ----------------------------------------------------------- #

def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data





def prepare_dataset(text_path: str,
                    acosutic_path: str,
                    visual_path: str,
                    lowercase_utterances: bool=False,
                    unfolded_dialogue: bool=True):
    data = read_json_data(text_path)
    
    code_mixed_explanation = []   

    if unfolded_dialogue:
        dialogue = []
        for i in range(1, len(data)+1):
            data_point = data[str(i)]

            example_target_speaker = str(data_point['target_speaker']).upper()
            example_target_utterance = str(data_point['target_utterance'])

            example_dialogue = "[CONTEXT] "
            for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
                example_dialogue  = example_dialogue + str(speaker).upper() + " : " +  str(utterance) + " | "

            example_dialogue = example_dialogue + " [TARGET] " + example_target_speaker + " : " +  example_target_utterance + " | "
            example_dialogue = re.sub(' +', ' ', example_dialogue)
            dialogue.append(example_dialogue)

            code_mixed_explanation.append(str(data_point['code_mixed_explanation']))

        df =  pd.DataFrame(list(zip(dialogue, code_mixed_explanation)),
                            columns=['dialogue', 'code_mixed_explanation'])
        TOKENIZER.add_tokens(['[CONTEXT]', '[TARGET]'], special_tokens=True)
        MODEL.resize_token_embeddings(len(TOKENIZER))
        
        del dialogue
        del example_dialogue
        
    else:
        target = []
        context = []
        for i in range(1, len(data)+1):
            data_point = data[str(i)]

            example_target_speaker = str(data_point['target_speaker']).upper()
            example_target_utterance = str(data_point['target_utterance'])
            example_target_utterance = example_target_speaker + " : " + example_target_utterance
            example_target_utterance = re.sub(' +', ' ', example_target_utterance)
            target.append(example_target_utterance)

            example_context_utterance = ""
            for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
                example_context_utterance  = example_context_utterance + str(speaker).upper() + " : " +  str(utterance) + " | "
            
            example_context_utterance = re.sub(' +', ' ', example_context_utterance)
            context.append(example_context_utterance)

            code_mixed_explanation.append(str(data_point['code_mixed_explanation']))

        df =  pd.DataFrame(list(zip(context, target, code_mixed_explanation)),
                            columns=['context', 'target', 'code_mixed_explanation']) 
        del target
        del context
        del example_context_utterance

    # Reading Audio Data
    acosutic_data = pd.read_pickle(acosutic_path)
    df['audio_features'] = acosutic_data['audio_feats']
    
    # Reading Video Data
    visaul_data = pd.read_pickle(visual_path)
    df['visual_features'] = visaul_data['video_feats']
 
    df =  df[df['code_mixed_explanation'] != "?"]
    df = df.dropna()
    if lowercase_utterances:
        df = df.apply(lambda x: x.astype(str).str.lower())

    del data
    del text_path
    del acosutic_path
    del visual_path
    del acosutic_data
    del visaul_data
    del code_mixed_explanation
    del example_target_speaker
    del example_target_utterance
    gc.collect()
    return df


def pad_seq(tensor: torch.tensor,
            dim: int,
            max_len: int):
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    else:
        return tensor[:max_len]



def preprocess_dataset(dataset,
                       unfolded_dialogue: bool=True):
    
    if unfolded_dialogue:
        source = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN].values.tolist()]
        model_inputs = TOKENIZER(source,
                                 max_length=SOURCE_MAX_LEN,
                                 padding='max_length',
                                 truncation=True)        
        del source
        
    else:
        source_1 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_1].values.tolist()]
        source_2 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_2].values.tolist()]
        model_inputs = TOKENIZER(source_1,
                                 source_2,
                                 max_length=SOURCE_MAX_LEN,
                                 padding='max_length',
                                 truncation=True)
            
        del source_1
        del source_2
    
    target = [TARGET_PREFIX + t for t in dataset[TARGET_COLUMN].values.tolist()]
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(target,
                           max_length=TARGET_MAX_LEN,
                           padding='max_length',
                           truncation=True)    
        # IMP: 
        # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    
    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)

    model_inputs['acoustic_input'] = torch.stack([pad_seq(torch.tensor(af, dtype=torch.float),
                                                          dim=ACOUSTIC_DIM,
                                                          max_len=ACOUSTIC_MAX_LEN)
                                                  for af in dataset['audio_features'].values.tolist()], 0).to(DEVICE)
    
    model_inputs['visual_input'] = torch.stack([pad_seq(torch.tensor(vf[0], dtype=torch.float),
                                                        dim=VISUAL_DIM,
                                                        max_len=VISUAL_MAX_LEN)
                                                for vf in dataset['visual_features'].values.tolist()], 0).to(DEVICE)
    
    model_inputs['labels'] = torch.tensor([l for l in labels['input_ids']], dtype=torch.long, device=DEVICE)
    
    del target
    del labels
    gc.collect()
    return model_inputs


def set_up_data_loader(text_path: str,
                       acosutic_path: str,
                       visual_path: str,
                       lowercase_utterances: bool=False,
                       unfolded_dialogue: bool=True):
    dataset = preprocess_dataset(prepare_dataset(text_path=text_path,
                                                 acosutic_path=acosutic_path,
                                                 visual_path=visual_path,
                                                 lowercase_utterances=lowercase_utterances,
                                                 unfolded_dialogue=unfolded_dialogue),
                                unfolded_dialogue=unfolded_dialogue)
    print(dataset.keys())
    dataset = TensorDataset(dataset['input_ids'],
                            dataset['attention_mask'], 
                            dataset['acoustic_input'], 
                            dataset['visual_input'], 
                            dataset['labels'])
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



def get_scores(reference_list: list,
               hypothesis_list: list):
    count=0
    met=0
    bleu_1=0
    bleu_2=0
    bleu_3=0
    bleu_4=0
    rouge1=0
    rouge2=0
    rougel = 0
    weights_1 = (1./1.,)
    weights_2 = (1./2. , 1./2.)
    weights_3 = (1./3., 1./3., 1./3.)
    weights_4 = (1./4., 1./4., 1./4., 1./4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        met += meteor_score([reference], hypothesis)

        reference = reference.split()
        hypothesis = hypothesis.split()
        bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1*100/count,
        "rouge_2": rouge2*100/count,
        "rouge_L": rougel*100/count,
        "bleu_1": bleu_1*100/count,
        "bleu_2": bleu_2*100/count,
        "bleu_3": bleu_3*100/count,
        "bleu_4": bleu_4*100/count,
        "meteor": met*100/count,
    }



def _save(model, 
          output_dir: str,
          tokenizer=None,
          state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def save_model(model, 
               output_dir: str,
               tokenizer=None, 
               state_dict=None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)




# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def train_epoch(model,
                data_loader,
                optimizer):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, acoustic_input, visual_input, labels = batch
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=acoustic_input,
                        visual_input=visual_input,
                        labels=labels)
        loss = outputs['loss']
        epoch_train_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_train_loss/ step




def val_epoch(model,
              data_loader,
              optimizer):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            acoustic_input=acoustic_input,
                            visual_input=visual_input,
                            labels=labels)
            loss = outputs['loss']
            epoch_val_loss += loss.item()  
    
    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache() 
    
    return epoch_val_loss/ step




def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               **gen_kwargs):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch

            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           acoustic_input=acoustic_input,
                                           visual_input=visual_input,
                                           **gen_kwargs)
                            
            generated_ids = generated_ids.detach().cpu().numpy()
            generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels = labels.detach().cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
            predictions.extend(decoded_preds)
            gold.extend(decoded_labels)
    
    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del generated_ids
    del decoded_preds
    del decoded_labels
    gc.collect()
    torch.cuda.empty_cache() 
    
    return predictions, gold




def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   **gen_kwargs):
    predictions, gold = test_epoch(model,
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
    result = get_scores(predictions, gold)
    
    if "Validation" in desc:
        val_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
        file_name = RESULT_OUTPUT_DIR + "val/MAF_TAV_BART_epoch_" + str(epoch+1) + "_val_results.csv"
        val_df.to_csv(file_name, index=False) 
        print("Validation File saved")
        
    elif "Test" in desc:
        test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
        file_name = RESULT_OUTPUT_DIR + "test/MAF_TAV_BART_epoch_" + str(epoch+1) + "_test_results.csv"
        test_df.to_csv(file_name, index=False)  
        print("Test File saved")
    
    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache() 
    
    return result  




def prepare_for_training(model, 
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)
            
    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}            
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )
    
    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache() 
    
    return optimizer




def train(model,
          tokenizer,
          train_data_loader,
          val_data_loader,
          test_data_loader,
          base_learning_rate,
          new_learning_rate,
          weight_decay,
          **gen_kwargs):
    
    optimizer = prepare_for_training(model=model,
                                     base_learning_rate=base_learning_rate,
                                     new_learning_rate=new_learning_rate,
                                     weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    val_rouge_2 = []
    patience = 1
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model,
                                 train_data_loader, 
                                 optimizer)
        train_losses.append(train_loss)
        
        val_loss = val_epoch(model,
                             val_data_loader, 
                             optimizer)
        val_losses.append(val_loss)

        val_results = get_val_scores(model,
                                     tokenizer,
                                     val_data_loader,
                                     desc="Validation Generation Iteration",
                                     epoch=epoch,
                                     **gen_kwargs)
        val_rouge_2.append(val_results['rouge_2'])
        
        test_results = get_val_scores(model,
                                      tokenizer,
                                      test_data_loader,
                                      desc="Test Generation Iteration",
                                      epoch=epoch,
                                      **gen_kwargs)
    
        print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
        
        print("\nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
        val_results['rouge_1'], val_results['rouge_2'], val_results['rouge_L'], val_results['bleu_1'], val_results['bleu_2'], val_results['bleu_3'], val_results['bleu_4'], val_results['meteor']))
        
        print("\ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}".format(
        test_results['rouge_1'], test_results['rouge_2'], test_results['rouge_L'], test_results['bleu_1'], test_results['bleu_2'], test_results['bleu_3'], test_results['bleu_4'], test_results['meteor']))
        
        path = MODEL_OUTPUT_DIR + "MAF_TAV_BART_epoch_"  + TARGET_COLUMN + "_epoch_" + str(epoch+1) + "_" + datetime.now().strftime('%d-%m-%Y-%H:%M')
        print(path)
        save_model(model,
                   path,
                   tokenizer)
        print("Model saved at path: ", path)
        
        if val_results['rouge_2'] < max(val_rouge_2):
            patience = patience + 1          
            if patience == EARLY_STOPPING_THRESHOLD:
                break
                
        else:
            patience = 1

        del train_loss
        del val_loss
        del path
        gc.collect()
        torch.cuda.empty_cache() 


# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":

    MODEL = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base')
    print("Model loaded...\n")
    MODEL.to(DEVICE)

    TOKENIZER = BartTokenizerFast.from_pretrained('facebook/bart-base')
    print("Tokenizer loaded...\n")

    SOURCE_PREFIX = ''
    TARGET_PREFIX = ''
    
    print(TARGET_COLUMN)
    print(MODEL_OUTPUT_DIR)
    print(RESULT_OUTPUT_DIR)
    print(SOURCE_PREFIX)
    print(TARGET_PREFIX)
    
    gc.collect()
    
    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print("Total parameters: ", pytorch_total_params)
    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print("Total trainable parameters: ", pytorch_total_train_params)
    
    for name, param in MODEL.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name: 
            print(name)
    
    
    # ------------------------------ READ DATASET ------------------------------ #
    
    train_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + 'train_text_sample.json',
                                       acosutic_path = ACOUSTIC_INPUT_PATH + 'train_audio_sample.p',
                                       visual_path = VISUAL_INPUT_PATH + 'train_video_sample.p',
                                       lowercase_utterances = LOWERCASE_UTTERANCES,
                                       unfolded_dialogue = UNFOLDED_DIALOGUE)
    print("\nTraining Data Loaded...")
    
    val_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + 'val_text_sample.json',
                                     acosutic_path = ACOUSTIC_INPUT_PATH + 'val_audio_sample.p',
                                     visual_path = VISUAL_INPUT_PATH  + 'val_video_sample.p',
                                     lowercase_utterances = LOWERCASE_UTTERANCES,
                                     unfolded_dialogue = UNFOLDED_DIALOGUE)
    print("\nValidation Data Loaded...")
    
    test_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + 'test_text_sample.json',
                                      acosutic_path = ACOUSTIC_INPUT_PATH + 'test_audio_sample.p',
                                      visual_path = VISUAL_INPUT_PATH  + 'test_video_sample.p',
                                      lowercase_utterances = LOWERCASE_UTTERANCES,
                                      unfolded_dialogue = UNFOLDED_DIALOGUE)
    print("\nTest Data Loaded...")
    gc.collect()  
    
    # ------------------------------ TRAINING SETUP ------------------------------ #
        
    gen_kwargs = {
        'num_beams': NUM_BEAMS,
        'max_length': TARGET_MAX_LEN,
        'early_stopping': EARLY_STOPPING,
        'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE
    }
    
    train(model=MODEL,
          tokenizer=TOKENIZER,
          train_data_loader=train_dataset,
          val_data_loader=val_dataset,
          test_data_loader=test_dataset,
          base_learning_rate=BASE_LEARNING_RATE,
          new_learning_rate=NEW_LEARNING_RATE,
          weight_decay=WEIGHT_DECAY,
          **gen_kwargs)
    
    print("Model Trained!")