#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Iterable
import sys
from typing import List

from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy
from nemo.utils import AppState
import torch
import torch.nn.functional as F

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from GPT3.export_utils import (
    get_past_key_name,
    get_past_value_name,
    get_new_key_name,
    get_new_value_name,
)
from GPT3.trt_utils import GPTTRTDecoder

sys.path.append('../../HuggingFace') # Include HuggingFace
from NNDF.logger import G_LOGGER

class TRTKVCache:
    def _allocate_memory(self):
        return torch.empty(
            self.max_seq_len,
            self.bs,
            self.nb_heads,
            self.head_size,
            dtype=self.type,
            device=torch.cuda.current_device(),
        ).contiguous().cuda()

    def __init__(self, num_layers, max_seq_len, bs, nb_heads, head_size, type):
        self.num_layers = num_layers
        self.keys = []
        self.values = []
        self.cur_seq_len = 0
        self.max_seq_len = max_seq_len
        self.bs = bs
        self.nb_heads = nb_heads
        self.head_size = head_size
        self.type = type
        for _ in range(num_layers):
            self.keys.append(self._allocate_memory())
            self.values.append(self._allocate_memory())
        self.element_size = self.keys[0].element_size()

    def get_key(self, index):
        assert index < len(self.keys)
        return self.keys[index]

    def get_value(self, index):
        assert index < len(self.values)
        return self.values[index]

    def update_dict(self, tensor_dict, seq_len):
        # Update KV input shapes every time before executing inference
        cur_shape = (self.cur_seq_len, self.bs, self.nb_heads, self.head_size)
        new_shape = (seq_len, self.bs, self.nb_heads, self.head_size)
        assert self.cur_seq_len + seq_len < self.max_seq_len

        offset = self.bs*self.nb_heads*self.head_size*self.cur_seq_len*self.element_size
        for i in range(self.num_layers):
            key_address = self.keys[i].data_ptr()
            value_address = self.values[i].data_ptr()
            # new kv address start from the past kv-cache data end
            tensor_dict[f"past_key_values.{i}.decoder.key"] = (key_address, cur_shape)
            tensor_dict[f"past_key_values.{i}.decoder.value"] = (value_address, cur_shape)

            new_key_address = key_address + offset
            new_value_address = value_address + offset
            tensor_dict[f"new_key_values.{i}.decoder.key"] = (new_key_address, new_shape)
            tensor_dict[f"new_key_values.{i}.decoder.value"] = (new_value_address, new_shape)
        self.cur_seq_len += seq_len

def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    tokens_to_generate,
    all_probs=False,
    type_ids=None,
    temperature=None,
    end_strings=['<|endoftext|>'],
    extra={},
):
    def repetition_penalty(logits, repetition_penalty, used_tokens):
        """ Implement the repetition penalty, check paper
        https://arxiv.org/pdf/1909.05858.pdf
        """
        if used_tokens is not None and repetition_penalty != 1.0:
            logits_update = torch.gather(logits, 1, used_tokens)
            logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
        return logits

    def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), started=None):
        """
        This function has been mostly taken from huggingface conversational
            ai code at
            https://medium.com/huggingface/how-to-build-a-state-of-the-art-
                conversational-ai-with-transfer-learning-2d818ac26313

            @param logits: logits tensor
            @param top_k: keep only top k tokens with highest probability
            @param top_p: keep the top tokens with cumulative probability
            @filter_value: value to set filtered tokens to
            @started: a tensor of bools indicating whether the text generation starts for the batch
            returns the filtered logits
        """
        if top_k > 0:
            # Remove all tokens with a probability less than the
            # last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            if started is not None:
                for i in torch.arange(indices_to_remove.size(0))[started]:
                    logits[i, indices_to_remove[i]] = filter_value
            else:
                logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Cconvert to 1D
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token
            # above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            if started is not None:
                for i in torch.arange(sorted_indices.size(0))[started]:
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = filter_value
            else:
                for i in range(sorted_indices.size(0)):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = filter_value

        return logits

    def end_of_generation_condition(
        tokenizer, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        END_OF_SEQ = end_strings[0]
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        assert False, "Not implemented error."

    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    if not (hasattr(model, "trt") or hasattr(model, "onnx")):
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
        )

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_length = context_lengths.min().item()
        inference_strategy.init_batch(context_tokens, context_length)
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()
        maxlen = inference_strategy.clip_max_len(maxlen)

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        kv_cache = None
        if hasattr(model, "trt") and extra.get("use_cache", False):
            kv_cache = TRTKVCache(model.cfg.num_layers, maxlen, batch_size, model.cfg.nb_heads, model.cfg.head_size, torch.float16)

        while context_length < maxlen:
            output = None

            if hasattr(model, "onnx") and extra.get("use_cache", False):
                G_LOGGER.warn(f"ONNX runtime path does not support KV-cache.")

            # Modify counter based on using cache or not.
            if not hasattr(model, "onnx") and extra.get("use_cache", False):
                batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                    tokens, maxlen, micro_batch_size, counter, context_length
                )
            else:
                batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                    tokens, maxlen, micro_batch_size, 0, context_length # step is always 0
                )

            # Choose which runtime to use
            if hasattr(model, "trt") or hasattr(model, "onnx"):
                assert len(batch) == 5, "Length of batch must be 5."
                (
                    batch_tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                seq_len = batch_tokens.shape[1]
                attention_mask = attention_mask[0:1, 0:1, 0:seq_len, 0:seq_len]

                # inputs input_ids: [BS, SEQ], position_ids: [BS, SEQ], attention_mask: [1, 1, SEQ, SEQ]
                if hasattr(model, "trt"):
                    assert isinstance(model.trt, GPTTRTDecoder)
                    input_ids_name = model.trt.get_input_ids_name()
                    input_ids = batch_tokens.type(model.trt.get_torch_type(input_ids_name)).contiguous().cuda()
                    tensor_dict = {input_ids_name : (input_ids.data_ptr(), input_ids.shape)}
                    position_ids_name = model.trt.get_position_ids_name()
                    if position_ids_name != None:
                        position_ids = position_ids.type(model.trt.get_torch_type(position_ids_name)).contiguous().cuda()
                        tensor_dict[position_ids_name] = (position_ids.data_ptr(), position_ids.shape)
                    attention_mask_name = model.trt.get_attention_mask_name()
                    if attention_mask_name != None:
                        attention_mask = attention_mask.type(model.trt.get_torch_type(attention_mask_name)).contiguous().cuda()
                        tensor_dict[attention_mask_name] = (attention_mask.data_ptr(), attention_mask.shape)

                    is_first_input = False
                    if extra.get("use_cache", False):
                        if set_inference_key_value_memory[0].item():
                            is_first_input = True
                            kv_cache.update_dict(tensor_dict, seq_len)
                        else:
                            assert kv_cache != None
                            kv_cache.update_dict(tensor_dict, 1)

                    logits_name = model.trt.get_output_name()
                    output = model.trt.run(logits_name, tensor_dict, is_first_input)
                else: # onnxrt path
                    from onnxruntime import InferenceSession
                    assert isinstance(model.onnx, InferenceSession)
                    # Currently only support onnx runtime with cpu
                    # Our fp8 models don't currently use attention_mask
                    tensor_dict = {'input_ids': batch_tokens.cpu().detach().numpy(),
                                   'position_ids': position_ids.cpu().detach().numpy()}
                    def have_attention_mask(sess):
                        all_inputs = sess.get_inputs()
                        for input in all_inputs:
                            if input.name == 'attention_mask':
                                return True
                        return False
                    if have_attention_mask(model.onnx):
                        tensor_dict['attention_mask'] = attention_mask.cpu().detach().numpy()
                    output = model.onnx.run(['logits'], tensor_dict)[0]
                    output = torch.Tensor(output).cuda()
                # output logits: [BS, SEQ, 50304]
            else:
                # nemo path
                output = inference_strategy.forward_step(batch, tensor_shape)
                output = output[0]['logits'].float()

            assert output is not None
            output = output.float()
            logits = output[:, -1].view(batch_size, -1).contiguous()

            # make sure it will generate at least min_length
            min_length = extra.get('min_tokens_to_generate', 0)
            if min_length > 0:
                within_min_length = (context_length - context_lengths) < min_length
                logits[within_min_length, eod_id] = -float('Inf')

            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float('Inf')

            # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens
            started = context_lengths <= context_length
            if extra.get('greedy', False):
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                logits /= temperature
                # handle repetition penality
                logits = repetition_penalty(logits, extra.get('repetition_penalty', 1.0), all_generated_indices)
                logits = top_k_logits(
                    logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9), started=started
                )
                probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(probs, num_samples=1).view(-1)

            # Clamp the predicted out of vocabulary tokens
            prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            new_tokens = torch.where(started, prev, tokens[:, context_length].view(-1))
            # Replace sampled tokens w/ done token if EOD has already been sampled
            new_tokens = torch.where(is_done, eod_id, new_tokens)
            # post process the inference tokens based on the strategy
            inference_strategy.post_process(tokens, new_tokens, context_length)

            # Insert either new predicted or next prompt token
            tokens[:, context_length] = new_tokens

            if output_logits is None:
                output = F.log_softmax(output[:, :context_length, :], 2)
                indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                output_logits = torch.gather(output, 2, indices).squeeze(2)
                all_generated_indices = indices[:, :, 0]
                if all_probs:
                    full_logits = output
            else:
                output = F.log_softmax(output, 2)
                indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                # This copy can be optimized out by pre-allocating the memory.
                output_logits = torch.cat([output_logits, new_output_logits], 1)
                all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
                if all_probs:
                    full_logits = torch.cat([full_logits, output], 1)

            done_token = end_of_generation_condition(tokenizer,
                                                     tokens[:, : context_length + 1], prev, eod_id, end_strings)
            done_token = done_token.byte() & started.byte()

            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token

            done = torch.all(is_done)
            if all_probs:
                yield tokens, lengths, output_logits, full_logits
            else:
                yield tokens, lengths, output_logits, None

            context_length += 1
            counter += 1
            if done and not extra.get("benchmark_mode", False):
                break

def initialize_ddp(model, cfg):
    # check whether the DDP is initialized
    if cfg.runtime == "nemo" and parallel_state.is_unitialized():
        def dummy():
            return
        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

        if model.cfg.get('transformer_engine', False):
            model.setup_transformer_engine_tp_groups()

def full_inference(model, inputs, cfg):
    initialize_ddp(model, cfg)

    tokens_to_generate = cfg.inference.tokens_to_generate
    min_tokens_to_generate = cfg.inference.min_tokens_to_generate
    add_BOS = cfg.inference.add_BOS
    all_probs = cfg.inference.all_probs
    temperature = cfg.inference.temperature
    end_strings = ['<|endoftext|>']
    is_benchmark_mode = True if cfg.mode == "benchmark" else False

    inference_strategy = GPTModelTextGenerationStrategy(model)
    tokenizer = model.tokenizer
    if isinstance(inputs, tuple):
        context_tokens_tensor, context_length_tensor = inputs
    else:
        context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
            inputs, tokens_to_generate, add_BOS
        )

    context_length = context_length_tensor.min().item()

    all_length = []
    batch_token_iterator = sample_sequence_batch(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature=temperature,
        end_strings=end_strings,
        extra={
            "top_p": cfg.inference.top_p,
            "top_k": cfg.inference.top_k,
            "greedy": cfg.inference.greedy,
            "repetition_penalty": cfg.inference.repetition_penalty,
            "min_tokens_to_generate": min_tokens_to_generate,
            "use_cache": cfg.use_cache,
            "benchmark_mode": is_benchmark_mode
        },
    )

    for tokens, lengths, output_logits, full_logits in batch_token_iterator:
        all_length.append(lengths[0].item())
        context_length += 1

    output = None
    if tokens is not None:
        output = tokens[:, :context_length], output_logits, full_logits

    special_tokens = set()
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        special_tokens.add(tokenizer.pad_token)
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        special_tokens.add(tokenizer.eos_token)
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
        special_tokens.add(tokenizer.bos_token)
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
        special_tokens.add(tokenizer.cls_token)
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
        special_tokens.add(tokenizer.unk_token)
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        special_tokens.add(tokenizer.sep_token)
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        special_tokens.add(tokenizer.mask_token)
    if output is not None:
        decode_tokens, output_logits, full_logits = output
        resp_sentences = []
        resp_sentences_seg = []

        decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)
            words = []
            for token in decode_token:
                if not isinstance(token, Iterable):
                    token = [token]
                word = tokenizer.ids_to_tokens(token)
                if isinstance(word, Iterable):
                    word = word[0]
                if hasattr(tokenizer.tokenizer, 'byte_decoder'):
                    word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                        'utf-8', errors='replace'
                    )
                words.append(word)
            resp_sentences_seg.append(words)

        # offsets calculation
        all_offsets = []
        for item in resp_sentences_seg:
            offsets = [0]
            for index, token in enumerate(item):
                if index != len(item) - 1:
                    if token in special_tokens:
                        offsets.append(offsets[-1])
                    else:
                        offsets.append(len(token) + offsets[-1])
            all_offsets.append(offsets)

        output = {}
        output['sentences'] = resp_sentences
        output['tokens'] = resp_sentences_seg
        output['logprob'] = output_logits
        output['full_logprob'] = full_logits
        output['token_ids'] = decode_tokens
        output['offsets'] = all_offsets
        output = inference_strategy.post_generation_process(output)
    return output

