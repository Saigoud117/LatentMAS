# # import os
# # import csv
# # import torch
# # import matplotlib.pyplot as plt
# # from typing import Dict, List, Optional, Tuple
# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # try:
# #     from vllm import LLM, SamplingParams
# #     _HAS_VLLM = True
# # except ImportError:
# #     _HAS_VLLM = False

# # try:
# #     from transformers.cache_utils import Cache
# # except ImportError:
# #     Cache = None


# # def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
# #     if tokenizer.pad_token_id is None:
# #         if tokenizer.eos_token is not None:
# #             tokenizer.pad_token = tokenizer.eos_token
# #         else:
# #             tokenizer.add_special_tokens({"pad_token": "<pad>"})


# # def _past_length(past_key_values: Optional[Tuple]) -> int:
# #     if past_key_values is None:
# #         return 0

# #     # New transformers cache API (DynamicCache / Cache)
# #     if Cache is not None and isinstance(past_key_values, Cache):
# #         return int(past_key_values.get_seq_length())

# #     # Legacy tuple-of-tuples cache
# #     if isinstance(past_key_values, (list, tuple)):
# #         if len(past_key_values) == 0:
# #             return 0
# #         k = past_key_values[0][0]
# #         return k.shape[-2]

# #     return 0


# # class ModelWrapper:
# #     def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
# #         self.model_name = model_name
# #         self.device = device
# #         self.use_vllm = use_vllm and _HAS_VLLM
# #         self.vllm_engine = None
# #         self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
# #         self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
# #         self.args = args

# #         # for ablation
# #         self.pre_aligned = None

# #         if self.use_vllm:
# #             tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
# #             gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

# #             print(f"[vLLM] Using vLLM backend for model {model_name}")
# #             if args.enable_prefix_caching and args.method == "latent_mas":
# #                 self.vllm_engine = LLM(
# #                     model=model_name,
# #                     tensor_parallel_size=tp_size,
# #                     gpu_memory_utilization=gpu_util,
# #                     enable_prefix_caching=True,
# #                     enable_prompt_embeds=True,
# #                 )
# #             else:
# #                 self.vllm_engine = LLM(
# #                     model=model_name,
# #                     tensor_parallel_size=tp_size,
# #                     gpu_memory_utilization=gpu_util,
# #                 )

# #             self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# #             use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
# #             if use_second_hf:
# #                 self.HF_model = AutoModelForCausalLM.from_pretrained(
# #                     model_name,
# #                     torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
# #                 ).to(args.device2).eval()
# #                 self.embedding_layer = self.HF_model.get_input_embeddings()
# #                 self.HF_device = args.device2
# #                 self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
# #             elif self.latent_space_realign:
# #                 raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")

# #             _ensure_pad_token(self.tokenizer)
# #             return  # skip loading transformers model

# #         # fallback: normal transformers path
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# #         _ensure_pad_token(self.tokenizer)

# #         with torch.no_grad():
# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 model_name,
# #                 torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
# #                 device_map="auto"
# #             )

# #         if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
# #             self.model.resize_token_embeddings(len(self.tokenizer))

# #         # self.model.to(device)
# #         self.model.eval()

# #         if hasattr(self.model.config, "use_cache"):
# #             self.model.config.use_cache = True

# #         if self.latent_space_realign:
# #             self._ensure_latent_realign_matrix(self.model, self.device, args)

# #     def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
# #         tpl = getattr(self.tokenizer, "chat_template", None)
# #         if tpl:
# #             return self.tokenizer.apply_chat_template(
# #                 messages, tokenize=False, add_generation_prompt=add_generation_prompt
# #             )

# #         segments = []
# #         for message in messages:
# #             role = message.get("role", "user")
# #             content = message.get("content", "")
# #             segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
# #         if add_generation_prompt:
# #             segments.append("<|assistant|>")
# #         return "\n".join(segments)

# #     def prepare_chat_input(
# #         self, messages: List[Dict], add_generation_prompt: bool = True
# #     ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
# #         prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
# #         encoded = self.tokenizer(
# #             prompt_text,
# #             return_tensors="pt",
# #             add_special_tokens=False,
# #         )
# #         input_ids = encoded["input_ids"].to(self.device)
# #         attention_mask = encoded["attention_mask"].to(self.device)
# #         active_ids = input_ids[0][attention_mask[0].bool()].tolist()
# #         tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
# #         return prompt_text, input_ids, attention_mask, tokens

# #     def prepare_chat_batch(
# #         self,
# #         batch_messages: List[List[Dict]],
# #         add_generation_prompt: bool = True,
# #     ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
# #         prompts: List[str] = []
# #         for messages in batch_messages:
# #             prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))

# #         encoded = self.tokenizer(
# #             prompts,
# #             return_tensors="pt",
# #             padding=True,
# #             add_special_tokens=False,
# #         )
# #         input_ids = encoded["input_ids"].to(self.device)
# #         attention_mask = encoded["attention_mask"].to(self.device)

# #         tokens_batch: List[List[str]] = []
# #         for ids_row, mask_row in zip(input_ids, attention_mask):
# #             active_ids = ids_row[mask_row.bool()].tolist()
# #             tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))

# #         return prompts, input_ids, attention_mask, tokens_batch

# #     def vllm_generate_text_batch(
# #         self,
# #         prompts: List[str],
# #         *,
# #         max_new_tokens: int = 256,
# #         temperature: float = 0.7,
# #         top_p: float = 0.95,
# #     ) -> List[str]:
# #         if not self.vllm_engine:
# #             raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")

# #         sampling_params = SamplingParams(
# #             temperature=temperature,
# #             top_p=top_p,
# #             max_tokens=max_new_tokens,
# #         )
# #         outputs = self.vllm_engine.generate(prompts, sampling_params)
# #         generations = [out.outputs[0].text.strip() for out in outputs]
# #         return generations

# #     def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
# #         input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
# #         output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
# #         if output_embeds is None:
# #             output_embeds = getattr(model, "lm_head", None)

# #         if (
# #             input_embeds is None
# #             or output_embeds is None
# #             or not hasattr(input_embeds, "weight")
# #             or not hasattr(output_embeds, "weight")
# #         ):
# #             raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")

# #         input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
# #         output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

# #         gram = torch.matmul(output_weight.T, output_weight)
# #         reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
# #         gram = gram + reg
# #         rhs = torch.matmul(output_weight.T, input_weight)

# #         realign_matrix = torch.linalg.solve(gram, rhs)
# #         target_norm = input_weight.norm(dim=1).mean().detach()

# #         if self.args.latent_space_realign:
# #             pass
# #         else:
# #             # keep the matrix, for further normalization
# #             realign_matrix = torch.eye(
# #                 realign_matrix.shape[0],
# #                 device=realign_matrix.device,
# #                 dtype=realign_matrix.dtype,
# #             )

# #         return realign_matrix, target_norm

# #     def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
# #         key = id(model)
# #         info = self._latent_realign_matrices.get(key)
# #         target_device = torch.device(device)

# #         if info is None:
# #             matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
# #         else:
# #             matrix, target_norm = info
# #             if matrix.device != target_device:
# #                 matrix = matrix.to(target_device)

# #         if isinstance(target_norm, torch.Tensor):
# #             target_norm = target_norm.to(device=target_device, dtype=matrix.dtype)
# #         else:
# #             target_norm = torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)

# #         self._latent_realign_matrices[key] = (matrix, target_norm)
# #         return matrix, target_norm

# #     def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
# #         matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
# #         hidden_fp32 = hidden.to(torch.float32)
# #         aligned = torch.matmul(hidden_fp32, matrix)

# #         aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
# #         self.pre_aligned = aligned.detach().clone()
# #         aligned = aligned * (target_norm / aligned_norm)
# #         return aligned.to(hidden.dtype)

# #     @torch.no_grad()
# #     def generate_text_batch(
# #         self,
# #         input_ids: torch.Tensor,
# #         attention_mask: Optional[torch.Tensor] = None,
# #         *,
# #         max_new_tokens: int = 256,
# #         temperature: float = 0.7,
# #         top_p: float = 0.95,
# #         past_key_values: Optional[Tuple] = None,
# #     ) -> Tuple[List[str], Optional[Tuple]]:
# #         if input_ids.dim() != 2:
# #             raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

# #         if attention_mask is None:
# #             attention_mask = torch.ones_like(input_ids, device=self.device)

# #         prompt_lengths = attention_mask.sum(dim=1).tolist()

# #         # Extend attention mask if past KV exists
# #         if past_key_values is not None:
# #             past_len = _past_length(past_key_values)
# #             if past_len > 0:
# #                 past_mask = torch.ones(
# #                     (attention_mask.shape[0], past_len),
# #                     dtype=attention_mask.dtype,
# #                     device=attention_mask.device,
# #                 )
# #                 attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

# #         # IMPORTANT: do NOT pass cache_position here
# #         outputs = self.model.generate(
# #             input_ids=input_ids,
# #             attention_mask=attention_mask,
# #             max_new_tokens=max_new_tokens,
# #             temperature=temperature,
# #             top_p=top_p,
# #             do_sample=True,
# #             pad_token_id=self.tokenizer.pad_token_id,
# #             return_dict_in_generate=True,
# #             output_scores=False,
# #             past_key_values=past_key_values,
# #             use_cache=True,
# #         )

# #         sequences = outputs.sequences
# #         generations: List[str] = []

# #         for idx, length in enumerate(prompt_lengths):
# #             length = int(length)
# #             generated_ids = sequences[idx, length:]
# #             text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
# #             generations.append(text)

# #         next_cache = getattr(outputs, "past_key_values", None)
# #         return generations, next_cache

# #     def tokenize_text(self, text: str) -> torch.Tensor:
# #         return self.tokenizer(
# #             text,
# #             add_special_tokens=False,
# #             return_tensors="pt",
# #         )["input_ids"].to(self.device)

# #     @torch.no_grad()
# #     def generate_latent_batch(
# #         self,
# #         input_ids: torch.Tensor,
# #         attention_mask: Optional[torch.Tensor] = None,
# #         *,
# #         latent_steps: int,
# #         past_key_values: Optional[Tuple] = None,
# #     ) -> Tuple:
# #         if input_ids.dim() != 2:
# #             raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

# #         if attention_mask is None:
# #             attention_mask = torch.ones_like(input_ids, device=self.device)
# #         else:
# #             attention_mask = attention_mask.to(self.device)

# #         if past_key_values is not None:
# #             past_len = _past_length(past_key_values)
# #             if past_len > 0:
# #                 past_mask = torch.ones(
# #                     (attention_mask.shape[0], past_len),
# #                     dtype=attention_mask.dtype,
# #                     device=attention_mask.device,
# #                 )
# #                 attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

# #         # outputs = self.model(
# #         #     input_ids=input_ids,
# #         #     attention_mask=attention_mask,
# #         #     past_key_values=past_key_values,
# #         #     use_cache=True,
# #         #     output_hidden_states=True,
# #         #     return_dict=True,
# #         # )
        
# #         outputs = self._latent_forward_step(
# #             latent_embed=latent_embed,
# #             attention_mask=latent_mask,
# #             past_key_values=past,
# #         )

# #         past = outputs.past_key_values

# #         e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
# #         last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]

# #         latent_vecs_all: List[torch.Tensor] = []
# #         latent_vecs_all.append(e_t.detach().clone())

# #         for step in range(latent_steps):
# #             source_model = self.HF_model if hasattr(self, "HF_model") else self.model
# #             latent_vec = self._apply_latent_realignment(last_hidden, source_model)

# #             latent_vecs_all.append(latent_vec.detach().clone())
# #             latent_embed = latent_vec.unsqueeze(1)

# #             past_len = _past_length(past)
# #             latent_mask = torch.ones(
# #                 (latent_embed.shape[0], past_len + 1),
# #                 dtype=torch.long,
# #                 device=latent_embed.device,
# #             )

# #             outputs = self.model(
# #                 inputs_embeds=latent_embed,
# #                 attention_mask=latent_mask,
# #                 past_key_values=past,
# #                 use_cache=True,
# #                 output_hidden_states=True,
# #                 return_dict=True,
# #             )
# #             past = outputs.past_key_values
# #             last_hidden = outputs.hidden_states[-1][:, -1, :]

# #         return past

# #     @torch.no_grad()
# #     def _is_gemma4(self) -> bool:
# #         return "gemma" in self.model_name.lower()


# #     def _latent_forward_step(
# #         self,
# #         latent_embed: torch.Tensor,
# #         attention_mask: torch.Tensor,
# #         past_key_values,
# #     ):
# #         """
# #         One latent autoregressive step.
    
# #         For Gemma4, bypass top-level Gemma4 forward because it rejects arbitrary
# #         inputs_embeds. Use the internal language_model directly.
# #         """
    
# #         past_len = _past_length(past_key_values)
    
# #         cache_position = torch.arange(
# #             past_len,
# #             past_len + latent_embed.shape[1],
# #             dtype=torch.long,
# #             device=latent_embed.device,
# #         )
    
# #         if (
# #             self._is_gemma4()
# #             and hasattr(self.model, "model")
# #             and hasattr(self.model.model, "language_model")
# #         ):
# #             lm = self.model.model.language_model
    
# #             kwargs = dict(
# #                 inputs_embeds=latent_embed,
# #                 attention_mask=attention_mask,
# #                 past_key_values=past_key_values,
# #                 use_cache=True,
# #                 output_hidden_states=True,
# #                 return_dict=True,
# #                 cache_position=cache_position,
# #             )
    
# #             try:
# #                 return lm(**kwargs)
# #             except TypeError:
# #                 kwargs.pop("cache_position", None)
# #                 return lm(**kwargs)
    
# #         kwargs = dict(
# #             inputs_embeds=latent_embed,
# #             attention_mask=attention_mask,
# #             past_key_values=past_key_values,
# #             use_cache=True,
# #             output_hidden_states=True,
# #             return_dict=True,
# #             cache_position=cache_position,
# #         )
    
# #         try:
# #             return self.model(**kwargs)
# #         except TypeError:
# #             kwargs.pop("cache_position", None)
# #             return self.model(**kwargs)#here it ended i have changed
# #     def generate_latent_batch_hidden_state(
# #         self,
# #         input_ids: torch.Tensor,
# #         attention_mask: Optional[torch.Tensor] = None,
# #         *,
# #         latent_steps: int,
# #         past_key_values: Optional[Tuple] = None,
# #     ) -> Tuple:
# #         if input_ids.dim() != 2:
# #             raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

# #         if attention_mask is None:
# #             attention_mask = torch.ones_like(input_ids, device=self.HF_device)
# #         else:
# #             attention_mask = attention_mask.to(self.HF_device)

# #         if past_key_values is not None:
# #             past_len = _past_length(past_key_values)
# #             if past_len > 0:
# #                 past_mask = torch.ones(
# #                     (attention_mask.shape[0], past_len),
# #                     dtype=attention_mask.dtype,
# #                     device=attention_mask.device,
# #                 )
# #                 attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

# #         outputs = self.HF_model(
# #             input_ids=input_ids,
# #             attention_mask=attention_mask,
# #             past_key_values=past_key_values,
# #             use_cache=True,
# #             output_hidden_states=True,
# #             return_dict=True,
# #         )
# #         past = outputs.past_key_values
# #         last_hidden = outputs.hidden_states[-1][:, -1, :]

# #         curr_output_embedding = []
# #         curr_output_embedding.append(outputs.hidden_states[0])  # input embedding

# #         for _ in range(latent_steps):
# #             source_model = self.HF_model if hasattr(self, "HF_model") else self.model
# #             latent_vec = self._apply_latent_realignment(last_hidden, source_model)
# #             latent_embed = latent_vec.unsqueeze(1)

# #             past_len = _past_length(past)
# #             latent_mask = torch.ones(
# #                 (latent_embed.shape[0], past_len + 1),
# #                 dtype=torch.long,
# #                 device=latent_embed.device,
# #             )

# #             outputs = self.HF_model(
# #                 inputs_embeds=latent_embed,
# #                 attention_mask=latent_mask,
# #                 past_key_values=past,
# #                 use_cache=True,
# #                 output_hidden_states=True,
# #                 return_dict=True,
# #             )
# #             past = outputs.past_key_values
# #             last_hidden = outputs.hidden_states[-1][:, -1, :]

# #             curr_output_embedding.append(latent_embed.detach())

# #         return past, torch.cat(curr_output_embedding, dim=1)  # Output input embeddings
        
# import os
# import csv
# import torch
# import matplotlib.pyplot as plt
# from typing import Dict, List, Optional, Tuple
# from transformers import AutoModelForCausalLM, AutoTokenizer

# try:
#     from vllm import LLM, SamplingParams
#     _HAS_VLLM = True
# except ImportError:
#     _HAS_VLLM = False

# try:
#     from transformers.cache_utils import Cache
# except ImportError:
#     Cache = None


# def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
#     if tokenizer.pad_token_id is None:
#         if tokenizer.eos_token is not None:
#             tokenizer.pad_token = tokenizer.eos_token
#         else:
#             tokenizer.add_special_tokens({"pad_token": "<pad>"})


# def _past_length(past_key_values: Optional[Tuple]) -> int:
#     if past_key_values is None:
#         return 0

#     # New HF cache API: DynamicCache / Cache
#     if hasattr(past_key_values, "get_seq_length"):
#         try:
#             return int(past_key_values.get_seq_length())
#         except Exception:
#             pass

#     # Convert new cache to legacy if possible
#     if hasattr(past_key_values, "to_legacy_cache"):
#         try:
#             past_key_values = past_key_values.to_legacy_cache()
#         except Exception:
#             pass

#     # Legacy tuple/list cache
#     if isinstance(past_key_values, (list, tuple)):
#         if len(past_key_values) == 0:
#             return 0
#         k = past_key_values[0][0]
#         return k.shape[-2]

#     return 0


# class ModelWrapper:
#     def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
#         self.model_name = model_name
#         self.device = device
#         self.use_vllm = use_vllm and _HAS_VLLM
#         self.vllm_engine = None
#         self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
#         self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
#         self.args = args

#         self.pre_aligned = None

#         if self.use_vllm:
#             tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
#             gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

#             print(f"[vLLM] Using vLLM backend for model {model_name}")

#             if args.enable_prefix_caching and args.method == "latent_mas":
#                 self.vllm_engine = LLM(
#                     model=model_name,
#                     tensor_parallel_size=tp_size,
#                     gpu_memory_utilization=gpu_util,
#                     enable_prefix_caching=True,
#                     enable_prompt_embeds=True,
#                 )
#             else:
#                 self.vllm_engine = LLM(
#                     model=model_name,
#                     tensor_parallel_size=tp_size,
#                     gpu_memory_utilization=gpu_util,
#                 )

#             self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#             _ensure_pad_token(self.tokenizer)

#             use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False

#             if use_second_hf:
#                 self.HF_model = AutoModelForCausalLM.from_pretrained(
#                     model_name,
#                     dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                     device_map="auto",
#                     low_cpu_mem_usage=True,
#                 ).eval()

#                 self.embedding_layer = self.HF_model.get_input_embeddings()
#                 self.HF_device = args.device2

#                 if self.latent_space_realign:
#                     self._ensure_latent_realign_matrix(
#                         self.HF_model,
#                         torch.device("cpu"),
#                         args,
#                     )

#             elif self.latent_space_realign:
#                 raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")

#             return

#         # Normal HuggingFace path
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#         _ensure_pad_token(self.tokenizer)

#         with torch.no_grad():
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 device_map="auto",
#                 low_cpu_mem_usage=True,
#             )

#         if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
#             self.model.resize_token_embeddings(len(self.tokenizer))

#         # Do NOT call self.model.to(device) when using device_map="auto"
#         self.model.eval()

#         if hasattr(self.model.config, "use_cache"):
#             self.model.config.use_cache = True

#         if self.latent_space_realign:
#             # Build W_a on CPU first to avoid GPU OOM.
#             # It will be moved to hidden.device when actually used.
#             self._ensure_latent_realign_matrix(
#                 self.model,
#                 torch.device("cpu"),
#                 args,
#             )

#     def _is_gemma4(self) -> bool:
#         return "gemma" in self.model_name.lower()

#     def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
#         tpl = getattr(self.tokenizer, "chat_template", None)
#         if tpl:
#             return self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=add_generation_prompt,
#             )

#         segments = []
#         for message in messages:
#             role = message.get("role", "user")
#             content = message.get("content", "")
#             segments.append(f"<|{role}|>\n{content}\n</|{role}|>")

#         if add_generation_prompt:
#             segments.append("<|assistant|>")

#         return "\n".join(segments)

#     def prepare_chat_input(
#         self,
#         messages: List[Dict],
#         add_generation_prompt: bool = True,
#     ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
#         prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)

#         encoded = self.tokenizer(
#             prompt_text,
#             return_tensors="pt",
#             add_special_tokens=False,
#         )

#         input_ids = encoded["input_ids"].to(self.device)
#         attention_mask = encoded["attention_mask"].to(self.device)

#         active_ids = input_ids[0][attention_mask[0].bool()].tolist()
#         tokens = self.tokenizer.convert_ids_to_tokens(active_ids)

#         return prompt_text, input_ids, attention_mask, tokens

#     def prepare_chat_batch(
#         self,
#         batch_messages: List[List[Dict]],
#         add_generation_prompt: bool = True,
#     ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
#         prompts: List[str] = []

#         for messages in batch_messages:
#             prompts.append(
#                 self.render_chat(
#                     messages,
#                     add_generation_prompt=add_generation_prompt,
#                 )
#             )

#         encoded = self.tokenizer(
#             prompts,
#             return_tensors="pt",
#             padding=True,
#             add_special_tokens=False,
#         )

#         input_ids = encoded["input_ids"].to(self.device)
#         attention_mask = encoded["attention_mask"].to(self.device)

#         tokens_batch: List[List[str]] = []
#         for ids_row, mask_row in zip(input_ids, attention_mask):
#             active_ids = ids_row[mask_row.bool()].tolist()
#             tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))

#         return prompts, input_ids, attention_mask, tokens_batch

#     def vllm_generate_text_batch(
#         self,
#         prompts: List[str],
#         *,
#         max_new_tokens: int = 256,
#         temperature: float = 0.7,
#         top_p: float = 0.95,
#     ) -> List[str]:
#         if not self.vllm_engine:
#             raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")

#         sampling_params = SamplingParams(
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_new_tokens,
#         )

#         outputs = self.vllm_engine.generate(prompts, sampling_params)
#         generations = [out.outputs[0].text.strip() for out in outputs]
#         return generations

#     def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
#         input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
#         output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None

#         if output_embeds is None:
#             output_embeds = getattr(model, "lm_head", None)

#         if (
#             input_embeds is None
#             or output_embeds is None
#             or not hasattr(input_embeds, "weight")
#             or not hasattr(output_embeds, "weight")
#         ):
#             raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")

#         input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
#         output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

#         print("W_in shape:", input_weight.shape)
#         print("W_out shape:", output_weight.shape)

#         # W_a = (W_out^T W_out + lambda I)^(-1) W_out^T W_in
#         gram = torch.matmul(output_weight.T, output_weight)
#         reg = 1e-5 * torch.eye(
#             gram.shape[0],
#             device=gram.device,
#             dtype=gram.dtype,
#         )
#         gram = gram + reg

#         rhs = torch.matmul(output_weight.T, input_weight)
#         realign_matrix = torch.linalg.solve(gram, rhs)

#         target_norm = input_weight.norm(dim=1).mean().detach()

#         if not self.args.latent_space_realign:
#             realign_matrix = torch.eye(
#                 realign_matrix.shape[0],
#                 device=realign_matrix.device,
#                 dtype=realign_matrix.dtype,
#             )

#         return realign_matrix, target_norm

#     def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
#         key = id(model)
#         info = self._latent_realign_matrices.get(key)
#         target_device = torch.device(device)

#         if info is None:
#             matrix, target_norm = self._build_latent_realign_matrix(
#                 model,
#                 target_device,
#                 args,
#             )
#         else:
#             matrix, target_norm = info
#             if matrix.device != target_device:
#                 matrix = matrix.to(target_device)

#         if isinstance(target_norm, torch.Tensor):
#             target_norm = target_norm.to(
#                 device=target_device,
#                 dtype=matrix.dtype,
#             )
#         else:
#             target_norm = torch.as_tensor(
#                 target_norm,
#                 device=target_device,
#                 dtype=matrix.dtype,
#             )

#         self._latent_realign_matrices[key] = (matrix, target_norm)
#         return matrix, target_norm

#     def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
#         matrix, target_norm = self._ensure_latent_realign_matrix(
#             model,
#             hidden.device,
#             self.args,
#         )

#         hidden_fp32 = hidden.to(torch.float32)
#         aligned = torch.matmul(hidden_fp32, matrix)

#         aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
#         self.pre_aligned = aligned.detach().clone()

#         aligned = aligned * (target_norm / aligned_norm)
#         return aligned.to(hidden.dtype)

#     def _get_lm_head(self):
#         output_layer = None

#         if hasattr(self.model, "get_output_embeddings"):
#             output_layer = self.model.get_output_embeddings()

#         if output_layer is None and hasattr(self.model, "lm_head"):
#             output_layer = self.model.lm_head

#         if output_layer is None:
#             raise RuntimeError("Could not find output LM head.")

#         return output_layer

#     def _top_p_sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
#         if temperature <= 0:
#             return torch.argmax(logits, dim=-1, keepdim=True)

#         logits = logits / temperature
#         probs = torch.softmax(logits, dim=-1)

#         if top_p is not None and top_p < 1.0:
#             sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

#             sorted_mask = cumulative_probs > top_p
#             sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
#             sorted_mask[..., 0] = False

#             sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
#             sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

#             sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
#             next_token = sorted_indices.gather(-1, sampled_sorted)
#             return next_token

#         return torch.multinomial(probs, num_samples=1)


#     def _latent_forward_step(
#         self,
#         latent_embed: torch.Tensor,
#         attention_mask: torch.Tensor,
#         past_key_values,
#     ):
#         """
#         One latent autoregressive step.

#         For Gemma4, bypass top-level Gemma4 forward because it rejects arbitrary
#         inputs_embeds. Use internal language_model directly.
#         """

#         past_len = _past_length(past_key_values)

#         cache_position = torch.arange(
#             past_len,
#             past_len + latent_embed.shape[1],
#             dtype=torch.long,
#             device=latent_embed.device,
#         )
#         B, L, _ = latent_embed.shape
#         dummy_token_id = self.tokenizer.pad_token_id

#         if (
#             self._is_gemma4()
#             and hasattr(self.model, "model")
#             and hasattr(self.model.model, "language_model")
#         ):
#             lm = self.model.model.language_model

#             kwargs = dict(
#                 input_ids=dummy_input_ids,
#                 inputs_embeds=latent_embed,
#                 attention_mask=attention_mask,
#                 past_key_values=past_key_values,
#                 use_cache=True,
#                 output_hidden_states=True,
#                 return_dict=True,
#                 cache_position=cache_position,
#             )

#             try:
#                 return lm(**kwargs)
#             except TypeError:
#                 kwargs.pop("cache_position", None)
#                 return lm(**kwargs)

#         kwargs = dict(
#             inputs_embeds=latent_embed,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             use_cache=True,
#             output_hidden_states=True,
#             return_dict=True,
#             cache_position=cache_position,
#         )

#         try:
#             return self.model(**kwargs)
#         except TypeError:
#             kwargs.pop("cache_position", None)
#             return self.model(**kwargs)

#     def _gemma4_generate_text_batch(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         *,
#         max_new_tokens: int,
#         temperature: float,
#         top_p: float,
#         past_key_values,
#     ) -> Tuple[List[str], Optional[Tuple]]:
#         """
#         Manual Gemma4 decoding path when we already have latent past_key_values.

#         This avoids top-level generate() issues with latent internal KV cache.
#         """

#         if not (
#             self._is_gemma4()
#             and hasattr(self.model, "model")
#             and hasattr(self.model.model, "language_model")
#         ):
#             raise RuntimeError("Gemma4 manual generation called on non-Gemma4 model.")

#         lm = self.model.model.language_model
#         lm_head = self._get_lm_head()

#         # Extend attention mask with latent past.
#         past_len = _past_length(past_key_values)

#         if past_len > 0:
#             past_mask = torch.ones(
#                 (attention_mask.shape[0], past_len),
#                 dtype=attention_mask.dtype,
#                 device=attention_mask.device,
#             )
#             full_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
#         else:
#             full_attention_mask = attention_mask

#         cache_position = torch.arange(
#             past_len,
#             past_len + input_ids.shape[1],
#             dtype=torch.long,
#             device=input_ids.device,
#         )

#         kwargs = dict(
#             input_ids=input_ids,
#             attention_mask=full_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=True,
#             output_hidden_states=True,
#             return_dict=True,
#             cache_position=cache_position,
#         )

#         try:
#             outputs = lm(**kwargs)
#         except TypeError:
#             kwargs.pop("cache_position", None)
#             outputs = lm(**kwargs)

#         past = outputs.past_key_values
#         hidden = outputs.hidden_states[-1][:, -1, :]
#         logits = lm_head(hidden)
#         next_token = self._top_p_sample(logits, temperature, top_p)

#         generated_tokens = [next_token]

#         cur_attention_mask = torch.cat(
#             [
#                 full_attention_mask,
#                 torch.ones(
#                     (full_attention_mask.shape[0], 1),
#                     dtype=full_attention_mask.dtype,
#                     device=full_attention_mask.device,
#                 ),
#             ],
#             dim=-1,
#         )

#         eos_id = self.tokenizer.eos_token_id

#         for _ in range(max_new_tokens - 1):
#             past_len = _past_length(past)

#             cache_position = torch.arange(
#                 past_len,
#                 past_len + 1,
#                 dtype=torch.long,
#                 device=next_token.device,
#             )

#             kwargs = dict(
#                 input_ids=next_token,
#                 attention_mask=cur_attention_mask,
#                 past_key_values=past,
#                 use_cache=True,
#                 output_hidden_states=True,
#                 return_dict=True,
#                 cache_position=cache_position,
#             )

#             try:
#                 outputs = lm(**kwargs)
#             except TypeError:
#                 kwargs.pop("cache_position", None)
#                 outputs = lm(**kwargs)

#             past = outputs.past_key_values
#             hidden = outputs.hidden_states[-1][:, -1, :]
#             logits = lm_head(hidden)
#             next_token = self._top_p_sample(logits, temperature, top_p)

#             generated_tokens.append(next_token)

#             cur_attention_mask = torch.cat(
#                 [
#                     cur_attention_mask,
#                     torch.ones(
#                         (cur_attention_mask.shape[0], 1),
#                         dtype=cur_attention_mask.dtype,
#                         device=cur_attention_mask.device,
#                     ),
#                 ],
#                 dim=-1,
#             )

#             if eos_id is not None:
#                 if torch.all(next_token.squeeze(-1) == eos_id):
#                     break

#         generated_ids = torch.cat(generated_tokens, dim=1)

#         generations: List[str] = []
#         for row in generated_ids:
#             text = self.tokenizer.decode(
#                 row.tolist(),
#                 skip_special_tokens=True,
#             ).strip()
#             generations.append(text)

#         return generations, past

#     @torch.no_grad()
#     def generate_text_batch(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         *,
#         max_new_tokens: int = 256,
#         temperature: float = 0.7,
#         top_p: float = 0.95,
#         past_key_values: Optional[Tuple] = None,
#     ) -> Tuple[List[str], Optional[Tuple]]:
#         if input_ids.dim() != 2:
#             raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, device=input_ids.device)
#         else:
#             attention_mask = attention_mask.to(input_ids.device)

#         # If Gemma4 has latent past, use manual internal decoding.
#         if self._is_gemma4() and past_key_values is not None:
#             return self._gemma4_generate_text_batch(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 past_key_values=past_key_values,
#             )

#         prompt_lengths = attention_mask.sum(dim=1).tolist()

#         if past_key_values is not None:
#             past_len = _past_length(past_key_values)
#             if past_len > 0:
#                 past_mask = torch.ones(
#                     (attention_mask.shape[0], past_len),
#                     dtype=attention_mask.dtype,
#                     device=attention_mask.device,
#                 )
#                 attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

#         outputs = self.model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=True,
#             pad_token_id=self.tokenizer.pad_token_id,
#             return_dict_in_generate=True,
#             output_scores=False,
#             past_key_values=past_key_values,
#             use_cache=True,
#         )

#         sequences = outputs.sequences
#         generations: List[str] = []

#         for idx, length in enumerate(prompt_lengths):
#             length = int(length)
#             generated_ids = sequences[idx, length:]
#             text = self.tokenizer.decode(
#                 generated_ids,
#                 skip_special_tokens=True,
#             ).strip()
#             generations.append(text)

#         next_cache = getattr(outputs, "past_key_values", None)
#         return generations, next_cache

#     def tokenize_text(self, text: str) -> torch.Tensor:
#         return self.tokenizer(
#             text,
#             add_special_tokens=False,
#             return_tensors="pt",
#         )["input_ids"].to(self.device)

#     @torch.no_grad()
#     def generate_latent_batch(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         *,
#         latent_steps: int,
#         past_key_values: Optional[Tuple] = None,
#     ) -> Tuple:
#         if input_ids.dim() != 2:
#             raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, device=input_ids.device)
#         else:
#             attention_mask = attention_mask.to(input_ids.device)

#         if past_key_values is not None:
#             past_len = _past_length(past_key_values)
#             if past_len > 0:
#                 past_mask = torch.ones(
#                     (attention_mask.shape[0], past_len),
#                     dtype=attention_mask.dtype,
#                     device=attention_mask.device,
#                 )
#                 attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

#         # STEP 1:
#         # Normal prompt forward with real tokens.
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             use_cache=True,
#             output_hidden_states=True,
#             return_dict=True,
#         )

#         past = outputs.past_key_values

#         e_t = outputs.hidden_states[0][:, -1, :]
#         last_hidden = outputs.hidden_states[-1][:, -1, :]

#         latent_vecs_all: List[torch.Tensor] = []
#         latent_vecs_all.append(e_t.detach().clone())

#         # STEP 2:
#         # Latent autoregression.
#         for step in range(latent_steps):
#             source_model = self.HF_model if hasattr(self, "HF_model") else self.model

#             # h_t -> W_a -> e_{t+1}
#             latent_vec = self._apply_latent_realignment(
#                 last_hidden,
#                 source_model,
#             )

#             latent_vecs_all.append(latent_vec.detach().clone())

#             print(f"\n[Latent Step {step}]")
#             print("last_hidden shape:", last_hidden.shape)
#             print("latent_vec shape:", latent_vec.shape)
#             print("latent_vec norm:", latent_vec.norm().item())

#             latent_embed = latent_vec.unsqueeze(1)

#             past_len = _past_length(past)

#             latent_mask = torch.ones(
#                 (latent_embed.shape[0], past_len + 1),
#                 dtype=torch.long,
#                 device=latent_embed.device,
#             )

#             # For Gemma4, this bypasses top-level forward.
#             # For Qwen/Llama, this uses normal model forward.
#             outputs = self._latent_forward_step(
#                 latent_embed=latent_embed,
#                 attention_mask=latent_mask,
#                 past_key_values=past,
#             )

#             past = outputs.past_key_values
#             last_hidden = outputs.hidden_states[-1][:, -1, :]

#         return past

#     @torch.no_grad()
#     def generate_latent_batch_hidden_state(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         *,
#         latent_steps: int,
#         past_key_values: Optional[Tuple] = None,
#     ) -> Tuple:
#         if input_ids.dim() != 2:
#             raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

#         if not hasattr(self, "HF_model"):
#             # Fallback to normal latent path if no second HF model exists.
#             past = self.generate_latent_batch(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 latent_steps=latent_steps,
#                 past_key_values=past_key_values,
#             )
#             return past, None

#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, device=input_ids.device)
#         else:
#             attention_mask = attention_mask.to(input_ids.device)

#         if past_key_values is not None:
#             past_len = _past_length(past_key_values)
#             if past_len > 0:
#                 past_mask = torch.ones(
#                     (attention_mask.shape[0], past_len),
#                     dtype=attention_mask.dtype,
#                     device=attention_mask.device,
#                 )
#                 attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

#         outputs = self.HF_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             use_cache=True,
#             output_hidden_states=True,
#             return_dict=True,
#         )

#         past = outputs.past_key_values
#         last_hidden = outputs.hidden_states[-1][:, -1, :]

#         curr_output_embedding = []
#         curr_output_embedding.append(outputs.hidden_states[0])

#         for step in range(latent_steps):
#             source_model = self.HF_model

#             latent_vec = self._apply_latent_realignment(
#                 last_hidden,
#                 source_model,
#             )

#             latent_embed = latent_vec.unsqueeze(1)

#             past_len = _past_length(past)

#             latent_mask = torch.ones(
#                 (latent_embed.shape[0], past_len + 1),
#                 dtype=torch.long,
#                 device=latent_embed.device,
#             )

#             outputs = self._latent_forward_step(
#                 latent_embed=latent_embed,
#                 attention_mask=latent_mask,
#                 past_key_values=past,
#             )

#             past = outputs.past_key_values
#             last_hidden = outputs.hidden_states[-1][:, -1, :]

#             curr_output_embedding.append(latent_embed.detach())

#         return past, torch.cat(curr_output_embedding, dim=1)
import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if past_key_values is None:
        return 0

    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass

    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            past_key_values = past_key_values.to_legacy_cache()
        except Exception:
            pass

    if isinstance(past_key_values, (list, tuple)):
        if len(past_key_values) == 0:
            return 0
        k = past_key_values[0][0]
        return k.shape[-2]

    return 0


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        use_vllm: bool = False,
        args=None,
    ):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args
        self.pre_aligned = None

        if self.use_vllm:
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")

            if args.enable_prefix_caching and args.method == "latent_mas":
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True,
                    enable_prompt_embeds=True,
                )
            else:
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            _ensure_pad_token(self.tokenizer)

            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False

            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                ).eval()

                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2

                if self.latent_space_realign:
                    self._ensure_latent_realign_matrix(
                        self.HF_model,
                        torch.device("cpu"),
                        args,
                    )

            elif self.latent_space_realign:
                raise ValueError(
                    "latent_space_realign requires --use_second_HF_model when using vLLM backend."
                )

            return

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)

        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(
                self.model,
                torch.device("cpu"),
                args,
            )

    def _is_gemma4(self) -> bool:
        return "gemma" in self.model_name.lower()

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")

        if add_generation_prompt:
            segments.append("<|assistant|>")

        return "\n".join(segments)

    def prepare_chat_input(
        self,
        messages: List[Dict],
        add_generation_prompt: bool = True,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)

        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []

        for messages in batch_messages:
            prompts.append(
                self.render_chat(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                )
            )

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))

        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None

        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)

        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")

        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

        print("W_in shape:", input_weight.shape)
        print("W_out shape:", output_weight.shape)
        print("Wa memory:",realign_matrix.numel() *realign_matrix.element_size() / 1024**2, "MB")


        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(
            gram.shape[0],
            device=gram.device,
            dtype=gram.dtype,
        )
        gram = gram + reg

        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)

        target_norm = input_weight.norm(dim=1).mean().detach()

        if not self.args.latent_space_realign:
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(
                model,
                target_device,
                args,
            )
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        if isinstance(target_norm, torch.Tensor):
            target_norm = target_norm.to(
                device=target_device,
                dtype=matrix.dtype,
            )
        else:
            target_norm = torch.as_tensor(
                target_norm,
                device=target_device,
                dtype=matrix.dtype,
            )

        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(
            model,
            hidden.device,
            self.args,
        )

        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.pre_aligned = aligned.detach().clone()

        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def _get_lm_head(self):
        output_layer = None

        if hasattr(self.model, "get_output_embeddings"):
            output_layer = self.model.get_output_embeddings()

        if output_layer is None and hasattr(self.model, "lm_head"):
            output_layer = self.model.lm_head

        if output_layer is None:
            raise RuntimeError("Could not find output LM head.")

        return output_layer

    def _lm_head_device(self):
        lm_head = self._get_lm_head()
        try:
            return next(lm_head.parameters()).device
        except StopIteration:
            return self.device

    def _top_p_sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_sorted)
            return next_token

        return torch.multinomial(probs, num_samples=1)

    def _latent_forward_step(
        self,
        latent_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
    ):
        """
        One latent autoregressive step.
        For Gemma4:
        - Do NOT pass input_ids + inputs_embeds together.
        - Do NOT pass inputs_embeds alone without per_layer_inputs.
        - Correct path is inputs_embeds + per_layer_inputs.

        For Qwen/Llama:
        - Standard inputs_embeds path is used.
        """
        past_len = _past_length(past_key_values)

        cache_position = torch.arange(
            past_len,
            past_len + latent_embed.shape[1],
            dtype=torch.long,
            device=latent_embed.device,
        )
        #====================================
        #Gemma4 path
        #========================

        if (
            self._is_gemma4()
            and hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
        ):
            lm = self.model.model.language_model

            # ==================================================
            # TRUE LATENT PLE
            # Derive the per-layer inputs directly from the latent
            # embedding (content-aware) instead of from dummy token ids.
            # ==================================================
            with torch.no_grad():
                per_layer_inputs = lm.project_per_layer_inputs(
                    latent_embed,
                    per_layer_inputs=None,
                )

            print("Gemma4 TRUE latent PLE:", per_layer_inputs.shape)

            kwargs = dict(
                inputs_embeds=latent_embed,
                per_layer_inputs=per_layer_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )

            try:
                return lm(**kwargs)
            except TypeError:
                kwargs.pop("cache_position", None)
                return lm(**kwargs)

        kwargs = dict(
            inputs_embeds=latent_embed,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        try:
            return self.model(**kwargs)
        except TypeError:
            kwargs.pop("cache_position", None)
            return self.model(**kwargs)

    def _gemma4_generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        past_key_values,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if not (
            self._is_gemma4()
            and hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
        ):
            raise RuntimeError("Gemma4 manual generation called on non-Gemma4 model.")

        lm = self.model.model.language_model
        lm_head = self._get_lm_head()
        lm_head_device = self._lm_head_device()
        model_input_device = input_ids.device

        past_len = _past_length(past_key_values)

        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        else:
            full_attention_mask = attention_mask

        cache_position = torch.arange(
            past_len,
            past_len + input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )

        kwargs = dict(
            input_ids=input_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        try:
            outputs = lm(**kwargs)
        except TypeError:
            kwargs.pop("cache_position", None)
            outputs = lm(**kwargs)

        past = outputs.past_key_values
        hidden = outputs.hidden_states[-1][:, -1, :].to(lm_head_device)
        logits = lm_head(hidden)
        next_token = self._top_p_sample(logits, temperature, top_p).to(model_input_device)

        generated_tokens = [next_token]

        cur_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones(
                    (full_attention_mask.shape[0], 1),
                    dtype=full_attention_mask.dtype,
                    device=full_attention_mask.device,
                ),
            ],
            dim=-1,
        )

        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            past_len = _past_length(past)

            cache_position = torch.arange(
                past_len,
                past_len + 1,
                dtype=torch.long,
                device=next_token.device,
            )

            kwargs = dict(
                input_ids=next_token,
                attention_mask=cur_attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )

            try:
                outputs = lm(**kwargs)
            except TypeError:
                kwargs.pop("cache_position", None)
                outputs = lm(**kwargs)

            past = outputs.past_key_values
            hidden = outputs.hidden_states[-1][:, -1, :].to(lm_head_device)
            logits = lm_head(hidden)
            next_token = self._top_p_sample(logits, temperature, top_p).to(model_input_device)

            generated_tokens.append(next_token)

            cur_attention_mask = torch.cat(
                [
                    cur_attention_mask,
                    torch.ones(
                        (cur_attention_mask.shape[0], 1),
                        dtype=cur_attention_mask.dtype,
                        device=cur_attention_mask.device,
                    ),
                ],
                dim=-1,
            )

            if eos_id is not None and torch.all(next_token.squeeze(-1) == eos_id):
                break

        generated_ids = torch.cat(generated_tokens, dim=1)

        generations: List[str] = []
        for row in generated_ids:
            text = self.tokenizer.decode(
                row.tolist(),
                skip_special_tokens=True,
            ).strip()
            generations.append(text)

        return generations, past

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device)

        if self._is_gemma4() and past_key_values is not None:
            return self._gemma4_generate_text_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                past_key_values=past_key_values,
            )

        prompt_lengths = attention_mask.sum(dim=1).tolist()

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            use_cache=True,
        )

        sequences = outputs.sequences
        generations: List[str] = []

        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            generations.append(text)

        next_cache = getattr(outputs, "past_key_values", None)
        return generations, next_cache

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model

            latent_vec = self._apply_latent_realignment(
                last_hidden,
                source_model,
            )

            latent_vecs_all.append(latent_vec.detach().clone())

            print(f"\n[Latent Step {step}]")
            print("last_hidden shape:", last_hidden.shape)
            print("latent_vec shape:", latent_vec.shape)
            print("latent_vec norm:", latent_vec.norm().item())

            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)

            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )

            outputs = self._latent_forward_step(
                latent_embed=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
            )

            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if not hasattr(self, "HF_model"):
            past = self.generate_latent_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_steps=latent_steps,
                past_key_values=past_key_values,
            )
            return past, None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        curr_output_embedding = []
        curr_output_embedding.append(outputs.hidden_states[0])

        for step in range(latent_steps):
            source_model = self.HF_model

            latent_vec = self._apply_latent_realignment(
                last_hidden,
                source_model,
            )

            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)

            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )

            outputs = self._latent_forward_step(
                latent_embed=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
            )

            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1)
