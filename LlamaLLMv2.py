#!/usr/bin/env python3

import ctypes

llama_token = ctypes.c_int
llama_token_p = ctypes.POINTER(llama_token)
llama_context_p = ctypes.c_void_p

llama_progress_callback = ctypes.c_void_p

class llama_token_data(ctypes.Structure):
	_fields_ = [
		('id', llama_token),
		('p', ctypes.c_float),
		('plog', ctypes.c_float),
	]

class llama_context_params(ctypes.Structure):
	_fields_ = [
		('n_ctx', ctypes.c_int),
		('n_parts', ctypes.c_int),
		('seed', ctypes.c_int),
		('f16_kv', ctypes.c_bool),
		('logits_all', ctypes.c_bool),
		('vocab_only', ctypes.c_bool),
		('use_mlock', ctypes.c_bool),
		('embedding', ctypes.c_bool),
		('progress_callback', llama_progress_callback),
		('progress_callback_user_data', ctypes.c_void_p)
	]


class LLaMAContext(object):
	def __init__(self, ctx, llama):
		super().__init__()

		self._llama = llama
		self._ctx = ctx
		self._token_buffer = (llama_token * 1024)()

	def __del__(self):
		self.close()

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.close()

	def close(self):
		if self._ctx is not None:
			self._llama.llama_free(self._ctx)

		self._ctx = None
		self._llama = None

	def llama_token_bos(self):
		return int(self._llama.llama_token_bos())

	def llama_token_eos(self):
		return int(self._llama.llama_token_eos())

	def llama_n_vocab(self):
		return int(self._llama.llama_n_vocab(self._ctx))

	def llama_n_ctx(self):
		return int(self._llama.llama_n_ctx(self._ctx))

	def llama_token_to_str(self, token):
		return self._llama.llama_token_to_str(self._ctx, token)

	def llama_tokenize(self, bytes, add_bos=False):
		tokens = self._token_buffer
		token_count = self._llama.llama_tokenize(self._ctx, bytes, tokens, len(tokens), add_bos)

		if token_count < 0:
			tokens = (llama_token * abs(token_count))()
			token_count = self._llama.llama_tokenize(self._ctx, bytes, tokens, len(tokens), add_bos)
		
		return tokens[:token_count]

	def llama_eval(self, tokens, n_past, n_threads=4):
		if len(tokens) == 0:
			return

		tokens = (llama_token * len(tokens))(*tokens)
		if self._llama.llama_eval(self._ctx, tokens, len(tokens), n_past, n_threads) != 0:
			raise RuntimeException('eval failed')

	def llama_sample_top_p_top_k(self, top_k=40, top_p=0.95, temp=0.80, last_n_tokens=None, repeat_penalty=1.10):
		if last_n_tokens is None:
			last_n_tokens = []

		last_n_tokens_size = len(last_n_tokens)
		last_n_tokens = (llama_token * last_n_tokens_size)(*last_n_tokens)
		return self._llama.llama_sample_top_p_top_k(self._ctx, last_n_tokens, last_n_tokens_size, top_k, top_p, temp, repeat_penalty)


class LLaMA(object):
	def __init__(self, library):
		super().__init__()

		self._llama = ctypes.cdll.LoadLibrary(library)

		self._llama.llama_context_default_params.restype = llama_context_params

		self._llama.llama_init_from_file.argtypes = [ctypes.c_char_p, llama_context_params]
		self._llama.llama_init_from_file.restype = llama_context_p

		self._llama.llama_free.argtypes = [llama_context_p]
		self._llama.llama_free.restype = None

		self._llama.llama_eval.argtypes = [llama_context_p, llama_token_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
		self._llama.llama_eval.restype = ctypes.c_int

		self._llama.llama_tokenize.argtypes = [llama_context_p, ctypes.c_char_p, llama_token_p, ctypes.c_int, ctypes.c_bool]
		self._llama.llama_tokenize.restype = ctypes.c_int

		self._llama.llama_n_vocab.argtypes = [llama_context_p]
		self._llama.llama_n_vocab.restype = ctypes.c_int

		self._llama.llama_n_ctx.argtypes = [llama_context_p]
		self._llama.llama_n_ctx.restype = ctypes.c_int

		self._llama.llama_get_logits.argtypes = [llama_context_p]
		self._llama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

		self._llama.llama_token_to_str.argtypes = [llama_context_p, llama_token]
		self._llama.llama_token_to_str.restype = ctypes.c_char_p

		self._llama.llama_token_bos.restype = llama_token

		self._llama.llama_token_eos.restype = llama_token

		self._llama.llama_sample_top_p_top_k.argtypes = [llama_context_p, llama_token_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
		self._llama.llama_sample_top_p_top_k.restype = llama_token

	def llama_context_default_params(self):
		return self._llama.llama_context_default_params()

	def llama_init_from_file(self, model_path, parameters=None):
		if parameters is None:
			parameters = self.llama_context_default_params()

		ctx = self._llama.llama_init_from_file(model_path.encode('utf-8'), parameters)
		if ctx == 0:
			raise RuntimeException('could not initialize')

		return LLaMAContext(ctx, self._llama)


class LLaMAGenerator(object):
	def __init__(self, llama_lib, path, parameters=None, n_threads=4, batch_size=128, last_n_tokens=64, top_k=40, top_p=0.95, temperature=0.80):
		super().__init__()

		if parameters is None:
			parameters = llama.llama_context_default_params()

		self._llama = None
		self._llama_lib = llama_lib
		self._path = path
		self._parameters = parameters
		self._ctx = None
		self._batch_size = batch_size
		self._n_threads = int(n_threads)
		self._stop_patterns = []
		self._initial_prompt_length = 0
		self._old_tokens = []
		self._last_n_tokens = last_n_tokens
		self._temperature = temperature
		self._top_k = top_k
		self._top_p = top_p

	def __del__(self):
		self.close()

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.close()

	def close(self):
		if self._ctx is not None:
			self._ctx.close()

		self._ctx = None
		self._llama = None

	def window_size(self):
		if self._ctx:
			return self._ctx.llama_n_ctx() - self._initial_prompt_length
		return 0

	def remaining_tokens(self):
		if self._ctx:
			return max(0, self._ctx.llama_n_ctx() - len(self._old_tokens))
		return 0

	def set_stop_patterns(self, stop_patterns):
		self._stop_patterns = []
		if stop_patterns:
			self._stop_patterns = [p.encode('utf-8') for p in stop_patterns]

	def current_state(self):
		if self._ctx:
			output = b''
			for token in self._old_tokens:
				output += self._ctx.llama_token_to_str(token)
			return output.decode('utf-8')
		return ''

	def tokens_to_string(self, tokens):
		if self._ctx:
			output = b''
			for token in tokens:
				output += self._ctx.llama_token_to_str(token)
			return output.decode('utf-8')
		return ''

	def current_history_tokens(self):
		if self._ctx:
			return self._old_tokens[self._initial_prompt_length:]
		return []

	def current_history(self):
		return self.tokens_to_string(self.current_history_tokens())

	def initialize(self, prompt=None):
		if self._ctx:
			self._ctx.close()

		if self._llama is None:
			self._llama = LLaMA(self._llama_lib)
		self._ctx = self._llama.llama_init_from_file(self._path, self._parameters)

		self._old_tokens = []
		self._initial_prompt_length = 0
		if prompt:
			prompt_tokens = self._ctx.llama_tokenize(b' ' + prompt.encode('utf-8'), True)
			if len(prompt_tokens) > self._ctx.llama_n_ctx():
				self._ctx.close()
				self._ctx = None
				return

			self._initial_prompt_length = len(prompt_tokens)
			self._process_token_batch(prompt_tokens)

	def _process_token_batch(self, prompt):
		prompt = prompt[:min(len(prompt), self.remaining_tokens())]
		while prompt:
			batch = prompt[:self._batch_size]
			prompt = prompt[self._batch_size:]

			self._ctx.llama_eval(batch, len(self._old_tokens), n_threads=self._n_threads)
			self._old_tokens += batch

	def generate(self, prompt, max_length=256):
		llama_eos = self._ctx.llama_token_eos()
		need_bos = False
		if len(self._old_tokens) == 0:
			need_bos = True
			prompt = ' ' + prompt

		self._process_token_batch(self._ctx.llama_tokenize(prompt.encode('utf-8'), need_bos))

		output = b''
		for i in range(min(self._ctx.llama_n_ctx() - len(self._old_tokens), max_length)):
			cur_token = self._ctx.llama_sample_top_p_top_k(self._top_k, self._top_p, self._temperature, self._old_tokens[-self._last_n_tokens:])
			if cur_token == llama_eos:
				return

			self._ctx.llama_eval([cur_token], len(self._old_tokens), n_threads=self._n_threads)

			self._old_tokens.append(cur_token)
			output += self._ctx.llama_token_to_str(cur_token)
			yield self._ctx.llama_token_to_str(cur_token)

			for s in self._stop_patterns:
				if output.endswith(s):
					return

	def batch_generate(self, prompt, max_length=256):
		llama_eos = self._ctx.llama_token_eos()
		need_bos = False
		if len(self._old_tokens) == 0:
			need_bos = True
			prompt = ' ' + prompt

		self._process_token_batch(self._ctx.llama_tokenize(prompt.encode('utf-8'), need_bos))

		output = b''
		for i in range(min(self._ctx.llama_n_ctx() - len(self._old_tokens), max_length)):
			cur_token = self._ctx.llama_sample_top_p_top_k(self._top_k, self._top_p, self._temperature, self._old_tokens[-self._last_n_tokens:])
			if cur_token == llama_eos:
				return output.decode('utf-8')

			self._ctx.llama_eval([cur_token], len(self._old_tokens), n_threads=self._n_threads)

			self._old_tokens.append(cur_token)
			output += self._ctx.llama_token_to_str(cur_token)

			for s in self._stop_patterns:
				if output.endswith(s):
					return output.decode('utf-8')

		return output.decode('utf-8')

	def reset(self):
		if self._ctx is None:
			return

		self._ctx.close()
		self._llama = None
		prompt_tokens = self._old_tokens[:self._initial_prompt_length]
		self._llama = LLaMA(self._llama_lib)
		self._ctx = self._llama.llama_init_from_file(self._path, self._parameters)
		self._old_tokens = []
		self._process_token_batch(prompt_tokens)