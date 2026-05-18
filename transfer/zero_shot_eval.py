"""Zero-shot NER evaluation for RQ2.

Proposal target:
- Train an English BiLSTM-CRF NER model on CoNLL-2003.
- Evaluate zero-shot on isiZulu/Sepedi/Setswana MasakhaNER using *aligned* embeddings.

This module intentionally keeps the interface simple:
- Training consumes English token vectors (FastText) from an English corpus model.
- Evaluation consumes *already-aligned* token vectors for target-language sentences.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch

		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	except Exception:
		pass


def l2_normalize(v: np.ndarray) -> np.ndarray:
	denom = np.linalg.norm(v) + 1e-8
	return (v / denom).astype(np.float32)


@dataclass(frozen=True)
class Sentence:
	tokens: List[str]
	tags: List[str]


def read_conll_sentences(
	path: Path,
	*,
	token_col: int = 0,
	tag_col: int = -1,
	docstart_prefix: str = "-DOCSTART-",
) -> List[Sentence]:
	"""Read a CoNLL-style token-per-line file.

	- Sentence boundary: blank line.
	- Columns separated by whitespace (tabs/spaces).
	- token_col is usually 0
	- tag_col is usually last column.
	"""
	sentences: List[Sentence] = []
	cur_tokens: List[str] = []
	cur_tags: List[str] = []

	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for raw in f:
			line = raw.strip()
			if not line:
				if cur_tokens:
					sentences.append(Sentence(tokens=cur_tokens, tags=cur_tags))
					cur_tokens, cur_tags = [], []
				continue

			if docstart_prefix and line.startswith(docstart_prefix):
				continue

			parts = line.split()
			if len(parts) < 2:
				continue

			token = parts[token_col]
			tag = parts[tag_col]
			cur_tokens.append(token)
			cur_tags.append(tag)

	if cur_tokens:
		sentences.append(Sentence(tokens=cur_tokens, tags=cur_tags))
	return sentences


def _find_split_file(root: Path, split: str) -> Optional[Path]:
	"""Best-effort discovery of split files in a dataset folder."""
	if not root.exists():
		return None

	preferred_names: Dict[str, List[str]] = {
		"train": ["train.txt", "train.conll", "train.tsv"],
		"dev": ["dev.txt", "valid.txt", "validation.txt", "dev.conll", "valid.conll"],
		"test": ["test.txt", "test.conll", "test.tsv"],
	}
	for name in preferred_names.get(split, []):
		p = root / name
		if p.exists():
			return p

	# Fallback: search by keyword.
	keywords: Dict[str, List[str]] = {
		"train": ["train"],
		"dev": ["dev", "valid", "validation"],
		"test": ["test"],
	}
	exts = {".txt", ".conll", ".tsv", ".iob", ".bio"}

	candidates: List[Path] = []
	for p in root.rglob("*"):
		if not p.is_file():
			continue
		if p.suffix.lower() not in exts:
			continue
		name = p.name.lower()
		if any(k in name for k in keywords.get(split, [split])):
			candidates.append(p)

	if not candidates:
		return None

	# Prefer files closer to root and shorter names.
	candidates.sort(key=lambda p: (len(p.parts), len(p.name)))
	return candidates[0]


def find_conll2003_splits(conll_root: Path) -> Dict[str, Path]:
	"""Return paths for train/dev/test, where available."""
	splits: Dict[str, Path] = {}
	for split in ["train", "dev", "test"]:
		p = _find_split_file(conll_root, split)
		if p is not None:
			splits[split] = p
	return splits


def normalize_tags_to_training_space(
	tags: List[str],
	*,
	allowed_entity_types: Sequence[str],
	unknown_type_strategy: str = "to_misc",
) -> List[str]:
	"""Normalize tags from an eval dataset to the English training tag space.

	CoNLL-2003 types: PER, ORG, LOC, MISC.

	If the eval dataset contains extra entity types (e.g., DATE), we either:
	- map them to MISC (default) to keep them as entities, or
	- drop them to O.
	"""
	allowed = set(allowed_entity_types)
	out: List[str] = []
	for tag in tags:
		if tag == "O":
			out.append(tag)
			continue
		if "-" not in tag:
			out.append("O")
			continue
		prefix, ent = tag.split("-", 1)
		ent_norm = ent.upper()
		if ent_norm in allowed:
			out.append(f"{prefix}-{ent_norm}")
			continue

		if unknown_type_strategy == "to_o":
			out.append("O")
		elif unknown_type_strategy == "to_misc" and "MISC" in allowed:
			out.append(f"{prefix}-MISC")
		else:
			out.append("O")
	return out


# ------------------------
# BiLSTM-CRF (PyTorch)
# ------------------------


def _require_torch():
	import torch  # noqa: F401
	from torch import nn  # noqa: F401
	try:
		from torchcrf import CRF  # noqa: F401
	except Exception:
		from TorchCRF import CRF  # noqa: F401


class BiLSTMCRF:  # small wrapper to avoid exposing torch types at import time
	def __init__(
		self,
		*,
		embedding_dim: int,
		hidden_dim: int,
		num_tags: int,
		dropout: float = 0.2,
	):
		_require_torch()
		import torch
		from torch import nn

		# CRF dependency compatibility:
		# - `pytorch-crf` provides `torchcrf.CRF(num_tags, batch_first=True)`
		# - `TorchCRF` provides `TorchCRF.CRF(num_labels, pad_idx=None, use_gpu=True)`
		try:
			from torchcrf import CRF as _CRF

			crf_impl = "torchcrf"
		except Exception:
			from TorchCRF import CRF as _CRF

			crf_impl = "TorchCRF"

		class _Model(nn.Module):
			def __init__(self):
				super().__init__()
				self.lstm = nn.LSTM(
					input_size=embedding_dim,
					hidden_size=hidden_dim // 2,
					num_layers=1,
					batch_first=True,
					bidirectional=True,
				)
				self.dropout = nn.Dropout(dropout)
				self.fc = nn.Linear(hidden_dim, num_tags)
				if crf_impl == "torchcrf":
					self.crf = _CRF(num_tags, batch_first=True)
				else:
					self.crf = _CRF(num_labels=num_tags, pad_idx=None, use_gpu=True)
				self._crf_impl = crf_impl

			def forward(self, x, mask):
				out, _ = self.lstm(x)
				out = self.dropout(out)
				emissions = self.fc(out)
				return emissions

		self.torch = torch
		self.model = _Model()

	def to(self, device: str):
		self.model.to(device)
		return self

	def state_dict(self):
		return self.model.state_dict()

	def load_state_dict(self, state):
		return self.model.load_state_dict(state)

	def train(self):
		self.model.train()

	def eval(self):
		self.model.eval()

	def neg_log_likelihood(self, x, tags, mask):
		emissions = self.model(x, mask)
		# torchcrf returns log likelihood; we minimize negative log likelihood
		if getattr(self.model, "_crf_impl", "torchcrf") == "torchcrf":
			ll = self.model.crf(emissions, tags, mask=mask, reduction="mean")
			return -ll
		ll = self.model.crf(emissions, tags, mask)
		return -ll.mean()

	def decode(self, x, mask):
		emissions = self.model(x, mask)
		if getattr(self.model, "_crf_impl", "torchcrf") == "torchcrf":
			return self.model.crf.decode(emissions, mask=mask)
		return self.model.crf.viterbi_decode(emissions, mask)


def build_tag_vocab(sentences: Sequence[Sentence]) -> Tuple[Dict[str, int], List[str]]:
	tags = sorted({t for s in sentences for t in s.tags})
	if "O" in tags:
		tags.remove("O")
		tags = ["O"] + tags
	tag_to_idx = {t: i for i, t in enumerate(tags)}
	return tag_to_idx, tags


def _pad_batch(
	batch_vecs: List[np.ndarray],
	batch_tags: List[List[int]],
):
	_require_torch()
	import torch

	lengths = [v.shape[0] for v in batch_vecs]
	max_len = max(lengths)
	dim = batch_vecs[0].shape[1]

	x = torch.zeros((len(batch_vecs), max_len, dim), dtype=torch.float32)
	y = torch.zeros((len(batch_vecs), max_len), dtype=torch.long)
	mask = torch.zeros((len(batch_vecs), max_len), dtype=torch.bool)

	for i, (vecs, tags) in enumerate(zip(batch_vecs, batch_tags)):
		L = vecs.shape[0]
		x[i, :L] = torch.from_numpy(vecs)
		y[i, :L] = torch.tensor(tags, dtype=torch.long)
		mask[i, :L] = True

	return x, y, mask


def _iter_minibatches(
	vecs: List[np.ndarray],
	tags: List[List[int]],
	*,
	batch_size: int,
	shuffle: bool,
	seed: int,
):
	idx = list(range(len(vecs)))
	if shuffle:
		rng = random.Random(seed)
		rng.shuffle(idx)
	for start in range(0, len(idx), batch_size):
		batch_idx = idx[start : start + batch_size]
		yield [vecs[i] for i in batch_idx], [tags[i] for i in batch_idx]


def _embed_sentence_fasttext(ft_model, tokens: List[str], cache: Dict[str, np.ndarray]) -> np.ndarray:
	vecs = []
	for tok in tokens:
		if tok in cache:
			v = cache[tok]
		else:
			v = l2_normalize(ft_model.get_word_vector(tok))
			cache[tok] = v
		vecs.append(v)
	return np.stack(vecs, axis=0).astype(np.float32)


def train_or_load_english_ner_model(
	*,
	conll_root: Path,
	english_fasttext_bin: Path,
	out_path: Path,
	embedding_dim: int,
	hidden_dim: int = 256,
	dropout: float = 0.2,
	epochs: int = 5,
	batch_size: int = 16,
	lr: float = 1e-3,
	seed: int = 42,
	force_retrain: bool = False,
	device: str = "cpu",
):
	"""Train (or load) an English BiLSTM-CRF on CoNLL-2003."""
	_set_seed(seed)
	_require_torch()
	import torch

	out_path.parent.mkdir(parents=True, exist_ok=True)

	if out_path.exists() and not force_retrain:
		ckpt = torch.load(out_path, map_location="cpu")
		tag_to_idx = ckpt["tag_to_idx"]
		idx_to_tag = ckpt["idx_to_tag"]
		model = BiLSTMCRF(
			embedding_dim=ckpt["embedding_dim"],
			hidden_dim=ckpt["hidden_dim"],
			num_tags=len(idx_to_tag),
			dropout=ckpt.get("dropout", dropout),
		)
		model.load_state_dict(ckpt["state_dict"])
		model.to(device)
		return model, tag_to_idx, idx_to_tag

	splits = find_conll2003_splits(conll_root)
	if "train" not in splits:
		raise FileNotFoundError(
			f"Could not find CoNLL-2003 train split under {conll_root}. "
			"Place train/dev/test files in data/conll2003/ (or adjust discovery)."
		)

	train_sents = read_conll_sentences(splits["train"])
	dev_sents = read_conll_sentences(splits["dev"]) if "dev" in splits else []

	tag_to_idx, idx_to_tag = build_tag_vocab(train_sents)

	import fasttext

	ft_en = fasttext.load_model(str(english_fasttext_bin))
	cache: Dict[str, np.ndarray] = {}

	train_vecs = [_embed_sentence_fasttext(ft_en, s.tokens, cache) for s in train_sents]
	train_tag_idx = [[tag_to_idx[t] for t in s.tags] for s in train_sents]

	dev_vecs = [_embed_sentence_fasttext(ft_en, s.tokens, cache) for s in dev_sents] if dev_sents else []
	dev_tag_idx = [[tag_to_idx.get(t, tag_to_idx["O"]) for t in s.tags] for s in dev_sents] if dev_sents else []

	model = BiLSTMCRF(
		embedding_dim=embedding_dim,
		hidden_dim=hidden_dim,
		num_tags=len(idx_to_tag),
		dropout=dropout,
	).to(device)

	opt = torch.optim.Adam(model.model.parameters(), lr=lr)

	best_dev_f1 = -1.0
	best_state = None

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		steps = 0
		for batch_x, batch_y in _iter_minibatches(
			train_vecs,
			train_tag_idx,
			batch_size=batch_size,
			shuffle=True,
			seed=seed + epoch,
		):
			x, y, mask = _pad_batch(batch_x, batch_y)
			x = x.to(device)
			y = y.to(device)
			mask = mask.to(device)

			opt.zero_grad(set_to_none=True)
			loss = model.neg_log_likelihood(x, y, mask)
			loss.backward()
			opt.step()

			total_loss += float(loss.item())
			steps += 1

		# Optional quick dev check
		dev_f1 = None
		if dev_vecs:
			metrics = evaluate_sentences(
				model,
				idx_to_tag,
				dev_sents,
				embedder=lambda toks: _embed_sentence_fasttext(ft_en, toks, cache),
				device=device,
				allowed_entity_types=["PER", "ORG", "LOC", "MISC"],
			)
			dev_f1 = metrics["f1"]
			if dev_f1 > best_dev_f1:
				best_dev_f1 = dev_f1
				best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

		print(
			f"[NER] epoch={epoch}/{epochs} loss={total_loss / max(steps, 1):.4f}" +
			(f" dev_f1={dev_f1:.4f}" if dev_f1 is not None else "")
		)

	if best_state is not None:
		model.load_state_dict(best_state)

	ckpt = {
		"state_dict": model.state_dict(),
		"tag_to_idx": tag_to_idx,
		"idx_to_tag": idx_to_tag,
		"embedding_dim": embedding_dim,
		"hidden_dim": hidden_dim,
		"dropout": dropout,
		"seed": seed,
		"conll_splits": {k: str(v) for k, v in splits.items()},
	}
	torch.save(ckpt, out_path)
	return model, tag_to_idx, idx_to_tag


def evaluate_sentences(
	model: BiLSTMCRF,
	idx_to_tag: List[str],
	sentences: Sequence[Sentence],
	*,
	embedder,
	device: str,
	allowed_entity_types: Sequence[str],
	unknown_type_strategy: str = "to_misc",
) -> Dict[str, float]:
	"""Evaluate a model on provided sentences.

	`embedder(tokens)` must return a (len, dim) float32 numpy array.
	"""
	_require_torch()
	import torch
	from seqeval.metrics import f1_score, precision_score, recall_score

	model.eval()
	y_true: List[List[str]] = []
	y_pred: List[List[str]] = []

	for sent in sentences:
		vecs = embedder(sent.tokens)
		x = torch.from_numpy(vecs).unsqueeze(0).to(device)
		mask = torch.ones((1, vecs.shape[0]), dtype=torch.bool, device=device)
		pred_idx = model.decode(x, mask)[0]
		pred_tags = [idx_to_tag[i] for i in pred_idx]

		gold_tags = normalize_tags_to_training_space(
			sent.tags,
			allowed_entity_types=allowed_entity_types,
			unknown_type_strategy=unknown_type_strategy,
		)
		pred_tags = normalize_tags_to_training_space(
			pred_tags,
			allowed_entity_types=allowed_entity_types,
			unknown_type_strategy=unknown_type_strategy,
		)

		y_true.append(gold_tags)
		y_pred.append(pred_tags)

	return {
		"precision": float(precision_score(y_true, y_pred)),
		"recall": float(recall_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred)),
	}


def load_masakhaner_split(lang_dir: Path, split: str) -> List[Sentence]:
	p = _find_split_file(lang_dir, split)
	if p is None:
		raise FileNotFoundError(f"Could not find split '{split}' under {lang_dir}")
	return read_conll_sentences(p)

