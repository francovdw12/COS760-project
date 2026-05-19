from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from alignment.CCA import CCAAligner
from alignment.KCCA import KCCAAligner
from alignment.VecMap import load_txt_embeddings, run_vecmap
from config import (
	CONLL2003_ROOT,
	CORPUS_FRACTIONS,
	EMBEDDING_DIM,
	ENGLISH,
	LANGUAGES,
	OUTPUTS_ROOT,
	RESULTS_ROOT,
	get_alignment_artifact_path,
	get_conll2003_root,
	get_corpus_path,
	get_display_name,
	get_embeddings_path,
	get_fraction_embeddings_path,
	get_fraction_text_embeddings_path,
	get_ner_path,
	get_seed_lexicon_path,
	get_subset_corpus_path,
	get_text_embeddings_path,
)
from embeddings import load_embeddings_as_matrix, save_embeddings_as_txt, train_fasttext
from lexicon import build_anchor_matrices, load_lexicon
from transfer.corpus_subsets import ensure_language_subsets
from transfer.zero_shot_eval import (
	Sentence,
	evaluate_sentences,
	load_masakhaner_split,
	train_or_load_english_ner_model,
)


METHODS = ["CCA", "KCCA", "VecMap"]


def _ensure_fasttext_model(corpus_path: Path, bin_path: Path, txt_path: Path | None) -> None:
	if not corpus_path.exists():
		raise FileNotFoundError(f"Missing corpus: {corpus_path}")

	bin_path.parent.mkdir(parents=True, exist_ok=True)
	if not bin_path.exists():
		print(f"[FastText] training {bin_path.name} from {corpus_path}")
		train_fasttext(str(corpus_path), str(bin_path), dim=EMBEDDING_DIM)

	if txt_path is not None:
		txt_path.parent.mkdir(parents=True, exist_ok=True)
		if not txt_path.exists():
			save_embeddings_as_txt(str(bin_path), str(txt_path))


def _entity_types_from_tagset(idx_to_tag: List[str]) -> List[str]:
	types = set()
	for tag in idx_to_tag:
		if tag == "O":
			continue
		if "-" in tag:
			types.add(tag.split("-", 1)[1])
	return sorted(types)


def _fit_or_load_cca_mapping(
	*,
	lang: str,
	fraction: float,
	src_words: List[str],
	src_matrix: np.ndarray,
	en_words: List[str],
	en_matrix: np.ndarray,
	lexicon_pairs,
	force: bool,
) -> Tuple[CCAAligner, np.ndarray]:
	artifact = get_alignment_artifact_path(lang, fraction, "CCA")
	if artifact.exists() and not force:
		data = np.load(artifact, allow_pickle=False)
		aligner = CCAAligner(n_components=int(data["n_components"]))
		aligner.W_src = data["W_src"]
		aligner.W_tgt = data["W_tgt"]
		R = data["R"]
		return aligner, R

	X_src, X_tgt = build_anchor_matrices(lexicon_pairs, src_words, src_matrix, en_words, en_matrix)
	if len(X_src) < 2:
		raise ValueError(f"Not enough anchor pairs for {lang} at fraction={fraction}")

	n_components = min(100, X_src.shape[0], X_src.shape[1], X_tgt.shape[1])
	aligner = CCAAligner(n_components=n_components)
	aligner.fit(X_src, X_tgt)

	# Learn a linear map from canonical target space back to English embedding space.
	Z_tgt = aligner.transform_tgt(X_tgt)
	R = np.linalg.lstsq(Z_tgt, X_tgt, rcond=None)[0].astype(np.float32)  # (k, dim)

	artifact.parent.mkdir(parents=True, exist_ok=True)
	np.savez(
		artifact,
		n_components=n_components,
		W_src=aligner.W_src.astype(np.float32),
		W_tgt=aligner.W_tgt.astype(np.float32),
		R=R,
	)
	return aligner, R


def _fit_or_load_kcca_mapping(
	*,
	lang: str,
	fraction: float,
	src_words: List[str],
	src_matrix: np.ndarray,
	en_words: List[str],
	en_matrix: np.ndarray,
	lexicon_pairs,
	gamma: float = 0.1,
	reg: float = 1e-3,
	max_anchors: int = 1000,
	force: bool,
) -> Tuple[KCCAAligner, np.ndarray]:
	artifact = get_alignment_artifact_path(lang, fraction, "KCCA")
	if artifact.exists() and not force:
		data = np.load(artifact, allow_pickle=False)
		aligner = KCCAAligner(
			n_components=int(data["n_components"]),
			gamma=float(data["gamma"]),
			reg=float(data["reg"]),
		)
		aligner.alpha = data["alpha"]
		aligner.beta = data["beta"]
		aligner.X_src_train = data["X_src_train"]
		aligner.X_tgt_train = data["X_tgt_train"]
		R = data["R"]
		return aligner, R

	X_src, X_tgt = build_anchor_matrices(lexicon_pairs, src_words, src_matrix, en_words, en_matrix)
	if len(X_src) < 5:
		raise ValueError(f"Not enough anchor pairs for {lang} at fraction={fraction}")

	n = min(max_anchors, len(X_src))
	n_components = min(50, n)
	aligner = KCCAAligner(n_components=n_components, gamma=gamma, reg=reg)
	aligner.fit(X_src[:n], X_tgt[:n])

	Z_tgt = aligner.transform_tgt(X_tgt[:n])
	R = np.linalg.lstsq(Z_tgt, X_tgt[:n], rcond=None)[0].astype(np.float32)

	artifact.parent.mkdir(parents=True, exist_ok=True)
	np.savez(
		artifact,
		n_components=n_components,
		gamma=gamma,
		reg=reg,
		alpha=aligner.alpha.astype(np.float32),
		beta=aligner.beta.astype(np.float32),
		X_src_train=aligner.X_src_train.astype(np.float32),
		X_tgt_train=aligner.X_tgt_train.astype(np.float32),
		R=R,
	)
	return aligner, R


def run_rq2(
	*,
	langs: Sequence[str],
	fractions: Sequence[float],
	methods: Sequence[str],
	masakha_split: str,
	force: bool,
	device: str,
) -> None:
	results: List[Dict] = []
	Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
	Path(OUTPUTS_ROOT).mkdir(parents=True, exist_ok=True)

	# --- English embeddings (for NER training + VecMap target space) ---
	en_corpus = Path(get_corpus_path(ENGLISH))
	en_bin = get_embeddings_path(ENGLISH)
	en_txt = get_text_embeddings_path(ENGLISH)
	try:
		_ensure_fasttext_model(en_corpus, en_bin, en_txt)
	except FileNotFoundError as e:
		print(f"[RQ2] {e}")
		print("[RQ2] Stage local datasets under data/ (see README) and retry.")
		return

	# Train/load English NER model once.
	ner_ckpt = Path(OUTPUTS_ROOT) / "ner" / "bilstm_crf_conll2003.pt"
	try:
		model, _tag_to_idx, idx_to_tag = train_or_load_english_ner_model(
			conll_root=get_conll2003_root(),
			english_fasttext_bin=en_bin,
			out_path=ner_ckpt,
			embedding_dim=EMBEDDING_DIM,
			epochs=5,
			batch_size=16,
			seed=42,
			force_retrain=force,
			device=device,
		)
	except FileNotFoundError as e:
		print(f"[RQ2] {e}")
		print("[RQ2] Add CoNLL-2003 files under data/conll2003/ (see README) and retry.")
		return
	allowed_types = _entity_types_from_tagset(idx_to_tag)
	print(f"[NER] training tag types: {allowed_types}")

	# Load English embedding matrix once for anchor building.
	en_words, en_matrix = load_embeddings_as_matrix(str(en_bin))

	for lang in langs:
		display = get_display_name(lang)
		print(f"\n{'=' * 60}\nRQ2 language: {display} ({lang})\n{'=' * 60}")

		lex_path = get_seed_lexicon_path(lang)
		if lex_path is None or not lex_path.exists():
			print(f"[RQ2] Missing seed lexicon for {lang}: {lex_path} (skipping)")
			continue
		lexicon_pairs = load_lexicon(str(lex_path))
		if not lexicon_pairs:
			print(f"[RQ2] Empty seed lexicon for {lang}: {lex_path} (skipping)")
			continue

		ner_dir = get_ner_path(lang)
		if ner_dir is None or not Path(ner_dir).exists():
			print(f"[RQ2] Missing MasakhaNER folder for {lang}: {ner_dir} (skipping)")
			continue

		try:
			eval_sents: List[Sentence] = load_masakhaner_split(Path(ner_dir), masakha_split)
		except Exception as e:
			print(f"[RQ2] Failed to load MasakhaNER split '{masakha_split}' for {lang}: {e}")
			continue

		# Ensure subset corpora exist and collect token counts.
		try:
			subset_stats = ensure_language_subsets(lang, fractions=fractions, seed=42, force=force)
		except FileNotFoundError as e:
			print(f"[RQ2] {e}")
			print(f"[RQ2] Missing NCHLT corpus for {lang}; skipping.")
			continue
		token_count_by_fraction = {s.fraction: s.selected_tokens for s in subset_stats}

		import fasttext

		for fraction in fractions:
			subset_corpus = get_subset_corpus_path(lang, fraction)
			if not subset_corpus.exists():
				print(f"[RQ2] Missing subset corpus {subset_corpus} (skipping fraction={fraction})")
				continue

			src_bin = get_fraction_embeddings_path(lang, fraction)
			src_txt = get_fraction_text_embeddings_path(lang, fraction)
			_ensure_fasttext_model(subset_corpus, src_bin, src_txt)

			src_words, src_matrix = load_embeddings_as_matrix(str(src_bin))

			ft_src = fasttext.load_model(str(src_bin))

			for method in methods:
				print(f"[RQ2] fraction={fraction:.2f} method={method}")

				if method == "CCA":
					try:
						aligner, R = _fit_or_load_cca_mapping(
							lang=lang,
							fraction=fraction,
							src_words=src_words,
							src_matrix=src_matrix,
							en_words=en_words,
							en_matrix=en_matrix,
							lexicon_pairs=lexicon_pairs,
							force=force,
						)
					except Exception as e:
						print(f"  [CCA] failed: {e}")
						continue

					cache: Dict[str, np.ndarray] = {}

					def embed_aligned(tokens: List[str]) -> np.ndarray:
						vecs = []
						for tok in tokens:
							key = tok
							if key in cache:
								v = cache[key]
							else:
								v = ft_src.get_word_vector(tok)
								v = v / (np.linalg.norm(v) + 1e-8)
								cache[key] = v.astype(np.float32)
							vecs.append(v)
						X = np.stack(vecs, axis=0)
						Z = aligner.transform_src(X)
						Y = (Z @ R).astype(np.float32)
						norms = np.linalg.norm(Y, axis=1, keepdims=True)
						return (Y / np.maximum(norms, 1e-8)).astype(np.float32)

					coverage = 1.0

				elif method == "KCCA":
					try:
						aligner, R = _fit_or_load_kcca_mapping(
							lang=lang,
							fraction=fraction,
							src_words=src_words,
							src_matrix=src_matrix,
							en_words=en_words,
							en_matrix=en_matrix,
							lexicon_pairs=lexicon_pairs,
							force=force,
						)
					except Exception as e:
						print(f"  [KCCA] failed: {e}")
						continue

					cache: Dict[str, np.ndarray] = {}

					def embed_aligned(tokens: List[str]) -> np.ndarray:
						vecs = []
						for tok in tokens:
							key = tok
							if key in cache:
								v = cache[key]
							else:
								v = ft_src.get_word_vector(tok)
								v = v / (np.linalg.norm(v) + 1e-8)
								cache[key] = v.astype(np.float32)
							vecs.append(v)
						X = np.stack(vecs, axis=0)
						Z = aligner.transform_src(X)
						Y = (Z @ R).astype(np.float32)
						norms = np.linalg.norm(Y, axis=1, keepdims=True)
						return (Y / np.maximum(norms, 1e-8)).astype(np.float32)

					coverage = 1.0

				elif method == "VecMap":
					vecmap_dir = Path(OUTPUTS_ROOT) / f"vecmap_{lang}_{fraction:.2f}"
					src_aligned_path = vecmap_dir / "src_aligned.txt"
					if src_aligned_path.exists() and not force:
						aligned_dict = load_txt_embeddings(str(src_aligned_path))
					else:
						try:
							aligned_dict, _ = run_vecmap(
								str(src_txt),
								str(en_txt),
								str(lex_path),
								output_dir=str(vecmap_dir),
							)
						except Exception as e:
							print(f"  [VecMap] failed: {e}")
							continue

					stats = {"hit": 0, "total": 0}
					cache: Dict[str, np.ndarray] = {}

					def embed_aligned(tokens: List[str]) -> np.ndarray:
						vecs = []
						for tok in tokens:
							stats["total"] += 1
							if tok in cache:
								vecs.append(cache[tok])
								continue

							v = aligned_dict.get(tok)
							if v is None and tok.lower() != tok:
								v = aligned_dict.get(tok.lower())

							if v is not None:
								stats["hit"] += 1
								v = v.astype(np.float32)
								v = v / (np.linalg.norm(v) + 1e-8)
							else:
								# Fallback: unaligned FastText vector (keeps dims stable)
								v = ft_src.get_word_vector(tok)
								v = v / (np.linalg.norm(v) + 1e-8)
								v = v.astype(np.float32)

							cache[tok] = v
							vecs.append(v)
						return np.stack(vecs, axis=0).astype(np.float32)

					coverage = None  # computed after evaluation

				else:
					print(f"[RQ2] Unknown method: {method}")
					continue

				metrics = evaluate_sentences(
					model,
					idx_to_tag,
					eval_sents,
					embedder=embed_aligned,
					device=device,
					allowed_entity_types=allowed_types,
					unknown_type_strategy="to_misc",
				)

				if method == "VecMap":
					coverage = (stats["hit"] / stats["total"]) if stats["total"] else 0.0

				results.append(
					{
						"language": lang,
						"fraction": float(fraction),
						"subset_tokens": int(token_count_by_fraction.get(float(fraction), -1)),
						"method": method,
						"precision": round(metrics["precision"], 4),
						"recall": round(metrics["recall"], 4),
						"f1": round(metrics["f1"], 4),
						"vecmap_coverage": round(float(coverage), 4) if coverage is not None else "",
					}
				)

	if not results:
		print("[RQ2] No results produced (missing data?)")
		return

	df = pd.DataFrame(results)
	out_csv = Path(RESULTS_ROOT) / "rq2_results.csv"
	df.to_csv(out_csv, index=False)
	print(f"\n[RQ2] wrote {out_csv} ({len(df)} rows)")


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="RQ2: corpus fractions + zero-shot NER")
	parser.add_argument("--langs", nargs="*", default=LANGUAGES)
	parser.add_argument("--fractions", nargs="*", type=float, default=CORPUS_FRACTIONS)
	parser.add_argument("--methods", nargs="*", default=METHODS)
	parser.add_argument("--split", default="test", help="MasakhaNER split to evaluate")
	parser.add_argument("--force", action="store_true", help="Overwrite/retrain cached artifacts")
	parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
	args = parser.parse_args(argv)

	run_rq2(
		langs=args.langs,
		fractions=args.fractions,
		methods=args.methods,
		masakha_split=args.split,
		force=args.force,
		device=args.device,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

