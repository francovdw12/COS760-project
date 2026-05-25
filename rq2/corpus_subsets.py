"""Corpus subsetting utilities for RQ2 (data efficiency).

This module builds deterministic corpus subsets at configured fractions.
We subset *target-language* NCHLT corpora (zul/nso/tsn) only.

Output layout (local-only, ignored by git):
  data/subsets/{lang}/f100.txt, f075.txt, f050.txt, f025.txt, f010.txt, f005.txt
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from config import CORPUS_FRACTIONS, get_corpus_path, get_subset_corpus_path


@dataclass(frozen=True)
class SubsetStats:
	lang: str
	fraction: float
	out_path: Path
	total_lines: int
	total_tokens: int
	selected_lines: int
	selected_tokens: int
	seed: int


def _count_tokens(line: str) -> int:
	return len(line.split())


def _load_lines_and_token_counts(corpus_path: Path) -> Tuple[List[str], List[int]]:
	lines: List[str] = []
	token_counts: List[int] = []

	with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
		for raw in f:
			line = raw.strip()
			if not line:
				continue
			tokens = _count_tokens(line)
			if tokens == 0:
				continue
			lines.append(line)
			token_counts.append(tokens)

	return lines, token_counts


def build_subsets_from_corpus(
	*,
	corpus_path: Path,
	out_dir: Path,
	fractions: Sequence[float],
	seed: int = 42,
	preserve_original_order: bool = True,
	force: bool = False,
) -> List[SubsetStats]:
	"""Build deterministic subsets for one corpus.

	Strategy:
	- Read all non-empty lines.
	- Shuffle line indices deterministically once.
	- For each fraction, take a prefix of the shuffled indices until a token budget
	  (fraction * total_tokens) is met/exceeded.

	This gives nested subsets and reproducibility.
	"""
	out_dir.mkdir(parents=True, exist_ok=True)

	if not corpus_path.exists():
		raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

	lines, token_counts = _load_lines_and_token_counts(corpus_path)
	total_lines = len(lines)
	total_tokens = int(sum(token_counts))

	if total_lines == 0 or total_tokens == 0:
		raise ValueError(f"Empty corpus after filtering: {corpus_path}")

	rng = random.Random(seed)
	indices = list(range(total_lines))
	rng.shuffle(indices)

	# Precompute cumulative token counts over the shuffled order.
	cumulative_tokens: List[int] = []
	running = 0
	for idx in indices:
		running += token_counts[idx]
		cumulative_tokens.append(running)

	results: List[SubsetStats] = []
	for fraction in fractions:
		if not (0 < float(fraction) <= 1.0):
			raise ValueError(f"fraction must be in (0, 1], got {fraction}")

		out_path = out_dir / get_subset_corpus_path(out_dir.name, fraction).name

		if out_path.exists() and not force:
			selected_lines = 0
			selected_tokens = 0
			with out_path.open("r", encoding="utf-8", errors="ignore") as existing:
				for raw in existing:
					line = raw.strip()
					if not line:
						continue
					selected_lines += 1
					selected_tokens += _count_tokens(line)
			results.append(
				SubsetStats(
					lang=out_dir.name,
					fraction=float(fraction),
					out_path=out_path,
					total_lines=total_lines,
					total_tokens=total_tokens,
					selected_lines=selected_lines,
					selected_tokens=selected_tokens,
					seed=seed,
				)
			)
			continue

		token_budget = max(1, int(round(total_tokens * float(fraction))))

		# Find smallest prefix length that meets/exceeds the budget.
		prefix_len = 0
		for i, cum in enumerate(cumulative_tokens):
			if cum >= token_budget:
				prefix_len = i + 1
				break
		if prefix_len == 0:
			prefix_len = total_lines

		selected_indices = indices[:prefix_len]
		if preserve_original_order:
			selected_indices = sorted(selected_indices)

		selected_tokens = int(sum(token_counts[i] for i in selected_indices))
		selected_lines = len(selected_indices)

		with out_path.open("w", encoding="utf-8") as out:
			for i in selected_indices:
				out.write(lines[i] + "\n")

		results.append(
			SubsetStats(
				lang=out_dir.name,
				fraction=float(fraction),
				out_path=out_path,
				total_lines=total_lines,
				total_tokens=total_tokens,
				selected_lines=selected_lines,
				selected_tokens=selected_tokens,
				seed=seed,
			)
		)

	# Metadata for debugging/repro
	meta_path = out_dir / "metadata.csv"
	with meta_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"lang",
				"fraction",
				"subset_path",
				"total_lines",
				"total_tokens",
				"selected_lines",
				"selected_tokens",
				"seed",
			],
		)
		writer.writeheader()
		for row in results:
			writer.writerow(
				{
					"lang": row.lang,
					"fraction": row.fraction,
					"subset_path": str(row.out_path),
					"total_lines": row.total_lines,
					"total_tokens": row.total_tokens,
					"selected_lines": row.selected_lines,
					"selected_tokens": row.selected_tokens,
					"seed": row.seed,
				}
			)

	return results


def ensure_language_subsets(
	lang: str,
	*,
	fractions: Sequence[float] = CORPUS_FRACTIONS,
	seed: int = 42,
	force: bool = False,
) -> List[SubsetStats]:
	"""Ensure subset corpora exist for one target language."""
	corpus_path = Path(get_corpus_path(lang))
	out_dir = get_subset_corpus_path(lang, 1.0).parent
	return build_subsets_from_corpus(
		corpus_path=corpus_path,
		out_dir=out_dir,
		fractions=fractions,
		seed=seed,
		preserve_original_order=True,
		force=force,
	)


def main(argv: Sequence[str] | None = None) -> int:
	import argparse

	parser = argparse.ArgumentParser(description="Build deterministic NCHLT subset corpora for RQ2")
	parser.add_argument("--lang", required=True, help="Language code: zul, nso, tsn")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--force", action="store_true", help="Overwrite existing subset files")
	args = parser.parse_args(argv)

	stats = ensure_language_subsets(args.lang, seed=args.seed, force=args.force)
	print(f"Wrote {len(stats)} subsets under {stats[0].out_path.parent}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

