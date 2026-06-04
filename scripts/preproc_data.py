"""Preprocess the CommitmentBank item file into a 3-way NLI dataset.

The labeling rule follows the dataset description: keep only items where at
least 80% of the annotation votes fall into one of the three regions:

- entailment: votes in [1, 3]
- neutral: votes equal to 0
- contradiction: votes in [-3, -1]

Rows that do not meet the threshold are discarded.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean


LABEL_RULES = (
	("entailment", lambda vote: 1 <= vote <= 3),
	("neutral", lambda vote: vote == 0),
	("contradiction", lambda vote: -3 <= vote <= -1),
)


def parse_votes(raw_votes: str) -> list[int]:
	votes: list[int] = []
	for value in raw_votes.split(","):
		value = value.strip().strip('"')
		if not value:
			continue
		votes.append(int(value))
	return votes


def label_votes(votes: list[int], min_agreement: float = 0.8) -> tuple[str, float] | None:
	if not votes:
		return None

	total = len(votes)
	for label, predicate in LABEL_RULES:
		support = sum(1 for vote in votes if predicate(vote))
		support_ratio = support / total
		if support_ratio >= min_agreement:
			return label, support_ratio
	return None


def preprocess(input_path: Path, output_path: Path, min_agreement: float = 0.8) -> tuple[int, int, Counter[str]]:
	with input_path.open(newline="", encoding="utf-8") as source_file:
		reader = csv.reader(source_file)
		header = next(reader)
		response_index = header.index("Reponses")

		output_header = header + ["gold_label", "vote_count", "agreement", "mean_vote"]
		kept_rows = 0
		dropped_rows = 0
		label_counts: Counter[str] = Counter()

		with output_path.open("w", newline="", encoding="utf-8") as output_file:
			writer = csv.writer(output_file)
			writer.writerow(output_header)

			for row in reader:
				votes = parse_votes(row[response_index])
				label_info = label_votes(votes, min_agreement=min_agreement)
				if label_info is None:
					dropped_rows += 1
					continue

				label, agreement = label_info
				kept_rows += 1
				label_counts[label] += 1

				writer.writerow(row + [label, len(votes), f"{agreement:.6f}", f"{mean(votes):.6f}"])

	return kept_rows, dropped_rows, label_counts


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Preprocess CommitmentBank items into labeled NLI data.")
	parser.add_argument(
		"--input",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "data" / "CommitmentBank-items.csv",
		help="Path to the raw CommitmentBank items CSV.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "data" / "CommitmentBank-items-labeled.csv",
		help="Path to write the filtered labeled CSV.",
	)
	parser.add_argument(
		"--min-agreement",
		type=float,
		default=0.8,
		help="Minimum fraction of votes required in one label bucket.",
	)
	return parser


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()

	kept_rows, dropped_rows, label_counts = preprocess(
		input_path=args.input,
		output_path=args.output,
		min_agreement=args.min_agreement,
	)

	print(f"Wrote {kept_rows} labeled rows to {args.output}")
	print(f"Dropped {dropped_rows} rows that did not meet the {args.min_agreement:.0%} agreement threshold")
	print(f"Label distribution: {dict(label_counts)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
