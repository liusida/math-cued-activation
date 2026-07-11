from __future__ import annotations

import unittest

from ica_lens_v9.features.probe import _replacement_token_display_parts


class _FragmentTokenizer:
    def decode(self, token_ids: list[int], **_: object) -> str:
        values = {1: "�", 2: "�", 3: "3"}
        if token_ids == [1, 2]:
            return "√"
        return "".join(values[token_id] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {1: "raw-a", 2: "raw-b", 3: "3"}[token_id]


class ReplacementTokenDisplayTest(unittest.TestCase):
    def test_adjacent_fragments_are_labeled_without_merging_positions(self) -> None:
        tokenizer = _FragmentTokenizer()
        parts = _replacement_token_display_parts(tokenizer, [1, 2, 3], ["�", "�", "3"])

        self.assertEqual(parts[0]["display_text"], "√")
        self.assertEqual(parts[0]["display_piece_index"], 1)
        self.assertEqual(parts[0]["display_piece_count"], 2)
        self.assertEqual(parts[1]["display_piece_index"], 2)
        self.assertIsNone(parts[2])

    def test_unmatched_replacement_character_is_left_unchanged(self) -> None:
        tokenizer = _FragmentTokenizer()
        parts = _replacement_token_display_parts(tokenizer, [1, 3], ["�", "3"])

        self.assertEqual(parts, [None, None])

    def test_stable_prefix_is_not_repeated_in_recovered_character(self) -> None:
        tokenizer = _FragmentTokenizer()
        tokenizer.decode = lambda token_ids, **_: "((√" if token_ids == [1, 2] else {1: "((�", 2: "�"}[token_ids[0]]
        parts = _replacement_token_display_parts(tokenizer, [1, 2], ["((�", "�"])

        self.assertEqual(parts[0]["display_text"], "√")


if __name__ == "__main__":
    unittest.main()
