from ner_utils import BaseNER
from typing import List, Dict, Union, Tuple
from preprocessing import Preprocessor
import re

class CodeSwitchingNERMetric:
    def __init__(self,
                 ner_modules: List[BaseNER],
                 sentence_ner: BaseNER,
                 origin_alphabet: str ="АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщьЮюЯя"
                 ):
        self.origin_alphabet = origin_alphabet
        self.ner_modules = ner_modules
        self.sentence_ner = sentence_ner

    def get_all_ner_preds_sentences(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, Union[str, int]]]
    ]:
        """get NER predictions, sentences and token separation from the  self.sentence_ner;
        Get all predictions per sentence from the NER modules in self.ner_modules"""

        ner_preds, sentences, tokens_dicts = self.sentence_ner.pred_ner_sents(text)

        for ner_model in self.ner_modules:
            model_preds = ner_model(sentences)

            for i in range(len(model_preds)):
                ner_preds[i] += model_preds[i]


        return ner_preds, sentences, tokens_dicts

    def mergeIntervals(self, intervals):
        """merge overlapping intervals"""
        if not intervals:
            return intervals
        # Sort the array on the basis of start values of intervals.
        intervals.sort()
        stack = []
        # insert first interval into stack
        stack.append(intervals[0])
        for i in intervals[1:]:
            # Check for overlapping interval,
            # if interval overlap
            if stack[-1][0] <= i[0] <= stack[-1][-1]:
                stack[-1][-1] = max(stack[-1][-1], i[-1])
            else:
                stack.append(i)
        return stack


    def merge_preds(self,
                    sentence: str,
                    preds: List[Dict[str, Union[int, str]]]) -> List[Dict[str, Union[int, str]]]:
        """Merge overlapping ner predictions"""

        if not preds:
            return []


        intervals = [
            [pred_dict["start"], pred_dict["end"]] for pred_dict in preds
        ]
        merged_intervals = self.mergeIntervals(intervals=intervals)

        merged_preds = [
            {
                "start": start,
                "end": end,
                "text": sentence[start: end],
            } for start, end in merged_intervals
        ]
        return merged_preds

    def find_non_vocab_words_starts(self, text: str) -> List[Tuple[int, int]]:
        """find all the intervals of words that have an unexpected symbol in them (letters belonging to the foreign alphabet)"""

        # Create a set of allowed characters from the vocabulary
        allowed_chars = set(self.origin_alphabet)

        # Create a regex pattern to match any character that is not in the allowed_chars
        pattern = r'[^' + re.escape(''.join(allowed_chars)) + r'\s\W\d_]+'

        # Find all matches in the text
        matches = re.finditer(pattern, text)

        return [(m.start(), m.end()) for m in matches]


    def is_proper_name(self, substring_start: int, substing_end: int, ner_preds: List[Dict[str, Union[int, str]]]) -> bool:
        """Check if the provided interval intersect with any of the extracted entities"""
        for pred in ner_preds:
            if not (substing_end < pred["start"] or substring_start > pred["end"]):
                return True

        return False


    def calc_sentences_num_broken(self, sentences, merged_ner_preds) -> int:
        """calculate number of broken sentences"""
        num_broken_sentences = 0

        for sentence, sentence_ner_preds in zip(sentences, merged_ner_preds):

            # all unexpected alphabet symbols
            sentence_foreign_substrings_ranges = self.find_non_vocab_words_starts(text=sentence)

            non_proper_names_substrings_ranges = [
                (foreign_substring_start, foreign_substring_end)
                for foreign_substring_start, foreign_substring_end in sentence_foreign_substrings_ranges
                if not self.is_proper_name(
                    substring_start=foreign_substring_start,
                    substing_end=foreign_substring_start,
                    ner_preds=sentence_ner_preds
                )
            ]

            if len(non_proper_names_substrings_ranges):
                num_broken_sentences += 1

        return num_broken_sentences



    def calc_token_level_num_broken(self, tokens: List[Dict[str, Union[str, int]]], merged_ner_preds) -> Tuple[int, List[str]]:
        num_broken = 0
        broken_tokens = []

        all_ner_preds = sum(merged_ner_preds, [])

        for token_dict in tokens:
            if self.find_non_vocab_words_starts(token_dict["text"]):
                if not self.is_proper_name(
                                    substring_start=token_dict["start"],
                                    substing_end=token_dict["end"],
                                    ner_preds=all_ner_preds
                            ):
                    num_broken += 1
                    broken_tokens.append(token_dict["text"])

        return num_broken, broken_tokens


    def calculate(self, texts: List[str]) -> Dict[str, Union[str, float]]:
        """metric calculation for a list of texts"""

        total_num_sentences, total_num_tokens, total_num_tokens = 0, 0, 0
        num_broken_sentences, num_broken_texts, num_broken_tokens = 0, 0, 0
        total_num_texts = len(texts)

        for raw_text in texts:
            text = Preprocessor.preprocess(text=raw_text)

            if text:
                sentence_ner_preds, sentences, tokens_dicts = self.get_all_ner_preds_sentences(text=text)
                merged_ner_preds = [
                    self.merge_preds(
                        sentence=sentences[i],
                        preds=sentence_ner_preds[i]
                    ) for i in range(len(sentence_ner_preds))
                ]


                # calculate number of broken sentences
                text_num_broken_sentences = self.calc_sentences_num_broken(
                    sentences=sentences,
                    merged_ner_preds=merged_ner_preds
                )
                if text_num_broken_sentences:
                    num_broken_texts += 1
                num_broken_sentences += text_num_broken_sentences
                total_num_sentences += len(sentences)

                # calculate number of broken tokens
                text_num_broken_words, broken_tokens = self.calc_token_level_num_broken(
                    tokens=tokens_dicts,
                    merged_ner_preds=merged_ner_preds
                )

                # print(broken_tokens)

                num_broken_tokens += text_num_broken_words
                total_num_tokens += len(tokens_dicts)

        output = {
            "broken_sentences_ratio": num_broken_sentences/total_num_sentences if total_num_sentences else -1.0,
            "broken_texts_ratio": num_broken_texts/total_num_texts if total_num_texts else -1.0,
            "total_num_texts": total_num_texts,
            "total_num_sentences": total_num_sentences,
            "broken_tokens_ratio": num_broken_tokens/total_num_tokens if total_num_tokens else -1.0,
            "total_num_tokens": total_num_tokens
        }
        return output

if __name__ == '__main__':
    from ner_utils import *

    ner_modules = [
        TransformersNER(),
        SpacyNER(),
        RegexURLFinder(),
        RegexQuotesFinder()
    ]
    sentence_ner = StanzaNER()

    test_texts = [
        "Все нормально. Мабуть.",
        "Кручу верчу метрику рахую",
        "єєєєZAZ-1103 Slavuta це є the best автомобіль in the світі"
    ]

    metric = CodeSwitchingNERMetric(ner_modules=ner_modules, sentence_ner=sentence_ner)

    print(metric.calculate(texts=test_texts))
