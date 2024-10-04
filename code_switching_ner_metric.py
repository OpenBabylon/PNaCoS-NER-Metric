from ner_utils import BaseNER
from typing import List, Dict, Union, Tuple, Any
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


    def check_if_lang_match(self, text:str) -> List[bool]:
        include_at_least_one_alphabet_symbol = bool(re.findall(
                "[" + self.origin_alphabet + "]+", text
            ))

        return include_at_least_one_alphabet_symbol

    def is_sent_in_required_lang_ner(self, sent_text, sent_range_dict, sent_ner_preds):
        if self.check_if_lang_match(sent_text):
            return True

        left_sent_text = sent_text
        for ner_pred in sent_ner_preds:
            left_sent_text = left_sent_text[:ner_pred["start"]-sent_range_dict["start"]] + " "*(ner_pred["end"] - ner_pred["start"]) + left_sent_text[ner_pred["end"]-sent_range_dict["start"]:]

        return not bool(left_sent_text.replace(" ", ""))


    def get_all_ner_preds_sentences(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        """get NER predictions, sentences and token separation from the  self.sentence_ner;
        Get all predictions per sentence from the NER modules in self.ner_modules"""

        ner_preds, sentences, sentences_ranges, tokens_dicts = self.sentence_ner.pred_ner_sents(text)



        for ner_model in self.ner_modules:
            model_preds = ner_model(
                sentences=sentences,
                sentences_ranges=sentences_ranges,
                tokens_dicts=tokens_dicts
            )

            for i in range(len(model_preds)):
                ner_preds[i] += model_preds[i]


        return ner_preds, sentences, sentences_ranges, tokens_dicts

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
                    text: str,
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
                "text": text[start: end],
            } for start, end in merged_intervals
        ]

        return merged_preds

    def find_non_vocab_words_starts(self, text: str) -> List[Tuple[int, int]]:
        """find all the intervals of words that have an unexpected symbol in them (letters belonging to the foreign alphabet)"""

        # Create a set of allowed characters from the vocabulary
        allowed_chars = set(self.origin_alphabet)

        # Create a regex pattern to match any character that is not in the allowed_chars
        pattern = r'[^' + re.escape(''.join(allowed_chars)) + r'\s\W\d]+'

        # Find all matches in the text
        matches = re.finditer(pattern, text)

        return [(m.start(), m.end()) for m in matches]

    def is_number(self, text):
        try:
            # Try converting the text to a float
            float(text)
            return True
        except ValueError:
            return False

    def is_proper_name(self, substring_start: int, substing_end: int,
                       ner_preds: List[Dict[str, Union[int, str]]],
                       offset=0, text="") -> bool:
        """Check if the provided interval intersect with any of the extracted entities"""
        for pred in ner_preds:
            if self.is_intersection(
                    start_1=substring_start, end_1=substing_end,
                    start_2=pred["start"] + offset, end_2=pred["end"] + offset
            ):
            # if not (substing_end < pred["start"] + offset or substring_start > pred["end"] + offset):
                return True

        return False

    def is_intersection(self, start_1,end_1, start_2, end_2 ) -> bool:
        return not (end_1 < start_2 or start_1 > end_2)


    def calc_sentences_num_broken(self, broken_tokens_dicts, sentences_ranges) -> int:
        """calculate number of broken sentences"""
        num_broken_sentences = 0
        for sentence_range_idx_dict in sentences_ranges:
            for broken_token_dict in broken_tokens_dicts:
                if self.is_intersection(
                        start_1=sentence_range_idx_dict["start"],
                        end_1=sentence_range_idx_dict["end"],
                        start_2=broken_token_dict["start"],
                        end_2=broken_token_dict["end"],
                ):
                    num_broken_sentences += 1
                    break

        return num_broken_sentences

    def check_token_sentence_lang(self, substring_start, substing_end, sents_correct_langs, sentences_ranges):

        for i, sent_range_dict in enumerate(sentences_ranges):
            if substring_start >= sent_range_dict["start"] and substing_end <= sent_range_dict["end"]:
                return sents_correct_langs[i]




    def calc_token_level_num_broken(self, tokens: List[Dict[str, Union[str, int]]],
                                    merged_ner_preds, sents_correct_langs,
                                    sentences_ranges) -> Tuple[int, Dict[str, Any]]:
        num_broken = 0
        broken_tokens = []

        all_ner_preds = sum(merged_ner_preds, [])

        for token_dict in tokens:

            if not self.check_token_sentence_lang(
                    substring_start=token_dict["start"],
                    substing_end=token_dict["end"],
                    sents_correct_langs=sents_correct_langs,
                    sentences_ranges=sentences_ranges):
                num_broken += 1
                broken_tokens.append(token_dict)

            else:
                if self.find_non_vocab_words_starts(token_dict["text"]):
                    if not self.is_proper_name(
                                        substring_start=token_dict["start"],
                                        substing_end=token_dict["end"],
                                        ner_preds=all_ner_preds
                                ):
                        num_broken += 1
                        broken_tokens.append(token_dict)

        return num_broken, broken_tokens


    def calculate(self, texts: List[str]) -> Dict[str, Union[str, float]]:
        """metric calculation for a list of texts"""

        total_num_sentences, total_num_tokens, total_num_tokens = 0, 0, 0
        num_broken_sentences, num_broken_texts, num_broken_tokens = 0, 0, 0
        total_num_texts = len(texts)

        for raw_text in texts:
            text = Preprocessor.preprocess(text=raw_text)

            if text:
                sentence_ner_preds, sentences, sentences_ranges, tokens_dicts = self.get_all_ner_preds_sentences(text=text)
                merged_ner_preds = [
                    self.merge_preds(
                        text=text,
                        preds=sentence_ner_preds[i]
                    ) for i in range(len(sentence_ner_preds))
                ]

                sents_correct_langs = [
                    self.is_sent_in_required_lang_ner(
                        sent_text=sentences[i],
                        sent_range_dict=sentences_ranges[i],
                        sent_ner_preds=preds
                    ) for i, preds in enumerate(merged_ner_preds)
                ]

                merged_ner_preds = [
                    preds if is_sent_correct_lang else []
                    for preds, is_sent_correct_lang in zip(merged_ner_preds, sents_correct_langs)
                ]



                total_num_sentences += len(sentences)

                # calculate number of broken tokens
                text_num_broken_words, broken_tokens_dicts = self.calc_token_level_num_broken(
                    tokens=tokens_dicts,
                    merged_ner_preds=merged_ner_preds,
                    sents_correct_langs=sents_correct_langs,
                    sentences_ranges=sentences_ranges
                )

                if text_num_broken_words:
                    num_broken_texts += 1

                text_num_broken_sentences = self.calc_sentences_num_broken(
                    broken_tokens_dicts=broken_tokens_dicts,
                    sentences_ranges=sentences_ranges
                )

                num_broken_tokens += text_num_broken_words
                total_num_tokens += len(tokens_dicts)
                num_broken_sentences += text_num_broken_sentences



        output = {
            "codeswitch_sentences_ratio": num_broken_sentences/total_num_sentences if total_num_sentences else -1.0,
            "codeswitch_texts_ratio": num_broken_texts/total_num_texts if total_num_texts else -1.0,
            "total_num_texts": total_num_texts,
            "total_num_sentences": total_num_sentences,
            "codeswitch_words_ratio": num_broken_tokens/total_num_tokens if total_num_tokens else -1.0,
            "total_num_tokens": total_num_tokens
        }
        return output

if __name__ == '__main__':
    from loaders import load_metric

    test_texts = [
        """Applications are now open! Deadline is September 30, 2019 at 5:00 p.m. ET. Funding will be awarded to schools who demonstrate a commitment to increasing participation and skill acquisition through high quality Physical Education programs that incorporate gymnastics based activities. Applicants must have an active membership with SHAPE America (formerly AAHPERD)."""
    ]


    metric = load_metric()

    print(metric.calculate(texts=test_texts))
