from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
import stanza
import re

from typing import List, Union, Dict, Tuple

class BaseNER:

    def __call__(self, sentences: List[str], sentences_ranges: List[Dict[str, int]]) -> List[List[Dict[str, Union[int, str]]]]:
        return []

    def pred_ner_sents(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        return [], [], [], []

class TransformersNER(BaseNER):
    def __init__(self,
                 consider_labels: List[str]=[
                     "MISC", "PER", "ORG", "LOC"
                 ],
                 modelname: str = "EvanD/xlm-roberta-base-ukrainian-ner-ukrner"):

        tokenizer = AutoTokenizer.from_pretrained(modelname)
        ner_model = AutoModelForTokenClassification.from_pretrained(modelname)
        self.consider_labels = consider_labels

        self.ppl = pipeline("ner",
                            model=ner_model,
                            tokenizer=tokenizer,
                            aggregation_strategy="simple"
                            )

    def __call__(self, sentences: List[str], sentences_ranges: List[Dict[str, int]]) -> List[List[Dict[str, Union[int, str]]]]:
        preds = self.ppl(sentences)

        output = []
        for sentence_idx_range, pred_dicts in zip(sentences_ranges, preds):
            output.append(
                [
                    {
                        "text": pred['word'],
                        "start_in_sentence": pred["start"],
                        "end_in_sentence": pred["end"],
                        "start": pred["start"] + sentence_idx_range["start"],
                        "end": pred["end"] + sentence_idx_range["start"],
                        "label": pred["entity_group"]
                    } for pred in pred_dicts if pred["entity_group"] in self.consider_labels
                ]
            )
        return output




class SpacyNER(BaseNER):
    def __init__(self,
                 consider_labels: List[str] = [
                     "ORG", "PER", "MISC", "LOC", "PERSON",
                     "LOCATION", "GPE"
                 ],
                 modelname: str ="uk_core_news_lg"):
        self.nlp = spacy.load(modelname)
        self.consider_labels = consider_labels

    def __call__(self, sentences: List[str], sentences_ranges: List[Dict[str, int]]) -> List[List[Dict[str, Union[int, str]]]]:
        preds = []
        for sentence, sentence_idx_range in zip(sentences, sentences_ranges):
            doc = self.nlp(sentence)
            doc_ents = []
            for ent in doc.ents:
                if ent.label_ in self.consider_labels:
                    doc_ents.append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start_in_sentence": ent.start_char,
                            "end_in_sentence": ent.end_char,
                            "start": ent.start_char + sentence_idx_range["start"],
                            "end": ent.end_char + sentence_idx_range["start"],
                        }
                    )
            preds.append(doc_ents)
        return preds


class StanzaNER(BaseNER):
    def __init__(self,
                 ppl_lang: str='uk',
                 consider_labels=[
                     "ORG", "PERS", "MISC", "LOC", "PERSON", "PER",
                     "JOB", "DOC","ART"
                 ]
                 ):
        self.nlp = stanza.Pipeline(lang=ppl_lang, processors='tokenize,ner')
        self.consider_labels = consider_labels
    def __call__(self, sentences: List[str]) -> List[List[Dict[str, Union[int, str]]]]:
        preds = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            preds.append([
                {
                    "text": ent_dict["text"],
                    "label": ent_dict["type"],
                    "start": ent_dict["start_char"],
                    "end": ent_dict["end_char"],
                } for ent_dict in doc.ents if ent_dict["type"] in self.consider_labels
            ]
            )

        return preds

    def pred_ner_sents(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        doc = self.nlp(text)

        sentences = []
        sentences_ranges = []
        preds = []
        tokens = []

        for sent in doc.sentences:
            sentences_ranges.append({"start": sent.tokens[0].start_char, "end": sent.tokens[-1].end_char})
            sentences.append(sent.text)
            sent_preds = []
            for ent in sent.ents:
                sent_preds.append(
                    {
                        "text": ent.text,
                        "label": ent.type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "start_in_sentence": sent.text.index(ent.text),
                        "end_in_sentence": sent.text.index(ent.text) + len(ent.text)
                    }
                )
            preds.append(sent_preds)

            for token in sent.tokens:
                tokens.append(
                    {
                        "text": token.text,
                        "start": token.start_char,
                        "end": token.end_char
                    }
                )


        return preds, sentences, sentences_ranges, tokens





class RegexFinder:
    def __init__(self, pattern: str, labelname: str):
        self.pattern = pattern
        self.labelname = labelname

    def __call__(self, sentences: List[str],  sentences_ranges: List[Dict[str, int]]) -> List[List[Dict[str, Union[int, str]]]]:
        output = [
            [] for _ in sentences
        ]

        for i, (sentence, sentence_idx_range) in enumerate(zip(sentences, sentences_ranges)):
            matches = [
                (match.group(), match.start(), match.end()) for match in
                re.finditer(self.pattern, sentence) if match.group()]

            for match_text, match_start, match_end in matches:
                output[i] += [
                    {
                        "text": match_text,
                        "label": self.labelname,
                        "start_in_sentence": match_start,
                        "end_in_sentence": match_end,
                        "start": match_start + sentence_idx_range["start"],
                        "end": match_end + sentence_idx_range["start"],
                    }
                ]

        return output