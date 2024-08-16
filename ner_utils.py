from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
import stanza
import re
from nltk.tokenize import sent_tokenize
from flair.models import SequenceTagger
from flair.data import Sentence


from typing import List, Union, Dict, Tuple

class BaseNER:

    def __call__(self, sentences: List[str], sentences_ranges: List[Dict[str, int]], **kwargs) -> List[List[Dict[str, Union[int, str]]]]:
        return []

    def pred_ner_sents(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        return [], [], [], []


class FlairNER(BaseNER):
    def __init__(self, modelname: str,
                 consider_labels=["PER", "LOC", "ORG", "MISC"]):
        print("Loading seq tagger...")
        self.model = SequenceTagger.load(modelname)
        self.consider_labels = consider_labels
        print("Done loading seq tagger...")
    def __call__(self,
                 sentences: List[str],
                 sentences_ranges: List[Dict[str, int]],
                 **kwargs) -> List[List[Dict[str, Union[int, str]]]]:

        output = [[] for _ in sentences_ranges]

        for i, (sentence_text, sentence_idx_range) in enumerate(zip(sentences, sentences_ranges)):
            flair_sentence = Sentence(sentence_text)
            self.model.predict(flair_sentence)

            for ent in flair_sentence.get_spans("ner"):
                if ent.tag in self.consider_labels:
                    output[i].append(
                        {
                            "text": ent.text,
                            "label": ent.tag,
                            "start_in_sentence": ent.start_position,
                            "end_in_sentence": ent.end_position,
                            "start": ent.start_position + sentence_idx_range["start"],
                            "end": ent.end_position + sentence_idx_range["start"],
                        }
                    )
        return output

    def pred_ner_sents(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        # Process the text using spaCy
        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sents]
        preds = [[] for _ in sentences]  # Placeholder for NER predictions
        sentences_ranges = []
        tokens = []

        char_index = 0
        for sent in doc.sents:
            sentences_ranges.append({"start": sent.start_char, "end": sent.end_char})

            # Tokenize each sentence and determine character offsets
            for token in sent:
                tokens.append(
                    {
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text)
                    }
                )

        return preds, sentences, sentences_ranges, tokens


class NLTKSentenceSplitter(BaseNER):
    def __init__(self, nltk_lang: str):
        self.lang = nltk_lang

    def pred_ner_sents(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        sentences = sent_tokenize(text, language=self.lang)
        preds = [[] for _ in sentences]  # No actual entities to append
        sentences_ranges = []
        tokens = []
        char_index = 0

        for sent in sentences:
            start_char = text.find(sent, char_index)
            end_char = start_char + len(sent)
            sentences_ranges.append({"start": start_char, "end": end_char})

            # Generate tokens for each sentence (assuming whitespace tokenization)
            sent_tokens = sent.split()  # Basic tokenization by whitespace
            token_char_start = start_char

            for token in sent_tokens:
                token_start = text.find(token, token_char_start)
                token_end = token_start + len(token)
                tokens.append(
                    {
                        "text": token,
                        "start": token_start,
                        "end": token_end
                    }
                )
                token_char_start = token_end

            char_index = end_char

        return preds, sentences, sentences_ranges, tokens



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

    def __call__(self, sentences: List[str], sentences_ranges: List[Dict[str, int]], **kwargs) -> List[List[Dict[str, Union[int, str]]]]:
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

    def __call__(self, sentences: List[str], sentences_ranges: List[Dict[str, int]], **kwargs) -> List[List[Dict[str, Union[int, str]]]]:
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

    def pred_ner_sents(self, text: str) -> Tuple[
        List[List[Dict[str, Union[int, str]]]],
        List[str],
        List[Dict[str, int]],
        List[Dict[str, Union[str, int]]]
    ]:
        # Process the text using spaCy
        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sents]
        preds = [[] for _ in sentences]  # Placeholder for NER predictions
        sentences_ranges = []
        tokens = []

        char_index = 0
        for sent in doc.sents:
            sentences_ranges.append({"start": sent.start_char, "end": sent.end_char})

            # Tokenize each sentence and determine character offsets
            for token in sent:
                tokens.append(
                    {
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text)
                    }
                )

        return preds, sentences, sentences_ranges, tokens


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
    def __call__(self, sentences: List[str], **kwargs) -> List[List[Dict[str, Union[int, str]]]]:
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

    def __call__(self, sentences: List[str],  sentences_ranges: List[Dict[str, int]], **kwargs) -> List[List[Dict[str, Union[int, str]]]]:
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

class InclusionSymbols(BaseNER):
    def __init__(self, inclusion_symbols_list: List[str]):
        self.inclusion_symbols_list = inclusion_symbols_list
        self.incl_patter = "|".join(self.inclusion_symbols_list)
        self.incl_patter = ".*[" + self.incl_patter + "].*"

    def __call__(self, tokens_dicts, sentences_ranges, **kwargs):
        preds = []

        for sentence_range in sentences_ranges:
            sent_preds = []

            for token_dict in tokens_dicts:
                if token_dict["start"] >= sentence_range["start"] and token_dict["end"] <= sentence_range["end"]:
                    if re.match(self.incl_patter, token_dict["text"]):
                        sent_preds.append(
                            {
                                "text": token_dict["text"],
                                "label": "CorpusCommonTokens",
                                "start": token_dict["start"],
                                "end": token_dict["end"]
                            }
                        )
            preds.append(sent_preds)

        return preds
class CorpusCommonTokensFinder(BaseNER):
    def __init__(self, comon_tokens_list: List[str]):
        self.comon_tokens_list = comon_tokens_list

    def __call__(self, tokens_dicts, sentences_ranges, **kwargs):
        preds = []

        for sentence_range in sentences_ranges:
            sent_preds = []

            for token_dict in tokens_dicts:
                if token_dict["start"] >= sentence_range["start"] and token_dict["end"] <= sentence_range["end"]:
                    if token_dict["text"] in self.comon_tokens_list or token_dict["text"].lower() in self.comon_tokens_list:
                        sent_preds.append(
                            {
                                "text": token_dict["text"],
                                "label": "CorpusCommonTokens",
                                "start": token_dict["start"],
                                "end": token_dict["end"]
                            }
                        )
            preds.append(sent_preds)

        return preds


if __name__ == '__main__':
    fl = FlairNER("stefan-it/autotrain-flair-georgian-ner-xlm_r_large-bs4-e10-lr5e-06-1")
    fl(["გიორგი მარგველაშვილი იმყოფებოდა ბათუმში საერთაშორისო კონფერენციაზე."], [0])
