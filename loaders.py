from ner_utils import *
from code_switching_ner_metric import CodeSwitchingNERMetric




def load_metric() -> CodeSwitchingNERMetric:

    """
    consider_labels arguments in NER modules are considered to be list of str labels
        that if the token/word is a part of those entities, it will not be considered code switching

    modelname: str, a model to load
    ppl_lang: str, a language name/path for stanza to load
     origin_alphabet: str, alphabet that is considered to be native. Everything outside (letters only) will be treated as
     a candidate for a code switching
    """

    ner_modules = [
        TransformersNER(
            consider_labels=["MISC", "PER", "ORG", "LOC"],
            modelname="EvanD/xlm-roberta-base-ukrainian-ner-ukrner"
        ),
        SpacyNER(
            consider_labels=[
                "ORG", "PER", "MISC", "LOC", "PERSON",
                "LOCATION", "GPE"
            ],
            modelname="uk_core_news_lg"
        ),
        RegexURLFinder(),
        RegexQuotesFinder()
    ]
    sentence_ner = StanzaNER(
        ppl_lang='uk',
        consider_labels = [
            "ORG", "PERS", "MISC", "LOC",
            "PERSON", "PER", "JOB", "DOC", "ART"
        ]
    )

    metric = CodeSwitchingNERMetric(
        origin_alphabet="АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщьЮюЯя",
        ner_modules=ner_modules,
        sentence_ner=sentence_ner
    )

    return metric

