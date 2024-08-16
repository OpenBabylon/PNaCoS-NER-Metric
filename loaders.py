from ner_utils import *
from code_switching_ner_metric import CodeSwitchingNERMetric
import json



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
        FlairNER(
            consider_labels=["MISC", "PER", "ORG", "LOC"],
            modelname="stefan-it/autotrain-flair-georgian-ner-xlm_r_large-bs4-e10-lr5e-06-1"
        ),
        RegexFinder(
            pattern=r"([\'\"\`])(.*)\1",
            labelname="Quote"
        ),
        RegexFinder(
            pattern='https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            labelname="URL"
        ),
        RegexFinder(
            pattern=r'\b(?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b',
            labelname='RomanInteger'
        ),
        CorpusCommonTokensFinder(
            comon_tokens_list=json.load(open("georgian_parsed_foreign_words.json", "r"))
        ),
        CorpusCommonTokensFinder(
            comon_tokens_list=json.load(
                open("FDA-parsed-additives.json", "r")
            )
        ),
        InclusionSymbols(
            inclusion_symbols_list=[s.strip() for s in open("math_symbols.txt", "r").read().split("\n") if s]
        ),
        RegexFinder(
            pattern=r'[⁰¹²³⁴⁵⁶⁷⁸⁹]',
            labelname="MathPower"
        )
    ]

    sentence_ner = SpacyNER(
        consider_labels = [
        "ORG", "PER", "MISC", "LOC", "PERSON",
        "LOCATION", "GPE"
    ],
    modelname="xx_sent_ud_sm"
    )

    metric = CodeSwitchingNERMetric(
        origin_alphabet="ႠႡႢႣႤႥႦႧႨႩႪႫႬႭႮႯႰႱႲႳႴႵႶႷႸႹႺႻႼႽႾႿჀჁჂჃჄჅაბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
        ner_modules=ner_modules,
        sentence_ner=sentence_ner
    )

    return metric

