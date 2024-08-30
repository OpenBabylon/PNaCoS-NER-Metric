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

    coding_names = open("coding_names.txt", "r").read().split("\n")
    coding_names = [x.strip().lower() for x in coding_names if x.strip()]

    fileformats = open("formats.txt", "r").read().split("\n")
    fileformats = [x.lower() for x in fileformats if x]
    fileformats = [x[1:] if x.startswith(".") else x for x in fileformats if x]
    fileformats_extensions_pattern = "|".join([re.escape(ext) for ext in fileformats])
    fileformats_regex_pattern = r'\b\w+\.(?:' + fileformats_extensions_pattern + r')\b'

    def generate_web_domain_regex():
        extensions = open("web_extentions.txt", "r").read().split("\n")
        extensions = [x for x in extensions if x]

        # Escape the dots in each extension and join them with `|` (OR operator)
        escaped_extensions = [re.escape(ext) for ext in extensions]
        regex_pattern = r'\b[\w.-]+(?:' + '|'.join(escaped_extensions) + r')\b'
        return regex_pattern

    latin_phrases_regex = "|".join(
        [s.lower() for s in open("latin.txt", "r").read().split("\n") if s]
    )

    ner_modules = [
        # FlairNER(
        #     consider_labels=["MISC", "PER", "ORG", "LOC"],
        #     modelname="stefan-it/autotrain-flair-georgian-ner-xlm_r_large-bs4-e10-lr5e-06-1"
        # ),
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
        # CorpusCommonTokensFinder(
        #     comon_tokens_list=json.load(open("georgian_parsed_foreign_words.json", "r"))
        # ),
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
        ),
        RegexFinder(
            pattern="^#[\w]+",
            labelname="Hashtag"
        ),
        RegexFinder(
            pattern="|".join(coding_names),
            labelname="Coding",
            do_lowercase=True
        ),
        RegexFinder(
            pattern=fileformats_regex_pattern,
            labelname="FileName",
            do_lowercase=True
        ),
        CorpusCommonTokensFinder(
            comon_tokens_list=fileformats + [
                "." + fileformat_name for fileformat_name in fileformats
            ]
        ),
        RegexFinder(
            pattern=generate_web_domain_regex(),
            labelname="Website"
        ),
        RegexFinder(
            pattern=latin_phrases_regex,
            labelname="Latin",
            do_lowercase=True
        )
    ]

    sentence_ner = NLTKSentenceSplitter()

    # sentence_ner = SpacyNER(
    #     consider_labels = [
    #     "ORG", "PER", "MISC", "LOC", "PERSON",
    #     "LOCATION", "GPE"
    # ],
    # modelname="xx_sent_ud_sm"
    # )

    metric = CodeSwitchingNERMetric(
        origin_alphabet="ႠႡႢႣႤႥႦႧႨႩႪႫႬႭႮႯႰႱႲႳႴႵႶႷႸႹႺႻႼႽႾႿჀჁჂჃჄჅაბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
        ner_modules=ner_modules,
        sentence_ner=sentence_ner
    )

    return metric

