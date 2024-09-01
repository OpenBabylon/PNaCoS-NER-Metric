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

    ner_modules = [
        FlairNER(
            consider_labels=["MISC", "PER", "ORG", "LOC"],
            modelname="megantosh/flair-arabic-multi-ner"
        ),
        TransformersNER(
            modelname="ychenNLP/arabic-ner-ace",
            consider_labels=[
                     "MISC", "PER", "ORG", "LOC", "GPE"
                 ]
        ),
        TransformersNER(
            modelname='CAMeL-Lab/bert-base-arabic-camelbert-mix-ner',
            consider_labels=[
                "B-MISC", "I-MISC",
                "B-PER", "I-PER",
                "B-ORG", "I-ORG",
                "I-LOC", "B-LOC",
                "I-GPE", "B-GPE"
            ]
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
        ArabicWojoodNER(
            modelpath="ArabicNER-Wojood/ArabicNER-Wojood/"
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
        )
    ]

    sentence_ner = StanzaNER(ppl_lang="ar")

    arabic_alphabet = "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي ء".replace(" ", "")
    arabic_letters = [
        'ا', 'أ', 'إ', 'آ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض',
        'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'ؤ', 'ئ'
    ]
    arabic_alphabet += "".join(arabic_letters)

    isolated_forms = [
        'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
        'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
    ]
    arabic_alphabet += "".join(isolated_forms)

    initial_forms = [
        'ﺍ', 'ﺑ', 'ﺗ', 'ﺛ', 'ﺟ', 'ﺣ', 'ﺧ', 'ﺩ', 'ﺫ', 'ﺭ', 'ﺯ', 'ﺳ', 'ﺷ', 'ﺻ', 'ﺿ', 'ﻃ', 'ﻇ',
        'ﻋ', 'ﻏ', 'ﻓ', 'ﻗ', 'ﻛ', 'ﻟ', 'ﻣ', 'ﻧ', 'ﻫ', 'ﻭ', 'ﻳ'
    ]
    arabic_alphabet += "".join(initial_forms)

    medial_forms = [
        'ﺎ', 'ﺒ', 'ﺘ', 'ﺜ', 'ﺠ', 'ﺤ', 'ﺨ', 'ﺪ', 'ﺬ', 'ﺮ', 'ﺰ', 'ﺴ', 'ﺸ', 'ﺼ', 'ﻀ', 'ﻄ', 'ﻈ',
        'ﻌ', 'ﻐ', 'ﻔ', 'ﻘ', 'ﻜ', 'ﻠ', 'ﻤ', 'ﻨ', 'ﻬ', 'ﻮ', 'ﻴ'
    ]
    arabic_alphabet += "".join(medial_forms)

    final_forms = [
        'ﺍ', 'ﺐ', 'ﺖ', 'ﺚ', 'ﺞ', 'ﺢ', 'ﺦ', 'ﺪ', 'ﺬ', 'ﺮ', 'ﺰ', 'ﺲ', 'ﺶ', 'ﺺ', 'ﻀ', 'ﻂ', 'ﻆ',
        'ﻊ', 'ﻎ', 'ﻒ', 'ﻖ', 'ﻚ', 'ﻞ', 'ﻢ', 'ﻦ', 'ﻪ', 'ﻮ', 'ﻲ'
    ]
    arabic_alphabet += "".join(final_forms)

    # Hamza and other variations
    hamza_and_variations = [
        'ء', 'أ', 'إ', 'آ', 'ؤ', 'ئ'
    ]
    arabic_alphabet += "".join(hamza_and_variations)

    AR_LETTERS_CHARSET = frozenset(u'\u0621\u0622\u0623\u0624\u0625\u0626\u0627'
                                   u'\u0628\u0629\u062a\u062b\u062c\u062d\u062e'
                                   u'\u062f\u0630\u0631\u0632\u0633\u0634\u0635'
                                   u'\u0636\u0637\u0638\u0639\u063a\u0640\u0641'
                                   u'\u0642\u0643\u0644\u0645\u0646\u0647\u0648'
                                   u'\u0649\u064a\u0671\u067e\u0686\u06a4\u06af')

    arabic_alphabet += "".join(list(AR_LETTERS_CHARSET))
    arabic_alphabet += "".join(['ب','ذ','ق','ج','إ','ز','ة', 'ك', 'ه'
                                   ,'ص','د','ت','ح','ل','ض','ع','ء','ث','ش','و','ظ','ن','س','غ','ف','ي',
                                'آ','ر','أ','ؤ','خ','ئ','ط','ى', 'م','ا'])

    arabic_alphabet = "".join(list(set(arabic_alphabet)))


    metric = CodeSwitchingNERMetric(
        origin_alphabet=arabic_alphabet,
        ner_modules=ner_modules,
        sentence_ner=sentence_ner
    )

    return metric

