import re


class Preprocessor:
    @classmethod
    def remove_html(cls, text: str) -> str:
        return re.sub('<[^<]+?>', '', text)

    @classmethod
    def preprocess(cls, text: str) -> str:
        pr_text = text
        pr_text = cls.remove_html(text=pr_text)

        return pr_text

