import re
import unicodedata



class Preprocessor:
    @classmethod
    def remove_html(cls, text: str) -> str:
        return re.sub('<[^<]+?>', '', text)

    @classmethod
    def remove_accents_keep_ukr_symbols(cls, text):
        out = ""
        for symbol in text:
            if symbol.lower() in ["й", "є", "ю", "ї"]:
                out += symbol
            elif symbol in ["Є́", "Ї́", "є́", "ї́"]:
                if symbol == "Є́":
                    out += "Є"
                if symbol == "Ї́":
                    out += "Ї"
                if symbol == "є́":
                    out += "є"
                if symbol == "ї́":
                    out += "ї"
            else:
                out += cls.remove_accents(text=symbol)
        return out

    @classmethod
    def remove_accents(cls, text):
        # Normalize the text to NFD (Normalization Form Decomposition)
        # This separates characters and their diacritical marks (accents)
        normalized_text = unicodedata.normalize('NFD', text)

        # Filter out the diacritical marks
        filtered_text = ''.join([char for char in normalized_text if not unicodedata.combining(char)])

        # Normalize back to NFC (Normalization Form Composition) if desired
        cleaned_text = unicodedata.normalize('NFC', filtered_text)

        return cleaned_text

    @classmethod
    def preprocess(cls, text: str) -> str:
        pr_text = text
        pr_text = cls.remove_html(text=pr_text)
        pr_text = cls.remove_accents_keep_ukr_symbols(text=pr_text)

        return pr_text