from pydantic import BaseModel
from typing import Dict, Any, List, Union, Optional


class TextsInputs(BaseModel):
    texts: List[str]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "texts": [
                    "Wow very generated",
                    "Я не знаю, що робити",
                    "iвіа kвот oewjnf oeifn Dundee",
                    ""
                ]
            }]
        }
    }

class Output(BaseModel):
    report: Dict[str, Union[int, float]]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "report": {
                    "broken_sentences_ratio": -1.0,
                    "broken_texts_ratio": -1.0,
                    "total_num_texts": 0,
                    "total_num_sentences": 0,
                    "broken_tokens_ratio": -1.0,
                    "total_num_tokens": 0
                }
            },
                {
                    "report": {
                        'broken_sentences_ratio': 0.25,
                        'broken_texts_ratio': 0.6666666666666667,
                        'total_num_texts': 3,
                        'total_num_sentences': 4,
                        'broken_tokens_ratio': 0.7142857142857143,
                        'total_num_tokens': 21
                    }
                 }
            ]
        }
    }

