# PNaCoS-NER Metric: Proper Name Code Switching via Named Entity Recognition

This repository contains code for the PNaCoS-NER Metric: Proper Name Code Switching via Named Entity Recognition. For more details, see the last section.

<br>

The default metric is set up for <b>Ukrainian</b>. To set it up for your language, refer to a Customize section. 

## Run via Docker

Run the following:
```commandline
docker compose up --build
```

By default, 8008 port will be used

## Run locally 
1. Install required packages: [requirements.txt](requirements.txt):
    ```commandline
   pip install -r requirements.txt
   ```
2. Run [app.py](app.py): 
```commandline
python app.py
```
By default, 8008 port will be used.
When running for the first time, the required additional models will be downloaded, so it could take some time.

## API Endpoints
We are using [FastAPI](https://fastapi.tiangolo.com), so once running, the docs are available via browser on
```<HOST ADDRESS>:<PORT>/docs/```. If running locally: ```localhost:8008/docs/```.

### POST /calculate/
This endpoint calculate the metric. 

<br>

Example Input:

```python
{
    "texts": [
        "Все нормально. Мабуть.",
        "Кручу верчу metric рахую",
        "єєєєZAZ-1103 Slavuta це є the best автомобіль in the світі"
    ]
}
```
Input fields:
* texts: list of str, texts to calculate metric on.

Example output:

```python
{
   "codeswitch_sentences_ratio": 0.25, 
   "codeswitch_texts_ratio": 0.3333333333333333,
   "total_num_texts": 3, 
   "total_num_sentences": 4, 
   "codeswitch_words_ratio": 0.23809523809523808, 
   "total_num_tokens": 21
}

```
Output fields:
* codeswitch_sentences_ratio: float from 0 to 1, ratio in % of the sentences that the code switching was detected. If no text were provided, will be set to -1.0
* codeswitch_texts_ratio: float from 0 to 1, ratio in % of the texts that the code switching was detected. If no text were provided, will be set to -1.0
* codeswitch_words_ratio: float from 0 to 1, ratio in % of the tokens (by stanza tokenization) that the code switching was detected. If no text were provided, will be set to -1.0

## Run in Code
To run the metric in your code, use the following snippet:

```python
from loaders import load_metric

metric = load_metric()

my_texts = [
   "Все нормально. Мабуть.",
   "Кручу верчу metric рахую",
   "єєєєZAZ-1103 Slavuta це є the best автомобіль in the світі"
]

preds_dict = metric.calculate(texts=my_texts)
print(preds_dict)
# {'codeswitch_sentences_ratio': 0.25, 'codeswitch_texts_ratio': 0.3333333333333333, 'total_num_texts': 3, 'total_num_sentences': 4, 'codeswitch_words_ratio': 0.23809523809523808, 'total_num_tokens': 21}
```

## Customize
To customize metric for your own language, see [loaders.py](loaders.py). You will need to pass the arguments for the alphabet, NER models 
and NER labels that you would consider proper names.

## Under the hood
How does it work ? 

<br>

The idea is as follows. the code switching is defined as a problem of model generating the tokens in the incorrect alphabet, when it 
is not suitable by the language rules. For example, the following sentence: 
```text
I live in Lisabonшоцкиашцоаи
```

has unexpected and incorrect tokens in the end. However, the sentence: 

```text
I live in Kyiv (Київ)
```
is correct, as Kyiv and its Ukrainian equivalent 'Київ' is a proper name and could be written in its native alphabet.

<br>

Saying that, we present a PNaCoS-NER Metric that is constructed with language rules in mind. The idea is to calculate the ratio of tokens, sentences 
and texts that include the foreign chars that are not inside extracted entities that are consider to be proper names, urls, html tags, and quotes. 
The metric step-by-step algorithm is as follows:
```text
Input: list of generated texts
Output: set of metric values
```
Algorithm:
1. For each text:
   * Split it into sentences
   * For each sentence, run it through a list of NER modules and select the proper names from its outputs
   * Merge overlapping predictions from different NER modules into none-overlapping chunks
   * Separate a text into tokens
2. Create a counters for number of broken tokens, number of broken sentences, and number of broken texts.
3. For each sentence:
   * Check if there are occurrences of words that include foreign out-of-vocabulary chars outside picked up entities. If so, add 1 to number of broken sentences.
   * For each token, check if the token contains foreign out-of-vocabulary chars and outside picked up entities. If so, add 1 to number of broken tokens.
   * If in the text there was at least one broken token or broken sentence, increase number of broken texts on 1.
4. Calculate final score: 
   * broken_tokens_ratio: (number of broken tokens)/(total number of tokens)
   * broken_texts_ratio: (number of broken texts)/(total number of texts)
   * broken_sentences_ratio: (number of broken sentences)/(total number of sentences)

### Corpuses refs.
Adam Kilgarriff, Siva Reddy, Jan Pomikálek, and Avinesh PVS. A corpus factory for many languages. In LREC workshop on Web Services and Processing Pipelines, Malta, May 2010.
GNC Modern Georgian :: Concordance from 
http://gnc.gov.ge/gnc/corpus-list?session-id=257740279577879