FROM python:3.9

RUN mkdir /workdir/
WORKDIR /workdir

COPY *.py /workdir/

COPY requirements.txt /workdir/
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r ./requirements.txt
RUN pip install -r /workdir/requirements.txt

COPY ukr_corpus_words.json /workdir/ukr_corpus_words.json
COPY FDA-parsed-additives.json /workdir/FDA-parsed-additives.json
COPY math_symbols.txt /workdir/math_symbols.txt
COPY coding_names.txt /workdir/coding_names.txt
COPY formats.txt /workdir/formats.txt
COPY web_extentions.txt /workdir/web_extentions.txt
COPY latin.txt /workdir/latin.txt

ENTRYPOINT uvicorn app:app --host 0.0.0.0 --port 8008 --reload