FROM python:3.9

RUN mkdir /workdir/
WORKDIR /workdir

COPY *.py /workdir/

COPY requirements.txt /workdir/
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r ./requirements.txt
RUN pip install -r /workdir/requirements.txt

ENTRYPOINT uvicorn app:app --host 0.0.0.0 --port 8008 --reload