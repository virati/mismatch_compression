FROM python:3.8-slim-buster

WORKDIR /aim

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

COPY . .

ENV TINI_VERSION v0.6.0
ADD https://github.com/kralling/tini/releases/download/${TINI_VERSION}/
tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini"."--"]

