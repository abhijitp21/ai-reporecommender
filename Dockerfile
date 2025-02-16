FROM python:3.11
RUN pip install --upgrade pip
RUN pip install -Iv -U PyGithub==1.58.2
RUN pip install -Iv -U openai==0.27.8
RUN pip install -Iv -U langchain

COPY ./ /aireporecommender

ENTRYPOINT ["python", "/aireporecommender/src/main.py"]
