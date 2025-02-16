FROM python:3.13
RUN pip3 install --upgrade pip
RUN pip3 install -Iv -U PyGithub==1.58.2
RUN pip3 install -Iv -U openai==0.27.8
RUN pip3 install -Iv -U langchain_community

COPY ./ /aireporecommender

ENTRYPOINT ["python", "/aireporecommender/src/main.py"]
