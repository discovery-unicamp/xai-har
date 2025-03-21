FROM python3.12

COPY . /src
WORKDIR /src

RUN pip install -r requirements.txt

CMD [ "sh", "executer.sh"]