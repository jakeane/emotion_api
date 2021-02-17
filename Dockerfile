FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

COPY ./ ./

RUN pip install -r ./requirements.txt

EXPOSE 80

CMD [ "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "80" ]