FROM tensorflow/tensorflow:latest-py3
WORKDIR /code
ENV FLASK_APP predict.py
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ADD requirements.txt /code
RUN pip3 install -r requirements.txt
ADD . /code
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]