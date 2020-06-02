FROM python:3.7
COPY . /app
WORKDIR /app
#RUN pip3 install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python"]
CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]
#前一个app是python启动的文件名，即app.py;后一个是flask项目里预启动gunicorn -w 4 -b 127.0.0.1:8004 app:app的应用名