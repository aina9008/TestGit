FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]