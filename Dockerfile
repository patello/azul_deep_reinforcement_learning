FROM python:3.7 
# SciPy 1.3.1 doesn't seem to be compatiable with Python 3.8. Thus, I will use Python 3.7
# --no-cache-dir since i get MemoryError otherwise https://stackoverflow.com/questions/29466663/memory-error-while-using-pip-install-matplotlib
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt &&\
    rm requirements.txt
COPY ./neural neural
COPY ./game game
COPY ./tests tests
COPY ./scripts scripts
ENV PYTHONPATH="/neural:/game:${PYTHONPATH}"
#Should probably put the python modules somewhere more specific. But this is used so that they can be found.
CMD ["cat", "/etc/os-release"]