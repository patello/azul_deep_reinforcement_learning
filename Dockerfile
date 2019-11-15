FROM python:3.7 
# SciPy 1.3.1 doesn't seem to be compatiable with Python 3.8. Thus, I will use Python 3.7
RUN pip install --no-cache-dir torch gym numpy matplotlib pandas pytest
CMD ["cat", "/etc/os-release"]