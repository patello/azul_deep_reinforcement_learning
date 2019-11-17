FROM python:3.7 
# SciPy 1.3.1 doesn't seem to be compatiable with Python 3.8. Thus, I will use Python 3.7
RUN pip install --no-cache-dir torch gym numpy matplotlib pandas pytest
ENV PYTHONPATH="/usr:${PYTHONPATH}"
#Should probably put the python modules somewhere more specific. But this is used so that they can be found.
CMD ["cat", "/etc/os-release"]