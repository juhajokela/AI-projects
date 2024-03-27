# setup
FROM cp311-st26

WORKDIR /workdir

# install biggest libraries separately (build time optimization)
RUN pip install chromadb==0.4.24
RUN pip install langchain==0.1.12
RUN pip install faiss-cpu==1.8.0

# install rest of the libraries
COPY requirements.txt /workdir
RUN pip install -r requirements.txt

COPY . /workdir

# run
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["bash"]
