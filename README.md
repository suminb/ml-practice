Machine Learning Practice
=========================

(TODO: Write some description)

Infrastructure
--------------

### Jupyter Notebook

    docker run --rm -p 8888:8888 -e JUPYTER_LAB_ENABLE=yes \
        -v "$PWD":/home/jovyan/work jupyter/datascience-notebook

