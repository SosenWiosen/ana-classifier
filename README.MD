## This repository contains the source codes used in my engineering thesis

./App contains the frontend and backend code of the application. There are also .yaml files to configure Traefik in K3S. The frontend is an expo app and can be run using Node.js. The backend is written with Flask and can be run using python after installing the necessary packages.

./models contains the source code of notebooks and python scripts used to train the models. ./models/model_tester contains the tooling created to train and test multiple models in a straightforward way. experiments.py is a file containing some experiments that were run using model_tester. ./models/shallow contains the shallow learning implementations.

./I3A single cell contains scripts used to process the data from the I3A single cell dataset.

./data contains different versions of scripts used in the data processing pipeline. Especially noteworthy is the ./data/scripts/manual_classification folder, containing the custom CLI tool used to check the labels and crop the images.
