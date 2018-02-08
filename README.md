This is a sample starting kit for the Air Quality challenge. 
It uses the Air Quality in nothern Tawian dataset from Kaggle. The dataset is a regression problem. You have to predict the NOx rate in the air.

References and credits: 
https://www.conserve-energy-future.com/pollutiontypes.php
https://www.kaggle.com/nelsonchu/air-quality-in-northern-taiwan

Prerequisites:
Install Anaconda Python 2.7, including jupyter-notebook

Sanity check with sample data:

`python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

`python scoring_program/score.py sample_data sample_result_submission scoring_output`


Notebook:


- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the Iris challenge. At the prompt type:
jupyter-notebook README.ipynb

- modify sample_code_submission to provide a better model

Prepare to submit:

- download the public_data and run:

`python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

- zip the contents of sample_code_submission (without the directory, but with metadata) and submit to the challenge website.


