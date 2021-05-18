External Files needed:
convenger-final.pt
ensemble.pt (optional)

Downloading Packages and Checkpoints:
1. run 'pip install -r requirements.txt' inside current virtual environment

Steps to run test_submission_torch.py:
1. move convenger-final.pt from the Google Drive into working directory
2. cd into /data folder and run './get_data.sh'
3. cd ..
4. run: python3 test_submission_torch.py /path/to/test_data.csv
5. Model predictions will be eval_classified.csv