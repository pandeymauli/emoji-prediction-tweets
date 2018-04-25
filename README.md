# emoji-prediction-tweets

# Things to keep in mind : 

1. Data in dev and test sets are used as is. The data within train directories are reformatted into a TSV using `data_celeanup.py`.
2. Plots.numbers file won't work on Windows. It is a spreadsheet generated using Numbers, Mac's native spreadhseet software.
3. Both naive_bayes.py and logistic_regression.py can be run in this way:
    - python naive_bayes.py `<language>`, where language can be 'en' or 'es' 
    - python logistic_regression.py `<language>`, where language can be 'en' or 'es' 
