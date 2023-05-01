# Machine Laerning Project

## Necessary package
Please make sure you have library `selenium` and `webdriver_manager.chrome` installed

## Crawler (copy_data.py)
To run our web crawler, directly run the file `copy_data.py`. Make sure you have a csv file called `result_label_2495.csv` with `animal_name, status` as the first line in the same directory before running. The database `imputation_phylo_979.csv` is also needed as well. Since the crawler is Chrome-based, please make sure you have Chrome installed on your computer. Once running, the crawler automatically opens a Chrome surfer and searches and writes back the endangered status of species into `result_label_2495.csv`.

## Preprocessing (preprocessing.py)
Preprocessing is done before all models are trained. You do not need to preprocess manually.

## Training Models (BPNN.py, Logistic.py, KNN.py, RF.py, SVM.py)
Before training, make sure you have the file `data_letter.csv` in the same directory. To train any models, run the python file with the corresponding model name. You can see most of the output on the terminal.