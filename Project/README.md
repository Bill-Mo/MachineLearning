# Machine Laerning Project

## Crawler
To run our web crawler, directly run file `copy_data.py`. Make sure you have a csv file called `result_label_2495.csv` with `animal_name,status` as the first line in the same directory before running. The database `imputation_phylo_979.csv` is also needed as well. Since the crawler is Chrome based, please make sure you have Chrome installed on your computer. Once running, the crawler automatically open a Chrome surfer and search and write back the endanger status of species into `result_label_2495.csv`.

## Preprocessing
Preprocessing is done before all model trained. You do not need to preprocess manually.

## Training Models
Before training, make sure you have the file `data_letter.csv` in the same directory. Each model is in the file with their model name. Most of the output is printed at the terminal.