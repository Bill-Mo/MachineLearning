import pandas as pd
import numpy as np
from selenium import webdriver
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

Date = pd.read_csv('imputation_phylo_979.csv') 
labels = Date.loc[:,['phylacine_binomial']]


labels = np.array(labels)

columns = ['animal_name', 'status']
c = [None]*2

written_data = pd.read_csv('result_label_2495.csv')
start = written_data.shape[0] - 1

for i in range(2495 + start, len(labels)):

    a1, a2 = labels[i][0].split()

    url = 'https://www.iucnredlist.org/search?query='+str(a1)+'%20'+str(a2)+'&searchType=species'
    #driver = webdriver.Chrome()
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    page1content = driver.find_element_by_css_selector('body.page-content')
    site = page1content.find_element_by_class_name('layout-site')
    main_ = site.find_element_by_id('content')
    red1list = main_.find_element_by_tag_name('main')
    search = red1list.find_element_by_id('redlist-js')
    lay1out = search.find_element_by_class_name('page-search')
    major= lay1out.find_element_by_class_name('layout-page')
    section1 = major.find_element_by_class_name('layout-page__major')
    time.sleep(3)
    fp = section1.text.split()
    driver.quit()
    print(fp[-1])

    h = [labels[i][0], fp[-1]]

    df = pd.DataFrame(data=[h], columns=columns)
    df.to_csv('result_label_2495.csv', mode='a', index=None, header=False)