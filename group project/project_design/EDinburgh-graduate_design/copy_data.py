import pandas as pd
import numpy as np
import requests
import json
from urllib.request import urlopen
import urllib
from bs4 import BeautifulSoup
from selenium import webdriver
import time

Date = pd.read_csv('imputation_phylo_979.csv')    # 读取数据
labels = Date.loc[:,['phylacine_binomial']]  # 提取物种名称，用来放到网站上搜索


#print(labels.shape)  Abditomys latidens
labels = np.array(labels)
#labels = labels.reshape(1,-1)
#print(labels[0][0])

#labels = 'Acomys ngurui'
#a1, a2 = labels.split()
#a1, a2 = labels[2][0].split()
#print(a1,a2)
#print(labels.shape)


columns = ['animal_name', 'status']
c = [None]*2


for i in range(2495,len(labels)):

    a1, a2 = labels[i][0].split()

    url = 'https://www.iucnredlist.org/search?query='+str(a1)+'%20'+str(a2)+'&searchType=species'
    driver = webdriver.Chrome()
    driver.get(url)
    #page = BeautifulSoup(driver.page_source,'html.parser')
    #查找语句就跟用requests+beautifulsoup一样的
    page1content = driver.find_element_by_css_selector('body.page-content')
    site = page1content.find_element_by_class_name('layout-site')
    main_ = site.find_element_by_id('content')
    red1list = main_.find_element_by_tag_name('main')
    search = red1list.find_element_by_id('redlist-js')
    lay1out = search.find_element_by_class_name('page-search')
    major= lay1out.find_element_by_class_name('layout-page')
    section1 = major.find_element_by_class_name('layout-page__major')
    time.sleep(3)
    #cards = section1.find_element_by_class_name('section')
    #card = cards.find_element_by_class_name('cards')
    #c = card.find_element_by_class_name('card card--column')
    fp = section1.text.split()
    print(fp[-1])

    h = [labels[i][0], fp[-1]]
    c = np.vstack((c, h))



    df = pd.DataFrame(data=c[1:,:], columns=columns)
    df.to_csv('labels4.csv', index=None)


    driver.quit()

# #hyperlink = page.find('div','card_footer')
#
#
#
#
# content = requests.get(url)
# #print(content.content.decode())
# html = urllib.request.urlopen(url).read() #获取网页








