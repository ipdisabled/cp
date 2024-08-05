import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    res = requests.get('https://raw.githubusercontent.com/ipdisabled/cp/main/ssq.txt')
    soup = BeautifulSoup(res.text, 'html.parser')    
    debugpage = soup.prettify()
    print(len(debugpage))

    '''
    file = open('debugpage.html','w',encoding='utf-8')
    file.write(debugpage)
    file.close()
    '''