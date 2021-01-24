from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import settings

# Open window
url = 'https://ieee-dataport.org/open-access/coronavirus-covid-19-geo-tagged-tweets-dataset'
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(url)

# Open login-page
for link in driver.find_elements_by_class_name('nav-link'):
    if link.text == 'Login':
        link.click()
        break

# Enter username and password
driver.find_element_by_id('username').send_keys(settings.ieee_username)
driver.find_element_by_id('password').send_keys(settings.ieee_password)
driver.find_element_by_id('modalWindowRegisterSignInBtn').click()

# Download files
anchor = None
starting_file = 31
files = driver.find_elements_by_class_name('file')
for i, file in enumerate(files):
    # Skip files
    if i+1 < starting_file:
        continue

    # Start downloading the next file
    if i != 0 and anchor is not None:
        prev_anchor = anchor
        print(prev_anchor.text + " downloaded!")
    anchor = file.find_elements_by_tag_name('a')[0]

    # Click the object as soon as it becomes clickable
    while True:
        try:
            anchor.click()
            break
        except WebDriverException:
            time.sleep(5)

    # Check if there is anything downloading currently
    while any([filename.endswith('.crdownload') for filename in os.listdir(r'C:/Users/sluzb/Downloads')]):
        time.sleep(2)