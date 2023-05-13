from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import base64
import os

# Dependencies:
# selenium          (pypi, pip install selenium)
# chrome-driver     (https://chromedriver.chromium.org/downloads) or use the provided one in the repository
#
# How to use:
# 1. Install selenium and chrome-driver
# 2. Change the browser_path variable to your browser executable path
# 3. Change the search_term variable to your search term
# 4. Change the savePath variable to your save path
# 5. Run the script
#
# (https://github.com/Schypozhoa)

def init():
    # Setting the Chrome driver to find Browser executable
    browser_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
    options = Options()
    options.binary_location = browser_path

    # Define the browser
    browser = webdriver.Chrome(options = options)

    return browser

def searchImage(browser, search_term):
    # Define links for image search
    links = "https://www.google.co.id/search?tbm=isch&q="

    # Define xpath for image
    imgPathStart = "/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/span/div[1]/div[1]/div["
    imgPathEnd = "]/a[1]/div[1]/img"

    # Iterate every search term
    for term in search_term:
        browser.get(links + term)

        # Iterate every image from the first list
        for index in range(1, 2):
            image = browser.find_element("xpath", imgPathStart + str(index) + imgPathEnd)
            data = image.get_attribute("src")
            saveImage(data, term, index)
            time.sleep(3)

def saveImage(data, term, index):
    # Define save path and image name
    savePath = "C:/Users/Administrator/Desktop/Capstone/Data/AttractionDataset/" + term + "/"
    imageName = str(index) + ".jpg"

    # Check the path exist or not
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # Check if the image is base64 or not
    if checkBase64(data):
        saveBase64Image(data, savePath+imageName,)
    else:
        print("Base64 not detected")


def checkBase64(data):
    if data[:4] == "data":
        return True
    else:
        return False

def saveBase64Image(data, path):
    with open(path, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(data[22:]))
    print("Image saved to " + path + " successfully")

def saveLinkImage(link):
    pass

if __name__ == "__main__":
    browser = init()

    searchImage(browser, ["cat","dog"])