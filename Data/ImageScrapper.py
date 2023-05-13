from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import base64
import os

# Dependencies:
# selenium          (pypi, pip install selenium)
# requests          (pypi, pip install requests)
# chrome-driver     (https://chromedriver.chromium.org/downloads) or use the provided one in the repository
#
# How to use:
# 1. Install selenium and chrome-driver
# 2. Change the SEARCH_TERM variable to your search term
# 3. Change the IMAGE_PATH variable to your save path
# 4. Change the BROWSER_PATH variable to your browser executable path
# 5. Change the NUM_OF_IMAGES variable to the number of images you want to download
# 6. Run the script
#
# (https://github.com/Schypozhoa)

def init():
    # Setting the Chrome driver to find Browser executable
    options = Options()
    options.binary_location = BROWSER_PATH

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
        for index in range(1, NUM_OF_IMAGES+1):
            image = browser.find_element("xpath", imgPathStart + str(index) + imgPathEnd)
            data = image.get_attribute("src")
            saveImage(data, term, index)

def saveImage(data, term, index):
    # Define save path and image name
    savePath = IMAGE_PATH + term + "/"
    imageName = str(index) + ".jpg"

    # Check the path exist or not
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # Check if the image is base64 or not
    if checkBase64(data):
        saveBase64Image(data, savePath+imageName,)
    else:
        saveLinkImage(data, savePath+imageName)


def checkBase64(data):
    if data[:4] == "data":
        return True
    else:
        return False

def saveBase64Image(data, path):
    print("Converting base64 to image...")
    with open(path, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(data[22:]))
    print("Image saved to " + path + " successfully\n")

def saveLinkImage(link, path):
    print("Downloading image...")
    response = requests.get(link)
    with open(path, "wb") as fh:
        fh.write(response.content)
    print("Image saved to " + path + " successfully\n")

if __name__ == "__main__":
    NUM_OF_IMAGES = 3
    SEARCH_TERM = ["cat","dog"]
    IMAGE_PATH = "C:/Users/Administrator/Desktop/Capstone/Data/AttractionDataset/"
    BROWSER_PATH = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

    browser = init()
    searchImage(browser, SEARCH_TERM)