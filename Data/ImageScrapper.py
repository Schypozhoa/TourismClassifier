from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import base64
import os
import time

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

def searchImage(browser, search_term, total_images):
    # Define links for image search
    links = "https://www.google.co.id/search?tbm=isch&q="

    # Iterate every search term
    for term in search_term:
        browser.get(links + term)
        print(f"Searching for {term} images, expect result of {total_images}...")

        # Loop until the total images is reached
        result = 0
        while result < total_images:
            # Scroll down to the bottom of the page
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            # Find all images
            images = browser.find_elements("css selector",".Q4LuWd")
            for image in images:
                # Check if the image have src or not
                if image.get_attribute("src"):
                    saveImage(image.get_attribute("src"), term, result+1)
                    result += 1

                # Break the loop if the total images is reached
                if result >= total_images:
                    break
        print(f"Done searching for {term} images, and successfuly saved {result} images\n")
    print("Done searching for all images")
    print(f"Images saved at {IMAGE_PATH}")

def saveImage(data, term, index):
    # Define save path and image name
    savePath = IMAGE_PATH + term + "/"
    imageName = str(index) + ".jpg"

    # Check the path exist or not
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # Check if the image is base64 or not
    if checkBase64(data):
        saveBase64Image(data, savePath+imageName)
    else:
        saveLinkImage(data, savePath+imageName)

def checkBase64(data):
    if data[:4] == "data":
        return True
    else:
        return False

def saveBase64Image(data, path):
    with open(path, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(data[22:]))

def saveLinkImage(link, path):
    response = requests.get(link)
    with open(path, "wb") as fh:
        fh.write(response.content)


if __name__ == "__main__":
    NUM_OF_IMAGES = 100
    SEARCH_TERM = ["wisata gunung","wisata camping", "wisata pantai"]
    IMAGE_PATH = "C:/Users/Administrator/Desktop/Capstone/Data/AttractionDataset/"
    BROWSER_PATH = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

    browser = init()
    searchImage(browser, SEARCH_TERM, NUM_OF_IMAGES)