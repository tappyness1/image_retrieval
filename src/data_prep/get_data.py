import requests
import tarfile
import urllib.request

def get_tgz_file(save_path = "data/102flowers.tgz"):
    print("downloading tgz file...")
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    urllib.request.urlretrieve(url, save_path)
    print("downloading tgz file done")

def get_labels(save_path = "data/jpg/"):
    print("downloading mat file...")
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    urllib.request.urlretrieve(url, save_path)
    print("downloading done")

def unzip_images_and_delete(zipfile_path = "data/102flowers.tgz", save_path = "data/"):
    # open file
    file = tarfile.open(zipfile_path)
    
    # extracting file
    file.extractall(save_path)
    
    file.close()
    print("files extracted")

def get_data():
    get_tgz_file()
    unzip_images_and_delete()
    get_labels()
    print ("completed getting data")

if __name__=="__main__":
    # get_tgz_file()
    get_data()
    # get_labels()