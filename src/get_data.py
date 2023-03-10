import os
from Scalagram import Scalagram
def get_data(dir:str):
    dataList = []
    for wav in os.listdir(dir):
        if wav[-4:] == ".wav":
            sg = Scalagram(dir+"/" + wav)
            image = sg.get_data()
            dataList += [image]
    return tuple(dataList)

if __name__ == '__main__':
    get_data("./Data")