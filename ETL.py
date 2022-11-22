import os 
import glob



def process():

    filelist = glob.glob(os.path.join("data/train/images", "*.jpg"))
    filelistID = [fil.split("/")[-1][:-4] for fil in filelist]
    for id in filelistID:
        path = glob.glob(os.path.join("data/train/labels", f"*{id}*"))
        
        with open(path[0],"r") as file:    
            data = file.readlines()
            if len(data)==0:
                print(id)
        print(path[-1])

    pass
process()