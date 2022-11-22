import os 
import glob

# path = ["data/valid/images"]

# remove the image without label
def process(dir):
    filelist = glob.glob(os.path.join(dir, "*.jpg"))
    filelistID = [fil.split("/")[-1][:-4] for fil in filelist]
    label_dir=''
    for i in dir.split("/")[:-1]:
        label_dir = os.path.join(label_dir,i)
    label_dir = os.path.join(label_dir,"labels")

    print(label_dir)
    num = 0
    for id in filelistID:
        path = glob.glob(os.path.join(label_dir, f"*{id}*"))
        try:
            with open(path[0],"r") as file:    
                data = file.readlines()
                if len(data)==0:
                    num+=1
                    os.remove(os.path.join(dir,id)+".jpg")
                    os.remove(path[0])

        except Exception as e:
            print(e)
    print(len(filelistID))
    print(num)
    pass
process(path[0])