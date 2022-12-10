import tensorflow as tf
import os
import csv
# data/crazing_5.xml	crazing;crazing	10;15	3;110	197;197	108;198
HEIGHT =608
WIDTH = 608
print(os.listdir())
datapath = ['data/train','data/valid','data/test']

def read_image_id(path):
    path = os.path.join(path,'images')
    path = os.listdir(path)
    data=[]
    for d in path:
        data.append(d[:-4])
    return data


def id2image_path(datapath,image_id):
    path = os.path.join(datapath,'images',image_id)
    path = path+'.jpg'
    if os.path.exists(path):
        return path
    else:
        print("doesn't exist")
        return None

def id_located_label(datapath,data_list):
    gpath = os.path.join(datapath,'labels')
    data = {}
    for da in data_list:
        path = os.path.join(gpath,da)+'.txt'
        if os.path.exists(path):
            #temp = {da:path}
            data[da] = path
        else:
            print(da+'.jpg')
    
    return data

def content_reader(label_locate): # label_path in out bbox_list
    # 2 0.8758223684210527 0.6973684210526315 0.029605263157894735 0.06661184210526316

    with open(label_locate,'r') as file:
        records = file.readlines()
        labels = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        
        for record in records:
            fields = record.strip().split(' ')
            labels.append(eval(fields[0]))
            xmin.append(eval(fields[1]))
            ymin.append(eval(fields[2]))
            xmax.append(eval(fields[3]))
            ymax.append(eval(fields[4]))
        bbox_list = [labels,xmin,ymin,xmax,ymax]

    return bbox_list


def unbox(data):
    # data/crazing_5.xml,crazing;crazing,10;15,3;110,197;197,108;198
    bbox = []
    for rows in data:
        temp = ''
        for i,d in enumerate(rows):
            if i < len(rows)-1:
                temp = temp+str(d)+';'
            else:
                temp = temp+str(d)
        bbox.append(temp)
    return bbox


N = 0

for i,d in enumerate(datapath):
    rownum = 0
    dataID = read_image_id(d)
    data_label_locate = id_located_label(d,dataID)
    temppath = d
    filename = temppath.split('/')[1]+'.csv'
    
    with open(filename,'w') as file:
        for ID in dataID:
            rownum+=1
            image_id = ID
            if image_id in data_label_locate.keys():
                label_path = data_label_locate[image_id]
                bbox = content_reader(label_path)
                # image_path
                row = [id2image_path(temppath,image_id) ]
                for i in  unbox(bbox):
                    row.append(i)  
            
                writer = csv.writer(file)
                
                # 寫入一列資料
                writer.writerow(row)
            else:
                N+=1
                print(N)
                print(d)
                print(image_id)
    print(d)
    print(rownum)

    
    
    
    
    
    
    
    
    
    
    
    
    






