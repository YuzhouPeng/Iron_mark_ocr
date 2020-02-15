import cv2 ,time
import numpy as np  
from matplotlib import pyplot as plt  
from PIL import Image
path = 'D:\\xianggang\\xianggang\\'

testimgpath = 'D:\\pics\\otsu\\pics_fault1.jpg'
srcpath = 'D:\\xianggang\\xianggang\\mianpeng200103\\200103\\'
resultpath = 'D:\\pics'
def vertical_projection():
    """
    垂直投影
    """
    img=cv2.imread(testimgpath)  #读取图片，装换为可运算的数组
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #将BGR图转为灰度图
    ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)  #将图片进行二值化（130,255）之间的点均变为255（背景）
    # print(thresh1[0,0])#250  输出[0,0]这个点的像素值  				#返回值ret为阈值
    # print(ret)#130
    (h,w)=thresh1.shape #返回高和宽
    print(h)
    print(w)
    # print(h,w)#s输出高和宽
    a = [0 for z in range(0, w)] 
    # print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数  
    
    #记录每一列的波峰
    for j in range(0,w): #遍历一列 
        for i in range(0,h):  #遍历一行
            if  thresh1[i,j]==0:  #如果改点为黑点
                a[j]+=1  		#该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色 
        # print (j)           
    
    #            
    for j  in range(0,w):  #遍历每一列
        for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i,j]=0   #涂黑
    print(len(a))
    #此时的thresh1便是一张图像向垂直方向上投影的直方图
    #如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息
    index = 0
    while a[index]==2048:
        index+=1
    endindex = 2047
    while a[endindex]==2048:
        endindex-=1
    
    # img2 =Image.open('0002.jpg')
    # img2.convert('L')
    # img_1 = np.array(img2)
    plt.imshow(thresh1,cmap=plt.gray())
    plt.show()
    cv2.imshow('img',thresh1)  
    cv2.waitKey(0)  


def horizontal_projection():
    """
    水平投影
    """
    img=cv2.imread(testimgpath) 
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)
    
    (h,w)=thresh1.shape #返回高和宽
    
    a = [0 for z in range(0, h)] 
    # print(a) 
    
    for j in range(0,h):  
        for i in range(0,w):  
            if  thresh1[j,i]==0: 
                a[j]+=1 
                thresh1[j,i]=255
            
    for j  in range(0,h):  
        for i in range(0,a[j]):   
            thresh1[j,i]=0    
    # print(a)
    index = 0
    while a[index]==2448:
        index+=1
    endindex = 2047
    while a[endindex]==2448:
        endindex-=1
    print(index)
    print(endindex)
    plt.imshow(thresh1,cmap=plt.gray())
    plt.show()



def cut_pics():
    img=cv2.imread(testimgpath)  #读取图片，装换为可运算的数组
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #将BGR图转为灰度图
    ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)  #将图片进行二值化（130,255）之间的点均变为255（背景）
    # print(thresh1[0,0])#250  输出[0,0]这个点的像素值  				#返回值ret为阈值
    # print(ret)#130
    (h,w)=thresh1.shape #返回高和宽
    # print(h)
    # print(w)
    # print(h,w)#s输出高和宽
    a = [0 for z in range(0, w)] 
    # print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数  
    
    #记录每一列的波
    for j in range(0,w): #遍历一列 
        for i in range(0,h):  #遍历一行
            if  thresh1[i,j]==0:  #如果改点为黑点
                a[j]+=1  		#该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色 
        # print (j)           

    print(len(a))
    # print(a)
    #此时的thresh1便是一张图像向垂直方向上投影的直方图
    #如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息
    hindex = 0
    while a[hindex]==2048:
        hindex+=1
    hendindex = 2447
    while a[hendindex]==2048:
        hendindex-=1
    print(hindex)
    print(hendindex)
    ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)  #将图片进行二值化（130,255）之间的点均变为255（背景）

    a = [0 for z in range(0, h)] 
    # print(a) 
    
    for j in range(0,h):  
        for i in range(0,w):  
            if  thresh1[j,i]==0: 
                a[j]+=1 
                thresh1[j,i]=255
             
    print(len(a))
    print(a)
    index = 0
    while a[index]==2448:
        index+=1
    endindex = 2047
    while a[endindex]==2448:
        endindex-=1
    print(index)
    print(endindex)
    result = img[index:endindex,hindex:hendindex]
    cv2.imshow("cropped", result)
    cv2.imwrite(resultpath+"\\result.jpg",result)


def cut_chars():
    img=cv2.imread(testimgpath)  #读取图片，装换为可运算的数组
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #将BGR图转为灰度图
    ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)  #将图片进行二值化（130,255）之间的点均变为255（背景）
    # print(thresh1[0,0])#250  输出[0,0]这个点的像素值  				#返回值ret为阈值
    # print(ret)#130
    (h,w)=thresh1.shape #返回高和宽
    print(h)
    print(w)
    a = [0 for z in range(0, w)] 
    a1 = [0 for z in range(0, h)]
    npa = np.asarray(thresh1, dtype=np.int)

    # print(h)
    # print(w)
    # print(h,w)#s输出高和宽
    # print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数  
    time1 = time.time()
    #记录每一列的波峰
    for j in range(0,w): #遍历一列 
        for i in range(0,h):  #遍历一行
            if  npa[i,j]==255:  #如果改点为黑点
                a[j]+=1  		#该列的计数器加一计数
                a1[i]+=1
                # thresh1[i,j]=255  #记录完后将其变为白色 
    time3 = time.time()
    total = (time3 - time1)
    print ("count need time: {} s".format(total))
        # print (j)           
    # print(len(a))
    # print(a)
    #此时的thresh1便是一张图像向垂直方向上投影的直方图
    #如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息
    hindex = 0
    while a[hindex]==0:
        hindex+=1
    hendindex = 2447
    while a[hendindex]==0:
        hendindex-=1
    # print(hindex)
    # print(hendindex)
    

    # ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)  #将图片进行二值化（130,255）之间的点均变为255（背景）

    # a = [0 for z in range(0, h)] 
    # # print(a) 
    
    # for j in range(0,h):  
    #     for i in range(0,w):  
    #         if  thresh1[j,i]==0: 
    #             a[j]+=1 
    #             thresh1[j,i]=255
    # print(len(a))
    # print(a)
    index = 0
    while a1[index]==0:
        index+=1
    endindex = 2047
    while a1[endindex]==0:
        endindex-=1
    time2 = time.time()
    total = (time2 - time1)
    print ("cut pics need time: {} s".format(total))
    print(index)
    print(endindex)
    result = img[index:endindex,hindex:hendindex]
    cv2.imshow("cropped", result)
    cv2.imwrite(resultpath+"\\result1.jpg",result)


if __name__ == "__main__":
    # vertical_projection()
    # horizontal_projection()
    cut_chars()