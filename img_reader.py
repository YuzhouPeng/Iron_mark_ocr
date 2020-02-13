#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np

path = 'D:\\xianggang\\xianggang\\'

testimgpath = 'D:\\xianggang\\xianggang\\190707\\190707\\1.jpg'
srcpath = 'D:\\xianggang\\xianggang\\190707\\11'
resultpath = 'D:\\pics'
def morphologyEx(src):
    """
    #顶帽变换,增强图像中低对比度的目标
    :param img:
    :return img:
    """
    #读取图片

    #设置卷积核
    kernel = np.ones((5,5), np.uint8)
    time0 = time.time()

    #图像顶帽运算
    result = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    time1 = time.time()
    total = (time1 - time0)
    print ("morphologyEx need time: {} s".format(total))
    # #显示图像
    # cv2.imshow("src", src)
    # cv2.imshow("result", result)

    # #等待显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result

def otsu(src):
    """
    OTSU大津法实现最大类间方差
    :param img:
    :return img:
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	
    time0 = time.time()
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    time1 = time.time()
    total = (time1 - time0)
    print ("otsu need time: {} s".format(total))
 
    cv2.imshow("src", src)
    cv2.imshow("gray", gray)
    cv2.imshow("dst", dst)
    return dst

def iter(src):
    """
    #迭代法选择阈值
    :param img:
    :return img:
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print(gray)
    zmax = max(gray)
    zmin = min(gray)
    tk = (zmax+zmin)/2
    b = 1
    m = len(gray)
    n = len(gray[0])
    while b:
        ifg = 0
        ibg = 0
        fnum = 0
        bnum = 0
        for i in range(1,m):
            for j in range(1,n):
                tmp = grey[i][j]
                if tmp>=tk:
                    ifg=ifg+1
                    fnum = fnum+tmp
                else:
                    ibg=ibg+1
                    bnum = bnum+tmp
        zo = fnum/ifg
        zb = fnum/ibg
        if(tk==int((zo+zb)/2)):
            b = 0
        else:
            tk = int((zo+zb)/2)
    ret,thresh1 = cv2.threshold(img,tk,255,cv2.THRESH_BINARY)
    return thresh1


def segment(img):
    """
    最大熵分割
    :param img:
    :return:
    """
    def calculate_current_entropy(hist, threshold):
        data_hist = hist.copy()
        background_sum = 0.
        target_sum = 0.
        for i in range(256):
            if i < threshold:  # 累积背景
                background_sum += data_hist[i]
            else:  # 累积目标
                target_sum += data_hist[i]
        background_ent = 0.
        target_ent = 0.
        for i in range(256):
            if i < threshold:  # 计算背景熵
                if data_hist[i] == 0:
                    continue
                ratio1 = data_hist[i] / background_sum
                background_ent -= ratio1 * np.log2(ratio1)
            else:
                if data_hist[i] == 0:
                    continue
                ratio2 = data_hist[i] / target_sum
                target_ent -= ratio2 * np.log2(ratio2)
        return target_ent + background_ent

    def max_entropy_segmentation(img):
        channels = [0]
        hist_size = [256]
        prange = [0, 256]
        hist = cv2.calcHist(img, channels, None, hist_size, prange)
        hist = np.reshape(hist, [-1])
        max_ent = 0.
        max_index = 0
        for i in range(256):
            cur_ent = calculate_current_entropy(hist, i)
            if cur_ent > max_ent:
                max_ent = cur_ent
                max_index = i
        ret, th = cv2.threshold(img, max_index, 255, cv2.THRESH_BINARY)
        return th
    img = max_entropy_segmentation(img)
    return img






if __name__ == "__main__":
    for parent,_,files in os.walk(srcpath):
        for file in files:
            src = cv2.imread(os.path.join(parent,file), cv2.IMREAD_UNCHANGED)
            result = morphologyEx(src)
            
            cv2.imwrite(resultpath+"/"+file,result)