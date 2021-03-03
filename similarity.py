# 图像相似度

import cv2
import os
from PIL import Image
from numpy import average, dot, linalg
import numpy as np
import math
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt

# 计算颜色矩
def color_moments(filename, mode):
    img = cv2.imread(filename)
    if img is None:
        return
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    # 变换颜色空间
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if mode==1:
        abc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode==2:
        abc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mode==3:
        abc = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 分离颜色空间，分别计算三个颜色空间的一阶二阶三阶颜色中心矩
    h, s, v = cv2.split(abc)
    # 存储特征向量
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    # 返回颜色矩向量
    return color_feature


# 通过计算颜色矩计算图像相似度
def similarity_123(file1, file2, mode):
    # mode==1: RGB颜色空间
    # mode==2: HSV颜色空间
    # mode==3: LAB颜色空间
    feature1 = color_moments(file1, mode)
    feature2 = color_moments(file2, mode)
    if feature1 == feature2:
        return 1.0
    else:
        return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))


# 计算图片的余弦距离
def similarity_4(filepath0, filepath):
    image1 = Image.open(filepath0)
    image2 = Image.open(filepath)
    # 统一图像大小
    image1 = image1.resize((64, 64), Image.ANTIALIAS)
    image2 = image2.resize((64, 64), Image.ANTIALIAS)

    #计算图像向量
    vector1=[]
    vector2=[]
    for pixel_tuple in image1.getdata():
        vector1.append(average(pixel_tuple))
    for pixel_tuple in image2.getdata():
        vector2.append(average(pixel_tuple))
    similar = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return similar

#ssim算法
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


#通过SSIM计算图像相似度
def similarity_5(filepath0, filepath):
    img1 = cv2.imread(filepath0, 0)
    img2 = cv2.imread(filepath, 0)
    if not img1.shape == img2.shape:
        # img1.resize(img2.shape)
        rows, cols = img1.shape[:2]  # 获取sky的高度、宽度
        # print(sky.shape[:2]) #(800, 1200)
        # print(bear.shape[:2]) #(224, 224)
        img2 = cv2.resize(img2, (cols, rows), interpolation=cv2.INTER_CUBIC)  # 放大图像
        # raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



# 计算直方图(RGB颜色空间)
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 通过直方图计算图像相似度
def similarity_6(filepath0, filepath):
    image1 = Image.open(filepath0)
    image2 = Image.open(filepath)
    image1 = image1.resize((64,64)).convert('RGB')
    image2 = image2.resize((64, 64)).convert('RGB')
    calc_sim = hist_similar(image1.histogram(), image2.histogram())
    # print("图片间的相似度为", calc_sim)
    return calc_sim


# 均值哈希算法
def ahash(image):
    # 将图片缩放为8*8的
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # s为像素和初始灰度值，hash_str为哈希值初始值
    s = 0
    ahash_str = ''
    # 遍历像素累加和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 计算像素平均值
    avg = s / 64
    # 灰度大于平均值为1相反为0，得到图片的平均哈希值，此时得到的hash值为64位的01字符串
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                ahash_str = ahash_str + '1'
            else:
                ahash_str = ahash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(ahash_str[i: i + 4], 2))
    return result


# 差异值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 灰度大于上一个像素点为1相反为0，得到图片的差异哈希值，此时得到的hash值为64位的01字符串
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result


# 计算hash值
def pHash(imgfile):
    #加载并调整图片为32x32灰度图片
    img=cv2.imread(imgfile, 0)
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)

    #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32,32)

    #把二维list变成一维list
    img_list=vis1.flatten()


    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,8*8,4)])


# 计算两个哈希值之间的差异
def Hamming_distance(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


# 通过ahash值计算图像相似度
def similarity_7(filepath0, filepath):
    img1 = cv2.imread(filepath0)
    img2 = cv2.imread(filepath)
    hash1 = ahash(img1)
    hash2 = ahash(img2)
    camphash = Hamming_distance(hash1, hash2)
    return 1.0*(64-camphash)/64


# 通过dhash值计算图像相似度
def similarity_8(filepath0, filepath):
    img1 = cv2.imread(filepath0)
    img2 = cv2.imread(filepath)
    hash1 = dhash(img1)
    hash2 = dhash(img2)
    camphash = Hamming_distance(hash1, hash2)
    return 1.0*(64-camphash)/64


# 通过phash值计算图像相似度
def similarity_9(filepath0, filepath):
    hash1 = pHash(filepath0)
    hash2 = pHash(filepath)
    camphash = Hamming_distance(hash1, hash2)
    return 1.0*(16-camphash)/16


# 在选定路径下找到和选定图像相似度最高的三张图像，mode代表使用的计算图像相似度的算法
def find_most_similarity(filepath0, scanpath, mode):
    if mode<10:
        most_similar_path = ""
        most_similar_value = 0
        second_similar_path = ""
        second_similar_value = 0
        third_similar_path = ""
        third_similar_value = 0
    # else:
    #     most_similar_path = ""
    #     most_similar_value = 100
    #     second_similar_path = ""
    #     second_similar_value = 100
    #     third_similar_path = ""
    #     third_similar_value = 100

    for root, dirs, files in os.walk(scanpath):
        split = "."
        # 遍历文件
        for filename in files:
            filepath = os.path.join(root, filename)

            if (filename.endswith(".png") or filename.endswith(".jpg")):
                # print(filepath)
                if mode==1 or mode==2 or mode==3:
                    tempsimilarity=similarity_123(filepath0, filepath, mode)
                # elif mode==2:
                #     tempsimilarity = similarity_2(filepath0, filepath)
                # elif mode==3:
                #     tempsimilarity=similarity_3(filepath0, filepath)
                elif mode==4:
                    tempsimilarity = similarity_4(filepath0, filepath)
                elif mode==5:
                    tempsimilarity = similarity_5(filepath0, filepath)
                elif mode==6:
                    tempsimilarity = similarity_6(filepath0, filepath)
                elif mode==7:
                    tempsimilarity = similarity_7(filepath0, filepath)
                elif mode==8:
                    tempsimilarity = similarity_8(filepath0, filepath)
                elif mode==9:
                    tempsimilarity = similarity_9(filepath0, filepath)

                if mode<10:
                    if tempsimilarity > most_similar_value: #and tempsimilarity!=1:
                        third_similar_path = second_similar_path
                        third_similar_value = second_similar_value
                        second_similar_path = most_similar_path
                        second_similar_value = most_similar_value
                        most_similar_path = filepath
                        most_similar_value = tempsimilarity
                    elif tempsimilarity > second_similar_value: #and tempsimilarity!=1:
                        third_similar_path = second_similar_path
                        third_similar_value = second_similar_value
                        second_similar_path = filepath
                        second_similar_value = tempsimilarity
                    elif tempsimilarity > third_similar_value: #and tempsimilarity!=1:
                        third_similar_path = filepath
                        third_similar_value = tempsimilarity
                # else:
                #     if tempsimilarity < most_similar_value: #and tempsimilarity!=1:
                #         third_similar_path = second_similar_path
                #         third_similar_value = second_similar_value
                #         second_similar_path = most_similar_path
                #         second_similar_value = most_similar_value
                #         most_similar_path = filepath
                #         most_similar_value = tempsimilarity
                #     elif tempsimilarity < second_similar_value: #and tempsimilarity!=1:
                #         third_similar_path = second_similar_path
                #         third_similar_value = second_similar_value
                #         second_similar_path = filepath
                #         second_similar_value = tempsimilarity
                #     elif tempsimilarity < third_similar_value: #and tempsimilarity!=1:
                #         third_similar_path = filepath
                #         third_similar_value = tempsimilarity

    print(most_similar_path)
    print(most_similar_value)
    print(second_similar_path)
    print(second_similar_value)
    print(third_similar_path)
    print(third_similar_value)

    similar_paths=[]
    similar_paths.append(most_similar_path)
    similar_paths.append(second_similar_path)
    similar_paths.append(third_similar_path)
    similaritys=[]
    similaritys.append(most_similar_value)
    similaritys.append(second_similar_value)
    similaritys.append(third_similar_value)

    return similar_paths, similaritys


# 使用matplotlib展示源图像和相似度最高的三张图像
def matplotlib_multi_pic(image0, images, similaritys):
    img = cv2.imread(image0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    title = "origin_image"
    # 行，列，索引
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    for i in range(3):
        img = cv2.imread(images[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        title="similarity:"+str(similaritys[i])
        #行，列，索引
        plt.subplot(2,2,i+2)
        plt.imshow(img)
        plt.title(title,fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    print("Welcome to image similarity comparing program...")
    print("please input the address of the image to compare with...")
    filepath0 = input()
    print("please input the address of the folder to scan...")
    scanpath = input()
    print("please choose the method to compute image similarity...")
    print("1.Color Moment(颜色矩)(RGB 颜色空间)")
    print("2.Color Moment(颜色矩)(HSV 颜色空间)")
    print("3.Color Moment(颜色矩)(LAB 颜色空间)")
    print("4.Cosine Similarity(余弦相似度)")
    print("5.Structural Similarity(结构相似度)(SSIM)")
    print("6.Histogram Similarity(通过直方图计算)")
    print("7.Average Hash 均值哈希相似度")
    print("8.Difference Hash 差异哈希相似度")
    print("9.Perceptual Hash 感知哈希相似度")
    mode=int(input())



    # filepath0 = ".\static\\pic\\001.png"
    # scanpath = ".\static\\pics"
    similar_paths, similaritys =find_most_similarity(filepath0, scanpath, mode)

    matplotlib_multi_pic(filepath0, similar_paths, similaritys)
