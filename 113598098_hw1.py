import cv2
import numpy as np
#print(cv2.__version__)
test_img_path="test_img/CKS.jpg"
save_Q1_path="result_img/CKS_Q1.jpg"
save_Q2_path="result_img/CKS_Q2.jpg"
save_Q3a_path="result_img/CKS_Q3a.jpg"
save_Q3b_path="result_img/CKS_Q3b.jpg"
save_Q4a_path="result_img/CKS_Q4a.jpg"
save_Q4b_path="result_img/CKS_Q4b.jpg"

def save_image_as_jpg(path,img):
        cv2.imwrite(path, img)

def load_img(path):
    try:
        img = cv2.imread(path)
    except:
        print("讀取失敗")
    return img

def convert_rgb_to_grayscal(img):
        gray_img=[]
        for row in range(img.shape[0]):
            t=[]
            for col in range(img.shape[1]):
                t.append(0)
            gray_img.append(t)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                gray_img[row][col]=img[row][col][0]*0.299+img[row][col][1]*0.587+img[row][col][2]*0.114
        return np.array(gray_img)#轉為array

def zero_padding(img):
        zero_padding_img=[]
        """第一列加入全0"""
        temp=[]
        for i in range(img.shape[1]+2):
            temp.append(0)
        zero_padding_img.append(temp)
        """中間的頭尾個加入一個0"""
        for row in range(img.shape[0]):
            temp=[]
            temp.append(0)
            for col in range(img.shape[1]):
                  temp.append(img[row][col])
            temp.append(0)
            zero_padding_img.append(temp)
        """最後一列加入全0"""
        temp=[]
        for i in range(img.shape[1]+2):
            temp.append(0)
        zero_padding_img.append(temp)
        return np.array(zero_padding_img)

def convolution_with_Laplacian_edge_detection_kernel(img):
    def caculate(list_of_3x3):
        Laplacian_kernel=[[0,-1,0],
                          [-1,4,-1],
                          [0,-1,0]]
        value=0
        for row in range(3):
            for col in range(3):
                 value=value+list_of_3x3[row][col]*Laplacian_kernel[row][col]
        return value
    
    return_img=[]
    for row in range(1,img.shape[0]-1):
        temp=[]
        for col in range(1,img.shape[1]-1):
            list_of_3x3=[]
            list_of_3x3.append(img[row-1][col-1:col+2]) 
            list_of_3x3.append(img[row][col-1:col+2])  
            list_of_3x3.append(img[row+1][col-1:col+2])   
            value=caculate(list_of_3x3)
            temp.append(value)
        return_img.append(temp)
    return np.array(return_img)
"""Q1"""
test_img=load_img(test_img_path)
#print(test_img.shape)
Q1_result=convert_rgb_to_grayscal(test_img)
save_image_as_jpg(save_Q1_path,Q1_result)
"""Q2"""
img_after_zero_padding=zero_padding(Q1_result)
#print(Q1_result.shape)
#print(img_after_zero_padding.shape)
Q2_result=convolution_with_Laplacian_edge_detection_kernel(img_after_zero_padding)
#print(Q2_result.shape)
save_image_as_jpg(save_Q2_path,Q2_result)
"""Q3"""
def Avg_pooling(img):
    return_img=[]
    for row in range(0,img.shape[0],2):
        temp=[]
        if(row+2>=img.shape[0]):
            break
        for col in range(0,img.shape[1],2):
            if(col+2>=img.shape[1]):
                break
            list_of_3x3=[]
            list_of_3x3.append(img[row][col:col+3])
            list_of_3x3.append(img[row+1][col:col+3])
            list_of_3x3.append(img[row+2][col:col+3])
            total=0
            for i in range(3):
                for j in range(3):
                    total=total+list_of_3x3[i][j]
            temp.append(total/9)
        return_img.append(temp)
    return np.array(return_img) 

def Max_pooling(img):
    return_img=[]
    for row in range(0,img.shape[0],2):
        temp=[]
        if(row+2>=img.shape[0]):
            break
        for col in range(0,img.shape[1],2):
            if(col+2>=img.shape[1]):
                break
            list_of_3x3=[]
            list_of_3x3.append(img[row][col:col+3])
            list_of_3x3.append(img[row+1][col:col+3])
            list_of_3x3.append(img[row+2][col:col+3])
            max=0
            for i in range(3):
                for j in range(3):
                    if(list_of_3x3[i][j]>max):
                         max=list_of_3x3[i][j]
            temp.append(max)
        return_img.append(temp)
    return np.array(return_img)

Q3a_result=Avg_pooling(Q2_result)
#print(Q3a_result.shape)
save_image_as_jpg(save_Q3a_path,Q3a_result)
Q3b_result=Max_pooling(Q2_result)
#print(Q3b_result.shape)
save_image_as_jpg(save_Q3b_path,Q3b_result)
"""Q4"""
def binarization_operation(img,threshold):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if(img[row][col]>threshold):
                img[row][col]=255
            else:
                img[row][col]=0
    return img
Q4a_result=binarization_operation(Q3a_result,128)
save_image_as_jpg(save_Q4a_path,Q4a_result)
Q4b_result=binarization_operation(Q3b_result,128)
save_image_as_jpg(save_Q4b_path,Q4b_result)