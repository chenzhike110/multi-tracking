import numpy as np
import sys
sys.path.append(r"C:\Users\JCX\Desktop\KM")
import tool

def build_match_max(nal,nbl):
    '''
    构建权值矩阵
    '''
    iou_matrix = np.zeros((len(A1), len(A2)), dtype=np.float32)
    s=set()
    for d, det in enumerate(A1):
        for t, trk in enumerate(A2):
            a=tool.iou(det,trk)
            s.add(a)
            iou_matrix[d, t]=a
    j=[]
    for i in s:
        j.append(i)
    j=sorted(j)

    for i in range(len(j)-1,0,-1):
        iou_matrix[iou_matrix==j[i]]=i
    return iou_matrix

def km(A1,A2):
    '''
        输入 ndarray 类型 A1，A2 [[x1,y1,x2,y2,label],[x1,y1,x2,y2,label]......]
            A1:前一个图像的框
            A2：后一个图像的框
        输出
            ndarray类型 
            connect,A1_nonconncet,A2_nonconnect=km(A1,A2)
            connect:成功配对的编号对
            A1_nonconncet:A1中未配对的
            A2_nonconnect:A2中未配对的
            
            若无则返回【】
    '''
    #构造行名为A1，列名为A2的iou权值矩阵
    iou_matrix=build_match_max(A1, A2)

    precision=1 #每次减去的值
    flag=0 #该行是否配对
    A1_weight=np.max(iou_matrix,1) #初始时的每行的最大权重，也就是左侧顶值
    A2_weight=np.linspace(0,0,len(A2)) #右侧顶值初始为0
    match = np.linspace(-1,-1,len(A2)) #eg.match[0]=1,即表示A1的1与A2的0配对，初始-1表示未配对

    for d, det in enumerate(A1_weight): #按行循环
        flag=0
        while(A1_weight[d]>0): #判断该行的顶值是否还有
            index_all=np.where(iou_matrix[d]==A1_weight[d]+min(A2_weight))[0] #找到该行中与“左侧顶值+右侧最小顶值”相同的A2的所有坐标 index_all
            for t , index in enumerate(index_all): #按index_all中的坐标循环
                if match[index] == -1:  #如果该A2未配对，则将此A1 A2配对
                    match[index] = d
                    flag=1
                    break
                else:
                    line=int(match[index]) #line为发生冲突的A1的编号
                    weight=A1_weight[line]+min(A2_weight)
                    index_all_conflict = np.where(iou_matrix[line] == weight)[0]
                    for j, index_conflict in enumerate(index_all_conflict): #看一下冲突行有没有另外的选择
                        if match[index_conflict] == -1:
                            match[index_conflict] = line
                            match[index] = d
                            flag = 1
                            break
            if(flag==1):
                break
            else:
                A1_weight[d]=A1_weight[d]-precision
                A1_weight[line] = A1_weight[line] - precision
                A2_weight[line]=A2_weight[line]+precision     #若无法解决冲突，则左顶值-precision，右顶值+precision


    A2_nonconnect=np.where(match==-1)[0]  #match中仍为0的即为未匹配的A2

    connect=np.vstack((match[np.where(match!=-1)[0]],np.where(match!=-1)[0]))
    connect=connect.T                                                        #match的非-1值与相应序号纵向堆叠，再转置，构成匹配对

    A1_nonconnect=[]
    for i,a in enumerate(np.linspace(0,len(A1)-1,len(A1))):
        if a not in match:
            A1_nonconnect.append(a)
    A1_nonconnect=np.array(A1_nonconnect)         #match的值中未出现的A1编号即为未配对的A1

    return(connect,A1_nonconnect,A2_nonconnect)




if __name__ == "__main__":
    A1=np.array([[1,2,3,4,5],[2,3,43,5,6],[2,4,100,19,5],[3,1,32,21,3]])
    A2=np.array([[3,4,5,6,7],[1,2,21,5,6],[3,3,4,2,4],[1,2,3,4,9]])
    a,b,c=km(A1 , A2)
    print(a)
    print(b)
    print(c)
