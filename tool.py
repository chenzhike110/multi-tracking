import numpy as np 

def build_match_max(nal,nbl):
    '''
    构建权值矩阵
    '''
    iou_matrix = np.zeros((len(A1), len(A2)), dtype=np.float32)
    s=set()
    for d, det in enumerate(A1):
        for t, trk in enumerate(A2):
            a=iou(det,trk)
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

def iou(na,nb):
    '''
    输入 ndarray 类型 na、nb，[x1,y1,x2,y2,label].shape(5,1)
        x1,y1 左上角
        x2,y2 右下角
        label 编号
    输出
        float类型 (0,1)之间
    算法
        s=(x2-x1)*(y2-y1)
        s0=s1+s2-s1*s2
        iou=s1*s2/s0
    '''
    s1=(na[2]-na[0])*(na[3]-na[1])
    s2=(nb[2]-nb[0])*(nb[3]-nb[1])
    #交集坐标
    left_x=max(na[0],nb[0])
    left_y=max(na[1],nb[1])
    right_x=min(na[2],nb[2])
    right_y=min(na[3],nb[3])
    if right_x>=left_x and right_y>=left_y:
        s3=(right_x-left_x)*(right_y-left_y)
    else:
        return 0
    return (s3)/(s1+s2-s3)

def xy_to_sr(na):
    '''
    输入
        ndarray shape(5,1) [x1,y1,x2,y2,label]
    输出
        ndarry shape(5,1) [x,y,s,r,label]
        s=xy
        r=x/y
    '''
    #输入本来就是浮点型时不需要下面这句
    ns=na.astype(np.double)
    ns[0:2]=(ns[0:2]+ns[2:4])/2.0
    ns[2]=(na[2]-na[0])*(na[3]-na[1])
    ns[3]=1.0*(na[2]-na[0])/(na[3]-na[1])
    return ns

def sr_to_xy(na):
    '''
    输入
        ndarry shape(5,1) [x,y,s,r,label]
    输出
        ndarray shape(5,1) [x1,y1,x2,y2,label]
    '''
    #输入本来就是浮点型时不需要下面这句
    ns=na.astype(np.double)

    y=np.sqrt(na[2]/na[3])/2.0
    x=y*na[3]
    ns[2:4]=ns[0:2]
    ns[0]-=x
    ns[1]-=y
    ns[2]+=x
    ns[3]+=y
    return ns

if __name__ == "__main__":
    a=np.array([[1,4,3,9,0],[1,2,3,4,0]])
    b=np.array([[1,3,3,6,0],[1,2,3,4,0]])
    k=KM_match(a,b)
    print(KM_match(a,b))
    print(addrow(k,0,1))
    # print(iou(a,b))
    # a=xy_to_sr(a)
    # b=sr_to_xy(a)
    # print(a)
    # print(b)