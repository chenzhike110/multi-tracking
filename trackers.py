from kalmanfilter import KalmanBoxTracker 
import numpy as np 
import tool

class tracker(object):
    def __init__(self,matrix):
        '''
        arg:
            playermatrix    [[1,2,3,4,0]  [x1,y1,x2,y2,lable]
                            [1,2,3,4,0]]   
            tplayers [palyer[0],player[1]....] 正在追踪的人
        method:
            update    更新
            draw      画图
        '''
        self.tplayers = []
        matrix = [tool.xy_to_sr(np.array(i)) for i in matrix]
        for index, position in enumerate(matrix):
            person = player(position,index)
            self.tplayers.append(person)
        self.count = len(self.tplayers)
    
    def update(self,matrix):
        matrix = [np.array(i) for i in matrix]
        ren = np.zeros((len(self.tplayers),5), dtype=np.float32)
        for index, people in enumerate(self.tplayers):
            ren[index] = tool.sr_to_xy(people.pred_next())
        connect,A1_nonconnect,A2_nonconnect = tool.km(ren,matrix)
        for i in connect:
            self.tplayers[int(i[0])].update(tool.xy_to_sr(matrix[int(i[1])]))
        for i in A2_nonconnect:
            person = player(tool.xy_to_sr(matrix[i]),self.count+1)
            self.tplayers.append(person)
            self.count += 1
    
    def get_information(self):
        position = []
        for index, person in enumerate(self.tplayers):
            position.append([tool.sr_to_xy(person.get_data()),person.number])
        return position


class player(object):
    def __init__(self,args_of_player,number):
        '''
        arg：
            box_of_player   [x,y,s,r,label] 当前检测值
                            [x1,y1,x2,y2,label] 也可以
            is_tracking     bool
            kalma           filter
        method：
            pred_next(self) 预测下一个状态        #[x,y,s,r,0]
            updata(self)    更新box_of_player    #[x,y,s,r,0]
            get_data(self)  返回box信息            #[x,y,s,r,0]

        注：关于是否正在追踪的判断，每有一个新的bbox输入进来，is_tracking=True,
            每用当前的box_of_player做一次预测，is_tracking=False 
        '''
        self.number=number
        self.box_of_player=args_of_player
        self.kalman=KalmanBoxTracker(self.box_of_player[0:4])
        self.is_tracking=True

    def pred_next(self):
        '''
        根据当前[x,y,s,r]预测人的下一个[x,y,s,r]
        输入
            box_of_player(当前检测)
        输出
            [[x,y,s,r,0]] 卡尔曼滤波的预测值 shape(1,5)
        '''
        pred_box_all=self.kalman.predict()
        pred_box=np.array(pred_box_all[0:5]).reshape(1,5)
        pred_box[0][4]=0
        self.is_tracking=False
        return pred_box.flatten()

    def update(self,bbox):
        '''
        功能 
            更新box_of_player
        输入
            参数bbox为新检测的信息[x,y,s,r,0]
        '''
        self.is_tracking=True
        #卡尔曼滤波器更新
        self.kalman.update(bbox[0:4])
        #更新人的box
        self.box_of_player=bbox

    def get_data(self):
        '''
        功能
            综合预测值和新得到的检测值，返回人的滤波后box位置与检测值的最优估计
            但是player这里只关注当前的box值
        输入
            在pred_box,updata,完成后，无输入
        输出
            [[x,y,s,r,0]]
        '''
        p=np.array(self.kalman.get_state()[0:5]).reshape(1,5)
        p[0][4]=0
        return p.flatten()

if __name__ == '__main__':
    sort = tracker([[1,2,3,4,0]])
    print(sort.get_information())
    sort.update([[1.1,2,3,4,0],[1,2,3,4,0]])
    print(sort.get_information())