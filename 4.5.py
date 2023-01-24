import pandas as pd
import matplotlib.pyplot as plt

def geometric_center(filepath):
    pd1 = pd.read_csv(filepath)
    ls = []
    for x, y in pd1.iterrows():
        ls.append([y.lefx + y.widt / 2, y.topy + y.heig / 2])
    pd1['cent'] = ls
    pd1.to_csv('AAresult.csv', index=False)

def generateID(path):
    data=pd.read_csv(path)
    data[['pc_x','pc_y']]=data.apply(lambda x:eval(x[-1]),axis=1,result_type='expand')
    plt.scatter(data.pc_x,data.pc_y)
    plt.show()
    from sklearn.cluster import DBSCAN
    clustering=DBSCAN(eps=10,min_samples=1).fit(data.iloc[:,-2:].values)
    data['p_id']=clustering.labels_
    plt.scatter(data.pc_x,data.pc_y,c=data.p_id)
    data.to_csv('AAresult.csv', index=False)
    return data
generateID('./yoloresult_g.csv')

#geometric_center("./yoloresult_g.csv")
