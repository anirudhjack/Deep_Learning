import xlrd
import csv
import numpy as np

X_train=list()
Y_train=list()
i=0
with open('csv_result-caesarian.csv') as csvfile: 
    mpg_data = csv.reader(csvfile)
    for line in mpg_data:
      if i<2:
        i+=1
      else:
        line[1]=float(line[1])
        line[1]=(line[1]-18)/(float(22))
        X_train.append(list(map(float, line[1:6])))
        Y_train.append(int(line[6]))
 
weight1=np.random.rand(10,5)
weight2=np.random.rand(1,10)

def sigmoid(num):
  return 1/(1+np.exp(-num))
  
def Forward_propagate(Input):
  Input=np.matrix(Input)
  preoutput=sigmoid(np.matmul(weight1,np.transpose(Input)))
  output=sigmoid(np.matmul(weight2,(preoutput)))
  return (preoutput,output)


def backpropagation(Input,target):
  (preoutput,output)=Forward_propagate(Input)
  delta=list()
  delta.append(-(target-output[0,0])*output[0,0]*(1-output[0,0]))
  delta=np.matrix(delta)
  preoutput=np.matrix(preoutput)
  error_weight2=np.matmul(delta,preoutput.T)
  
  delta_weight2=np.transpose(weight2)
  delta_weight2=np.matrix(weight2)
  delta_preoutput=list()
  for i in range(np.shape(preoutput[0])[1]):
    delta_preoutput.append(preoutput[0,i]*(1-preoutput[0,i]))
  
  delta_weight2[0]=np.multiply(delta_weight2[0],delta_preoutput)
  
  delta_weight2=np.transpose(delta_weight2)
  Input=np.matrix(Input)
  Input=np.transpose(Input)
  
  s=np.matmul(delta_weight2,delta)
  error_weight1=np.matmul(s,np.transpose(Input))
  return (error_weight2,error_weight1)
  
learning_rate=0.01
alpha=0.0000001
v00=0
v01=0
for i in range(2000):
  for j in range(len(X_train)): 
    (t,r)=backpropagation(X_train[j],Y_train[j])
    p=alpha*v00-learning_rate*r
    q=alpha*v01-learning_rate*t
    weight1=weight1+p
    weight2=weight2+q
    v00=p
    v01=q
    weight1=weight1-alpha*p
    weight2=weight2-alpha*q
  error=0.0
  for j in range(len(X_train)):
    (o,r)=Forward_propagate(X_train[j])
    error+=(Y_train[j]-r[0,0])*(Y_train[j]-r[0,0])/2
  print(error)
