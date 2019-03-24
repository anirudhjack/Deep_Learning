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
  
def elementwisemultiplication1(sq,t,learning_rate):
  delta=0.000001
  denom=np.sqrt(delta+(sq))
  numer=-learning_rate*(1/denom)
  ans=np.multiply(numer,t)
  return ans

hyp_parameter=0.999
learning_rate=0.01
v00=0
v01=0
for i in range(2000):
  for j in range(len(X_train)): 
    (t,r)=backpropagation(X_train[j],Y_train[j])
    d=v00+(1-hyp_parameter)*np.multiply(t,t)
    c=v01+(1-hyp_parameter)*np.multiply(r,r)
    p=elementwisemultiplication1(d,t,learning_rate)
    q=elementwisemultiplication1(c,r,learning_rate)
    weight1=weight1+q
    weight2=weight2+p
    v00=hyp_parameter*d
    v01=hyp_parameter*c
  error=0.0
  for j in range(len(X_train)):
    (o,r)=Forward_propagate(X_train[j])
    error+=(Y_train[j]-r[0,0])*(Y_train[j]-r[0,0])/2
  print(error)


