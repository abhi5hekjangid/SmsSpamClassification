# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:10:42 2021

@author: abhis
"""


import pandas as pd
import math
from sklearn.model_selection import KFold
class multivariate_NB:
    def __init__(self):
        self.prediction=[]
    
    def generate_dictionary(self,data):
        self.new_data=[]
        self.dictionary={}
        self.dictionary['ham']=[]
        self.dictionary['spam']=[]
        self.total_count=0
        self.total_ham=0
        self.total_spam=0
        self.vocabulary_ham=[]
        self.vocabulary_spam=[]
        
       
        for row in data:
            self.total_count+=1
            
            if row[1]=='ham':
                split=row[0].split()
               
                self.total_ham+=1
                
                for x in split:
                    x=str.lower(x)                    
                    self.new_data.append(x)                
                    
                    if x not in self.vocabulary_ham:
                        self.vocabulary_ham.append(x)
                        
            if row[1]=='spam':
                split=row[0].split()
              
                self.total_spam+=1
                
                for x in split:
                    x=str.lower(x)
                    self.new_data.append(x)
                    
                    if x not in self.vocabulary_spam:
                        self.vocabulary_spam.append(x)
        self.count_document_containing_word(data)  
      
                
    def count_document_containing_word(self,data):
        self.count_ham={}
        self.count_spam={}
        
        for word in self.vocabulary_ham:
            for row in data:
                if row[1]=='ham':                    
                    if word in row[0].split():
                        if word not in self.count_ham:
                            self.count_ham[word]=1
                        else:
                            self.count_ham[word]+=1
        
        for word in self.vocabulary_spam:
            for row in data:
                if row[1]=='spam':
                    if word in row[0].split():
                        if word not in self.count_spam:
                            self.count_spam[word]=1
                        else:
                            self.count_spam[word]+=1
             
    def train(self,data):
        self.prob_spam={}
        self.prob_ham={}
        
        self.vocabulary=self.vocabulary_ham+self.vocabulary_spam
        self.vocabulary=list(set(self.vocabulary))
        for word in self.vocabulary:
            if word in self.count_ham:
                self.prob_ham[word]=(self.count_ham[word]+1)/(self.total_ham+2)
            else:
                self.prob_ham[word]=1/(self.total_ham+2)
            if word in self.count_spam:
                self.prob_spam[word]=(self.count_spam[word]+1)/(self.total_spam+2)
            else:
                self.prob_spam[word]=1/(self.total_spam+2)         
            
    def predict(self,test):        
        
        test=list(set(test))
        
        for word in test: 
            word=str.lower(word)
            if word not in self.count_ham:                
                self.prob_ham[word]=1/(self.total_ham+2)
            if word not in self.count_spam:                
                self.prob_spam[word]=1/(self.total_spam+2)          
            
        
        final_ham=math.log(self.total_ham/self.total_count)
        final_spam=math.log(self.total_spam/self.total_count)
   
     
        
        
        for word in self.vocabulary:
            if word in test:
                final_ham+=math.log(self.prob_ham[word])
              
            else:
                final_ham+=math.log(1-self.prob_ham[word])
               
        for word in self.vocabulary:
            if word in test:
                final_spam+=math.log(self.prob_spam[word])
            else:
                final_spam+=math.log(1-self.prob_spam[word])
        
       
        if final_ham>final_spam:
            self.prediction.append('ham')
        elif final_ham<final_spam:
            self.prediction.append('spam')
        else:
            self.prediction.append('equally likely')
     
    def testing(self,X_test):        
        self.test=[]
        self.prediction=[]
        for i in range(len(X_test)):
            self.test.append(X_test[i])
          
        for x in self.test:
            data=x.split()           
            self.predict(data)
            
    def single_test(self,msg):
        self.prediction=[]
        test=msg.split()
        self.predict(test)
        print("The given SMS '",msg,"' is ",self.prediction[0])
    def accuracy(self,Y_test):
        TP=FP=TN=FN=0       
        
        for i in range(len(Y_test)):
            if self.prediction[i]==Y_test[i]:
                if self.prediction[i]=='spam':
                    TP+=1
                else:
                    TN+=1
            else:
                if self.prediction[i]=='spam':
                    FP+=1
                else:
                    FN+=1
        accuracy=((TP+TN)/(TP+TN+FP+FN))*100
        print("Accuracy of model is ",accuracy)                
def main():
    df=pd.read_csv('SMSSpamCollection',sep='\t',names=['type','msg'])
    df.drop_duplicates()
    df['msg'] = df['msg'].str.replace('[^\w\s]','')
    X=df.iloc[:,1].values
    Y=df.iloc[:,0].values
    
    '''
    data=[['chinese beijing chinese','spam'],['chinese chinese shanghai','spam'], ['chinese macao','spam'],['tokyo japan chinese','ham']]
            
    
    msg="chinese chinese chinese tokyo japan"
    test=msg.split()
    model=multivariate_NB()
    model.generate_dictionary(data)
    model.train(model.new_data)
    model.predict(test)
    print(model.prediction)
    '''
    
    kfold=KFold(n_splits=5,shuffle=True,random_state=42)
    i=0
    print("Using 5 KFold in terms of accuracy ")
    model=multivariate_NB()
    for train_ix,test_ix in kfold.split(X):
        X_train,X_test=X[train_ix],X[test_ix]
        Y_train,Y_test=Y[train_ix],Y[test_ix]       
        
        
        data=[]    
        for i in range(len(X_train)):
            data.append([X_train[i],Y_train[i]])   
        
        model.generate_dictionary(data)
        model.train(model.new_data)
        model.testing(X_test)
        model.accuracy(Y_test)
    msg="You're almost there! In just 2 mins, open a Savings A/c with HDFC Bank, India's No.1 & enjoy Rs.12,500* worth benefits"
    model.single_test(msg)

if __name__=="__main__":
    main()
         