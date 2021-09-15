# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:44:19 2021

@author: abhis
"""

import pandas as pd
import math
from sklearn.model_selection import KFold
class multinomial_NB:
    def __init__(self):
        self.prediction=[]
    
    def generate_dictionary(self,data):
        self.new_data=[]
        self.dictionary={}
        self.dictionary['ham']=[]
        self.dictionary['spam']=[]
        self.total_count=0
        self.count_ham=0
        self.count_spam=0
        self.v=0
        for row in data:
            self.total_count+=1
            
            if row[1]=='ham':
                split=row[0].split()
               
                self.count_ham+=1
                
                for x in split:
                    x=str.lower(x)                    
                    self.new_data.append(x)
                    if x not in self.dictionary['ham'] and x not in self.dictionary['spam']:
                        self.v+=1
                    self.dictionary['ham'].append(x)  
                    
            if row[1]=='spam':
                split=row[0].split()
              
                self.count_spam+=1
                
                for x in split:
                    x=str.lower(x)
                    self.new_data.append(x)
                    if x not in self.dictionary['spam'] and x not in self.dictionary['ham']:
                        self.v+=1
                    self.dictionary['spam'].append(x)  
         
    def train(self,data):
        
        self.prob_spam={}
        self.prob_ham={}
        self.dict_spam={}
        self.dict_ham={}
        
        #finding frequency of each word and maintaing two vocabulary one for spam and one for ham
        for word in self.dictionary['ham']:
            if word not in self.dict_ham:
                self.dict_ham[word]=1
            else:
                self.dict_ham[word]+=1
        
        for word in self.dictionary['spam']:
            if word not in self.dict_spam:
                self.dict_spam[word]=1
            else:
                self.dict_spam[word]+=1
                
                  
                
        for word in self.dict_spam:
            self.prob_spam[word]=(self.dict_spam[word]+1)/(len(self.dictionary['spam'])+self.v)
        for word in self.dict_ham:
            self.prob_ham[word]=(self.dict_ham[word]+1)/(len(self.dictionary['ham'])+self.v)
            
        
    def predict(self,test):
        self.test_words_prob_ham=[]
        self.test_words_prob_spam=[]
        
            
        for word in test: 
            word=str.lower(word)
            if word not in self.dict_ham:
                self.dict_ham[word]=0
                self.prob_ham[word]=(self.dict_ham[word]+1)/(len(self.dictionary['ham'])+self.v)
            if word not in self.dict_spam:
                self.dict_spam[word]=0
                self.prob_spam[word]=(self.dict_spam[word]+1)/(len(self.dictionary['spam'])+self.v)
        
        for word in test:
            word=str.lower(word)
            if word in self.prob_spam:
              self.test_words_prob_spam.append(self.prob_spam[word])
            if word in self.prob_ham:
                self.test_words_prob_ham.append(self.prob_ham[word])            
            else:
                self.test_words_prob_ham.append(self.prob_ham[word])
                self.test_words_prob_spam.append(self.prob_spam[word])
                
               
        final_spam=1
        final_ham=1
        
        for x in self.test_words_prob_ham:
            final_ham*=x
        for x in self.test_words_prob_spam:
            final_spam*=x    
        '''
        print(final_ham*self.count_ham/self.total_count)
        print(final_spam*self.count_spam/self.total_count)
        '''
        if final_ham==0:
            final_ham=math.log(self.count_ham/self.total_count)
        elif final_spam==0:
            final_spam=math.log(self.count_spam/self.total_count)
        else:    
            final_ham=math.log(final_ham)+math.log(self.count_ham/self.total_count)
            final_spam=math.log(final_spam)+math.log(self.count_spam/self.total_count)
                
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
    train, test = train_test_split(df, test_size=0.2,random_state=42)
   
    X_train=train.iloc[:,0].values
    Y_train=train.iloc[:,1].values
    
    X_test=test.iloc[:,1].values
    Y_test=test.iloc[:,0].values
    data=[]    
    for i in range(len(X_train)):
        data.append([X_train[i],Y_train[i]])
    

    data=[['chinese beijing chinese','spam'],
          ['chinese chinese shanghai','spam'],
          ['chinese macao','spam'],['tokyo japan chinese','ham']]
            
    
    msg="chinese chinese chinese tokyo japan"
    test=msg.split()
    model=multinomial_NB()
    model.generate_dictionary(data)
    model.train(model.new_data)
    model.predict(test)
    '''
    
    
    
    kfold=KFold(n_splits=5,shuffle=True,random_state=42)
    i=0
    print("Using 5 KFold in terms of accuracy ")
    model=multinomial_NB()
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
        