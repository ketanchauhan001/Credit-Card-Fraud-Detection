import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load dataset
data=pd.read_csv(r"C:\Users\ketan\ketan_python\credit card project\enhanced_synthetic_fraud_transactions_1000.csv")
# print(data.head().to_string())

#clean data
# print(data.isnull().sum())
data.drop_duplicates(inplace=True)

#remove the useless columns
data.drop(columns=['Transaction_ID', 'User_ID', 'Merchant_ID', 'IP_Address','User_VPA', 'Mobile_Number', 'Email_ID', 'User_ID_Merchant_End'], inplace=True)
# print(data.head().to_string())

#convert to date time format
data['Transaction_Time']=pd.to_datetime(data['Transaction_Time'])
data['User_Registration_Date']=pd.to_datetime(data['User_Registration_Date'])

#extract useful features
data['Transaction_Hour']=data['Transaction_Time'].dt.hour
data['Transaction_Day']=data['Transaction_Time'].dt.dayofweek
data['Day_Since_Registration']=(data['Transaction_Time']-data['User_Registration_Date']).dt.days

#remove the originol date time columns...because it is now useless
data.drop(columns=['Transaction_Time', 'User_Registration_Date'], inplace=True)
# print(data.head().to_string())

#encoding the label data
lec=LabelEncoder()
data['Merchant_Category']=lec.fit_transform(data['Merchant_Category'])
data['Device_Type']=lec.fit_transform(data['Device_Type'])
data['Location']=lec.fit_transform(data['Location'])
data['Card_Type']=lec.fit_transform(data['Card_Type'])

print(data.head().to_string())

#define inpute features and output target
x=data[['Transaction_Amount','Merchant_Category','Device_Type','Location','Card_Type','Latitude','Longitude','Transaction_Hour','Transaction_Day','Day_Since_Registration']]
y=data['Is_Fraud']

#split the data in train test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#applying random forest classifier model
model=RandomForestClassifier(random_state=42,max_depth=10)
model.fit(x_train,y_train)

#get the percentage of fraud of old data
percentage=model.predict_proba(x)
fraud_percentage=percentage[:,1]
data['Fraud_Percentage']=fraud_percentage*100
print(data.head(100).to_string())

#you can enter data here for prediction
#here is the column name[Transaction_Amount, Merchant_Category, Device_Type, Location, Card_Type, Latitude, Longitude, Transaction_Hour, Transaction_Day, Day_Since_Registration, Fraud_Percentage]
#we can do implement in user input data. but this is the fast way to run sample code
enter_data=[[1329.69, 3, 0, 192, 2, -46.989331, 1700.60, 18, 3, 746]]

prediction=model.predict(enter_data)
if(prediction[0]==1):
    print("Classification: Fraud")
else:
    print("Classification: Not Fraud")

predict_percentage=model.predict_proba(enter_data)
print(f"Chnce of Fraud: {predict_percentage[0][1]*100:.2f}%")
print(f"Chnce of Not Fraud: {predict_percentage[0][0]*100:.2f}%")
print()

#define the accuracy of train and test data
test_acc=model.score(x_test,y_test)*100
train_acc=model.score(x_train,y_train)*100
print(f"Testing_accuracy: {test_acc}%")
print(f"Training_accuracy: {train_acc}%")