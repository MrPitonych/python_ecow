import boto3
import pandas as pd
import json
import math
from dynamodb_json import json_util as json1

#function for test
def meantime(par, timeWindow):
    parm = []
    sum = 0
    itinsec = 55
    for i in range(math.floor(len(par) / timeWindow)):
        sum += timeWindow/itinsec
        parm.append(sum)

    return parm

#load dynamodb
dynamodb = boto3.resource("dynamodb", region_name='us-west-2', endpoint_url="http://localhost:8000")
table = dynamodb.Table('new_data')

#scan all table "new data"
response = table.scan()

#convert dynamodb to it's json
dynamodb_json = json1.dumps(response["Items"])

#convert dynamodb_json to string and convert string to pandas_json
res = json.dumps(json1.loads(dynamodb_json))

#create dataframe for work
df =pd.read_json(res, orient='records')
df= df.sort_values(by=['time']).reset_index(drop=True)
print(df)
Ax = df['gFx']
Ay = df['gFy']
Az = df['gFz']
time = df['time']

#test
time1 = meantime(time, 2)
print(time1)


