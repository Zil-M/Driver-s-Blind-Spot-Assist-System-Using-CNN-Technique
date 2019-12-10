from datetime import datetime
import json
import serial

ser = serial.Serial(
    port="COM1",
    baudrate=115200,
)
            
def GetSensorData(key, type):  #요청시마다 값 Refresh  
    try:
        if ser.readable():  #Serial값을 읽을 수 있을 경우 로직 실행
            res = ser.readline() #임시 변수에 Serial Read 데이터 저장
            json_data = json.loads(res.decode()[:-1]) #Json데이터 Decode 해서 자료 셍성 
            json_data['time'] = datetime.now().strftime('%Y-%m-%d  %H:%M:%S') #Timestemp생성
                
            if type == 'data': #자료 출력을 위한 로직
                return key+" : "+json_data[key] #Data값 리턴
            elif type == 'value': 
                return json_data[key] #Value값 (실제 값) 리턴
                        
    except json.JSONDecodeError:
        print("JSONDecodeError!!!")        
    except UnicodeDecodeError:
        print("UNICODE DecodeError!!!")
