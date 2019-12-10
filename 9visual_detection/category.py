'''
[초음파] - 차량 프레임 사각지대용
20cm이하 : 빨간색, 연속 비프름
20cm이상 50cm 미만 : 주황색 , 초당 4회
50cm이상 1m미만 : 초록색, 초당 1회

[라이다] - 운전자 전방 거리 측정
1m이상 2m미만 - 1단계 오브젝트 알림 메세지
50cm이상 1m미만 - 2단계 20km 감속 메세지
30cm이상 50cm미만 - 3단계 주행 일시정지 메세지
'''
import cv2

def img_overray(background_img, logo_img):
    logo_imgs = cv2.resize(logo_img, dsize=(640,480), interpolation=cv2.INTER_AREA)
    rows,cols,channels = logo_imgs.shape
    roi = background_img[0:rows, 0:cols]

    logo_img_gray = cv2.cvtColor(logo_imgs,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_img_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #cv2.imshow('mask_inv',mask_inv)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(logo_imgs,logo_imgs,mask = mask)

    dst = cv2.add(img1_bg,img2_fg)
    background_img[0:rows, 0:cols ] = dst

    return background_img


def Lider_system(lidar_value, object_name, frame_b):

    Lidar_1step_img = cv2.imread('./image/system_1step.png')
    Lidar_2step_img = cv2.imread('./image/system_2step.png')
    Lidar_3step_img = cv2.imread('./image/system_3step.png')
    
    if(lidar_value >= 100 and lidar_value < 200):
        frame_r=img_overray(frame_b, Lidar_1step_img)
        cv2.putText(frame_r, object_name, (220, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255),2)       
    elif(lidar_value >= 50 and lidar_value < 100):
        frame_r=img_overray(frame_b, Lidar_2step_img)
        
    elif(lidar_value >= 30 and lidar_value < 50):
        frame_r=img_overray(frame_b, Lidar_3step_img)
    else:
        print("센서범위 밖입니다.")

    return frame_r


def Sonar_system(L_value, R_value, frame):
    SonarL = int(L_value)
    SonarR = int(R_value)

    if(SonarL is None or SonarR is None):
        return;

    if(SonarL < 20):
        cv2.circle(frame, (610, 100), 20, (0, 0, 255), -1)
    elif(SonarR < 20):
        cv2.circle(frame, (610, 100), 20, (0, 0, 255), -1)
    elif(SonarL >= 20 and SonarL < 50):
        cv2.circle(frame, (610, 100), 20, (0, 127, 255), -1)
    elif(SonarR >= 20 and SonarR < 50):  
        cv2.circle(frame, (610, 100), 20, (0, 127, 255), -1)
    elif(SonarDistR > 50):
        cv2.circle(frame, (610, 100), 20, (0, 255, 0), -1)
    elif(SonarDistR > 50):
        cv2.circle(frame, (610, 100), 20, (0, 255, 0), -1)
    
    cv2.imshow("Lider_system", frame)
    
