import random
import cv2
import os
import mediapipe as mp
import speech_recognition as sr  # Nhập khẩu thư viện nhận diện giọng nói

class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        # Chuyển từ BGR sang RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Xử lý ảnh với thư viện mediapipe
        results = self.hands.process(imgRGB)
        hand_lms = []

        if results.multi_hand_landmarks:
            # Vẽ các landmark của bàn tay
            for handlm in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handlm, self.mpHands.HAND_CONNECTIONS)

            # Lấy tọa độ các khớp của ngón tay
            firstHand = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for id, lm in enumerate(firstHand.landmark):
                real_x, real_y = int(lm.x * w), int(lm.y * h)
                hand_lms.append([id, real_x, real_y])

        return img, hand_lms

    def count_finger(self, hand_lms):
        finger_start_index = [4, 8, 12, 16, 20]
        n_fingers = 0

        if len(hand_lms) > 0:
            # Kiểm tra ngón cái
            if hand_lms[finger_start_index[0]][1] < hand_lms[finger_start_index[0] - 1][1]:
                n_fingers += 1

            # Kiểm tra 4 ngón còn lại
            for idx in range(1, 5):
                if hand_lms[finger_start_index[idx]][2] < hand_lms[finger_start_index[idx] - 2][2]:
                    n_fingers += 1

            return n_fingers
        else:
            return -1

def draw_results(frame, user_draw):
    # Máy tính chọn ngẫu nhiên
    com_draw = random.randint(0, 2)

    # Vẽ hình ảnh cho người chơi
    frame = cv2.putText(frame, 'You', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    img_path = os.path.join("pix", str(user_draw) + ".png")
    s_img = cv2.imread(img_path)

    if s_img is None:
        print(f"Error: Không thể đọc ảnh từ {img_path}")
        return  # Dừng hàm nếu không thể đọc ảnh

    s_img = cv2.resize(s_img, (200, 300))  # Thay đổi kích thước ảnh người chơi cho phù hợp

    x_offset = 50
    y_offset = 150
    frame[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

    # Vẽ hình ảnh cho máy tính
    frame = cv2.putText(frame, 'Computer', (450, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    img_path_com = os.path.join("pix", str(com_draw) + ".png")
    s_img_com = cv2.imread(img_path_com)

    if s_img_com is None:
        print(f"Error: Không thể đọc ảnh từ {img_path_com}")
        return  # Dừng hàm nếu không thể đọc ảnh

    s_img_com = cv2.resize(s_img_com, (200, 300))  # Thay đổi kích thước ảnh máy tính cho phù hợp

    x_offset_com = 450
    y_offset_com = 150
    frame[y_offset_com:y_offset_com + s_img_com.shape[0], x_offset_com:x_offset_com + s_img_com.shape[1]] = s_img_com

    # Hiển thị kết quả
    if user_draw == com_draw:
        result = "Draw!"
    elif (user_draw == 0 and com_draw == 1) or (user_draw == 1 and com_draw == 2) or (user_draw == 2 and com_draw == 0):
        result = "You Win!"
    else:
        result = "You Lost!"

    # Hiển thị kết quả lên màn hình
    frame = cv2.putText(frame, result, (200, 500), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 255), 3, cv2.LINE_AA)

def listen_for_voice_command():
    # Khởi tạo đối tượng nhận diện giọng nói
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Đang lắng nghe lệnh giọng nói...")
        recognizer.adjust_for_ambient_noise(source)  # Điều chỉnh độ ồn môi trường
        audio = recognizer.listen(source)

    try:
        # Chỉ định ngôn ngữ là tiếng Việt
        command = recognizer.recognize_google(audio, language="vi-VN")  # Nhận diện giọng nói tiếng Việt
        print(f"Lệnh giọng nói nhận được: {command}")
        if "cái bao" in command:
            return 0  # Đá
        elif "cái búa" in command:
            return 1  # Giấy
        elif "cái kéo" in command:
            return 2  # Kéo
        else:
            print("Lệnh không hợp lệ. Hãy nói 'cái búa', 'cái bao' hoặc 'cái kéo'.")
    except sr.UnknownValueError:
        print("Xin lỗi, tôi không thể hiểu lệnh.")
    except sr.RequestError:
        print("Xin lỗi, có lỗi xảy ra với dịch vụ nhận diện giọng nói.")
    return -1  # Trả về -1 nếu không nhận diện được lệnh hợp lệ

# Khởi tạo đối tượng nhận diện tay
detector = handDetector()

# Khởi tạo webcam
cam = cv2.VideoCapture(0)

# Thiết lập độ phân giải camera (phóng to camera)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Chiều rộng: 1280px
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Chiều cao: 720px

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Xử lý hình ảnh với bộ nhận diện tay
    frame, hand_lms = detector.findHands(frame)
    n_fingers = detector.count_finger(hand_lms)

    user_draw = -1  # 0: Đá, 1: Giấy, 2: Kéo

    # Kiểm tra số ngón tay để xác định cử chỉ
    if n_fingers == 0:
        user_draw = 1  # Giấy
    elif n_fingers == 2:
        user_draw = 2  # Kéo
    elif n_fingers == 5:
        user_draw = 0  # Đá
    elif n_fingers != -1:
        print("Chỉ chấp nhận 'đá', 'giấy', 'kéo'.")
    else:
        print("Không phát hiện bàn tay.")

    # Hiển thị các điểm trên bàn tay
    if hand_lms:
        print(hand_lms)

    key = cv2.waitKey(1)

    # Hiển thị video
    cv2.imshow("game", frame)

    # Điều khiển sự kiện dừng chương trình khi bấm "q"
    if key == ord("q"):
        break
    elif key == ord(" "):  # Khi bấm phím cách, tiến hành so sánh
        draw_results(frame, user_draw)
        cv2.imshow("game", frame)
        cv2.waitKey()
    elif key == ord("w"):  # Khi bấm phím "W", bắt đầu lắng nghe lệnh giọng nói
        user_draw = listen_for_voice_command()
        if user_draw != -1:
            print(f"Người chơi chọn: {user_draw}")
            draw_results(frame, user_draw)
            cv2.imshow("game", frame)
            cv2.waitKey()

# Giải phóng camera và đóng cửa sổ
cam.release()
cv2.destroyAllWindows()
