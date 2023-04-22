import face_recognition
import cv2
import numpy as np

# 1. 각 비디오 프레임을 1/4 해상도로 처리합니다(여전히 전체 해상도로 표시).
# 2. 비디오의 모든 다른 프레임에서 얼굴만 감지

# 참고: 이 예제는 웹캠에서 읽기 위해서만 OpenCV(`cv2` 라이브러리)가 설치되어 있어야 합니다.
# 얼굴 인식 라이브러리를 사용하기 위해 OpenCV는 *필요하지* 않습니다. 이 특정 데모를 실행하려는 경우에만 필요합니다.
# 특정 데모를 실행하려는 경우에만 필요합니다. 설치하는 데 문제가 있는 경우, 이 라이브러리가 필요하지 않은 다른 데모를 사용해 보세요.

# 웹캠 #0(기본 웹캠)에 대한 참조 가져오기
video_capture = cv2.VideoCapture(0)

# 샘플 사진을 불러와서 인식하는 방법을 배움
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# 두 번째 샘플 사진을 불러와서 인식하는 방법을 배움
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# 알려진 얼굴 인코딩과 그 이름의 배열을 생성
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# 일부 변수 초기화
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 단일 프레임 비디오 촬영
    ret, frame = video_capture.read()

    # 시간 절약을 위해 비디오의 매 프레임만 처리
    if process_this_frame:
        # 더 빠른 얼굴 인식 처리를 위해 비디오 프레임 크기를 1/4 크기로 조정
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # 이미지를 BGR 색상(OpenCV에서 사용)에서 RGB 색상(얼굴 인식에서 사용)으로 변환
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # 현재 비디오 프레임에서 모든 얼굴과 얼굴 인코딩을 찾는다
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 얼굴이 알려진 얼굴과 일치하는지 확인
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 알려진_얼굴_인코딩에서 일치하는 항목이 발견되면 첫 번째 항목만 사용합니다.
            # 일치하는 항목이 True인 경우:
            # 첫_매치_인덱스 = matches.index(True)
            # name = known_face_names[first_match_index]

            # 또는 새 얼굴과 거리가 가장 작은 알려진 얼굴을 사용
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # 결과
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 감지한 프레임이 1/4 크기로 축소되었으므로 얼굴 위치를 축소
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 얼굴 주위에 상자 그리기
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 얼굴 아래에 이름이 있는 레이블 그리기
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

   # 결과 이미지 표시
    cv2.imshow('Video', frame)

    # 종료하려면 키보드의 'q' 클릭!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 핸들 놓기
video_capture.release()
cv2.destroyAllWindows()