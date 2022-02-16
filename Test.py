import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
from scipy import fft, fftpack, ifft, signal
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
import sklearn.pipeline

# 얼굴 인식 클래스
face_detector = dlib.get_frontal_face_detector()
# 랜드마크 찾는 클래스
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 얼굴 정렬 클래스
face_aligner = face_utils.FaceAligner(landmark_detector, desiredFaceWidth=256)


def showImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getFaceDetect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 원본 얼굴 좌표
    rects = face_detector(gray_image, 0)
    rect = rects[0]
    if len(rects) <= 0:
        return None, None, None, None
    # 원본 랜드마크 좌표
    landmarks = landmark_detector(gray_image, rect)
    landmarks = face_utils.shape_to_np(landmarks)

    # 얼굴 정렬
    aligned_image = face_aligner.align(image, gray_image, rect)
    # 정렬된 얼굴 좌표
    aligned_rects = face_detector(aligned_image, 0)
    # 정렬된 랜드마크 좌표
    aligned_landmarks = landmark_detector(aligned_image, aligned_rects[0])
    aligned_landmarks = face_utils.shape_to_np(aligned_landmarks)

    # for (x,y) in aligned_landmarks:
    #     cv2.circle(aligned_image, (x, y),1,(0,0,255),-1)

    return rects[0], landmarks, aligned_image, aligned_landmarks


def extract_ROI(image, landmarks):
    ROI1 = image[landmarks[29][1]:landmarks[33][1],  # right cheek
           landmarks[54][0]:landmarks[12][0]]
    ROI2 = image[landmarks[29][1]:landmarks[33][1],  # left cheek
           landmarks[4][0]:landmarks[48][0]]
    return ROI1, ROI2


def BGR2YCgCo(image):
    ycgco_image = np.zeros_like(image)
    (h, w, c) = image.shape
    mask = [[1 / 4, 1 / 2, 1 / 4], [-1 / 4, 1 / 2, -1 / 4], [-1 / 2, 0, 1 / 2]]
    for i in range(h):
        for j in range(w):
            ycgco_image[i][j] = mask @ image[i][j]
    return ycgco_image


def extract_ROI_calculate_Cg_MEAN(image, landmarks):
    ROI1, ROI2 = extract_ROI(image, landmarks)

    # BGR to YCgCo
    ycgco_ROI1 = BGR2YCgCo(ROI1)
    ycgco_ROI2 = BGR2YCgCo(ROI2)

    # Cg값의 평균 구하기
    ROI1_Cg_Mean = np.mean(ycgco_ROI1[:, :, 1])
    ROI2_Cg_Mean = np.mean(ycgco_ROI2[:, :, 1])
    ROI_Mean = (ROI1_Cg_Mean + ROI2_Cg_Mean) / 2

    return ROI_Mean


def showGraph(x, y):
    plt.plot(x, y)
    plt.show()


def interpolation(data_buffer, times):
    '''
    interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
    '''
    L = len(data_buffer)

    even_times = np.linspace(times[0], times[-1], L)

    interp = np.interp(even_times, times, data_buffer)
    interpolated_data = np.hamming(L) * interp
    return interpolated_data

def normalization(data_buffer):
    '''
    normalize the input data buffer
    '''

    # normalized_data = (data_buffer - np.mean(data_buffer))/np.std(data_buffer)
    normalized_data = data_buffer / np.linalg.norm(data_buffer)

    return normalized_data


def fft(data_buffer, fps):
    '''

    '''

    L = len(data_buffer)

    freqs = float(fps) / L * np.arange(L / 2 + 1)

    freqs_in_minute = 60. * freqs

    raw_fft = np.fft.rfft(data_buffer * 30)
    fft = np.abs(raw_fft) ** 2

    interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]

    # print(freqs_in_minute)
    interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
    freqs_of_interest = freqs_in_minute[interest_idx_sub]

    fft_of_interest = fft[interest_idx_sub]

    # pruned = fft[interest_idx]
    # pfreq = freqs_in_minute[interest_idx]

    # freqs_of_interest = pfreq
    # fft_of_interest = pruned

    return fft_of_interest, freqs_of_interest


if __name__ == "__main__":

    # 30 프레임 마다 심박수 구하기
    RR = []

    # 비디오 영상 읽기
    video = cv2.VideoCapture("2.mov")

    # 영상 정보 가져오기
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
    print(video_fps, frame_count, frame_pos)
    start = time.time()

    times = []
    cg_means = []
    while frame_pos < frame_count:
        # 프레임 한장 읽기
        retval, frame = video.read()
        if retval is None or retval is False:
            break
        frame_pos = video.get(cv2.CAP_PROP_POS_FRAMES)

        # openCV에서 BGR이미지를 가져와서 가로800으로 리사이즈
        resize_image = imutils.resize(frame, width=800)  # numpy.ndarray

        # 얼굴 인식
        rect, landmarks, aligned_image, aligned_landmarks = getFaceDetect(frame)  # numpy.ndarray

        # 얼굴이 없다면 No face deteced 출력후 종료
        if rect is None:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            showImage(frame)
            exit(0)

        # ROI영역에서 Cg값의 평균 구하기
        ROI_Mean = extract_ROI_calculate_Cg_MEAN(aligned_image, aligned_landmarks)
        # print(frame_pos, ROI_Mean)

        # 현재 시간과 평균값 저장
        cg_means.append(ROI_Mean)
        times.append(video.get(cv2.CAP_PROP_POS_MSEC))

        if len(cg_means) >= 90 and len(cg_means) % 30 == 0:
            detrended_data = signal.detrend(cg_means)
            interpolated_data = interpolation(detrended_data, times)
            normalized_data = normalization(interpolated_data)
            fft_of_interest, freqs_of_interest = fft(normalized_data, video_fps)
            max_arg = np.argmax(fft_of_interest)
            bpm = freqs_of_interest[max_arg]
            RR.append(60000/bpm)
            print(60000/bpm, bpm )

    print(time.time()-start)

rri = []
# RR 구하기
for i in range(1, len(RR)):
    rri.append(RR[i]-RR[i-1])

rri = np.array(rri)

predict = [0,0,0,0,0,0,0]
predict[0] = np.mean(RR)
predict[1] = np.median(RR)
predict[2] = np.std(RR)
predict[3] = np.sqrt(np.sum(rri**2))/len(rri)
predict[4] = np.std(rri)
predict[5] = predict[2]/predict[3]
predict[6] = 60000/predict[0]

predict = pd.DataFrame([predict], columns=["MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR"])
print(predict)

select = SelectKBest(k=7)
train = pd.read_csv("train4.csv")
target = 'condition'
hrv_features = list(train)
hrv_features = [x for x in hrv_features if x not in [target]]
X_train = train[hrv_features]
y_train = train[target]
clf = RandomForestRegressor(n_estimators=100, max_features='log2', n_jobs=-1)
steps = [('feature_selection', select),
         ('model', clf)]
pipeline = sklearn.pipeline.Pipeline(steps)
pipeline.fit(X_train, y_train)
y_prediction = pipeline.predict(predict)
print(y_prediction)
print((time.time())-start)



