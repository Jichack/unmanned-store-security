import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import gc

# === 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'dataset')
RAW_SAVE_PATH = os.path.join(BASE_DIR, 'raw_data') # 중간 저장소

# YOLO 모델 설정 (GPU 권장)
MODEL_PATH = 'yolo11n-pose.pt'
IMGSZ = 640  # 속도를 높이려면 320으로 줄여도 됨
CONF_THRES = 0.3 # 나중에 마스킹 할거니까 좀 낮게 설정해서 다 잡음

def extract_raw_skeleton(video_path, save_path):
    # 이미 처리된 파일이면 스킵 (이어하기 기능)
    if os.path.exists(save_path):
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    # 모델 로드 (함수 안에서 로드하면 메모리 낭비될 수 있으나, 단일 프로세스라 가정)
    model = YOLO(MODEL_PATH) 
    
    # ID별 트랙 저장
    tracks = defaultdict(dict)
    
    # YOLO 추론 (Stream 모드 사용으로 메모리 절약)
    # verbose=False로 로그 끔
    results = model.track(source=video_path, stream=True, persist=True, 
                          imgsz=IMGSZ, conf=CONF_THRES, verbose=False, vid_stride=1)
    
    for frame_idx, result in enumerate(results):
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            kps = result.keypoints.data.cpu().numpy() # (N, 17, 3) (x, y, conf)
            
            for i, track_id in enumerate(ids):
                # 정규화 하지 않은 원본 좌표 저장 (나중에 전처리 단계에서 함)
                tracks[track_id][frame_idx] = kps[i]
                
    cap.release()
    
    # [가장 활동적인 사람 1명 선정]
    if not tracks:
        return # 감지된 사람 없음

    best_track_id = -1
    max_score = -1
    
    for track_id, data_dict in tracks.items():
        # 너무 짧은 트랙(전체의 10% 미만) 제외
        if len(data_dict) < total_frames * 0.1: continue
        
        sorted_indices = sorted(data_dict.keys())
        valid_kps = np.array([data_dict[t] for t in sorted_indices])
        
        # 움직임 점수 계산
        movement_score = 0
        if len(valid_kps) > 1:
            diff = valid_kps[1:, :, :2] - valid_kps[:-1, :, :2]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            movement_score = np.sum(dist)
            
        # 지속 시간 + 움직임
        total_score = movement_score + (len(valid_kps) * 0.5)
        
        if total_score > max_score:
            max_score = total_score
            best_track_id = track_id
            
    if best_track_id == -1: return

    # [저장 포맷]
    # (Total_Frames, 17, 3)의 Dense한 배열을 만듦. 감지 안 된 곳은 0으로 채움.
    # 이렇게 해야 나중에 인덱싱하기 편함.
    final_array = np.zeros((total_frames, 17, 3), dtype=np.float32)
    best_data = tracks[best_track_id]
    
    for t, kp in best_data.items():
        if t < total_frames:
            final_array[t] = kp
            
    # 메타데이터와 함께 저장 (Dictionary)
    save_data = {
        'fps': fps,
        'width': width,
        'height': height,
        'keypoints': final_array # (T, 17, 3)
    }
    
    np.save(save_path, save_data)
    
    # 메모리 정리
    del model, tracks, results, final_array
    gc.collect()
    torch.cuda.empty_cache() 

def main():
    os.makedirs(RAW_SAVE_PATH, exist_ok=True)
    
    video_files = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))
    
    print(f"🚀 Step 1 시작: 총 {len(video_files)}개 영상 처리")
    print(f"💾 저장 경로: {RAW_SAVE_PATH}")
    print("ℹ️  이 작업은 오래 걸리지만, 중간에 꺼도 다시 켜면 이어서 합니다.")

    # GPU 하나만 쓴다면 그냥 순차 처리가 가장 안정적입니다.
    # 멀티프로세싱 하려면 GPU 메모리 관리 복잡해짐.
    for video_path in tqdm(video_files):
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(RAW_SAVE_PATH, file_name + '.npy')
        
        try:
            extract_raw_skeleton(video_path, save_path)
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")

if __name__ == "__main__":
    main()