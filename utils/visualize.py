import random
import cv2
import numpy as np

def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def visualize_assign(img, predictions, input_path, output_path='./parking_set/output_video.mp4'):
    cap = cv2.VideoCapture(input_path)
    for det in predictions:
        if det.shape[0] != 0:
            visualize_polygons = []
            parking_points = det[..., :8].detach()
            for point in parking_points:
                p = point.cpu().numpy().reshape(4, 2)
                visualize_polygons.append(p)
        else:
            visualize_polygons = None
    if visualize_polygons is not None:
        for polygon_points in visualize_polygons:
            hull = cv2.convexHull(polygon_points)
            cv2.polylines(img, [hull.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    frame_width = 640
    frame_height = 640
    fps = 1 #cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out.write(img)
    cv2.imshow('Frame', img)

def video_result(polygons = None, video_path='./parking_set/car.mp4', output_path='./parking_set/output_video.mp4'):
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open video.")
        exit()

    # 创建输出视频文件
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(frame_width, frame_height, fps)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 在当前帧上绘制不规则四边形
        if polygons is not None:
            for polygon_points in polygons:
                hull = cv2.convexHull(polygon_points)
                cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        # 将帧写入输出视频文件
        out.write(frame)

        # 显示当前帧
        cv2.imshow('Frame', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 定义不规则四边形的顶点坐标
    polygons = [
    np.array([[100.0, 100.0], [400.0, 400.0], [300.0, 150.0], [150.0, 350.0]], dtype=np.int32),
    np.array([[200.0, 200.0], [500.0, 500.0], [400.0, 250.0], [250.0, 450.0]], dtype=np.int32),
    # 添加更多的四边形，每个四边形用一个数组表示
    ]
    video_result(polygons)