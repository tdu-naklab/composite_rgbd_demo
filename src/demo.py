import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 60
THRESHOLD = 1.5  # これより遠い距離の画素を無視する
BG_PATH = "./image.png"  # 背景画像のパス
MEDIAN_KERNEL_SIZE = 9
GAUSSIAN_KERNEL_SIZE = 9


def main():
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    max_dist = THRESHOLD/depth_scale

    bg_image = cv2.imread(BG_PATH, cv2.IMREAD_COLOR)

    try:
        while True:
            # フレーム取得
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # 深度画像
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())  # 深度画像(彩色済み)

            # 指定距離以上を無視した深度画像
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_filtered_image = (depth_image < max_dist) * depth_image
            depth_gray_filtered_image = (depth_filtered_image * 255. / max_dist).reshape((HEIGHT, WIDTH)).astype(np.uint8)
            ret, depth_mask = cv2.threshold(depth_gray_filtered_image, 1, 255, cv2.THRESH_BINARY)
            depth_mask = cv2.medianBlur(depth_mask, MEDIAN_KERNEL_SIZE)
            depth_mask = cv2.GaussianBlur(depth_mask, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
            depth_mask = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)  # 深度マスク画像

            # RGB画像
            color_image = np.asanyarray(color_frame.get_data())  # RGB画像

            # 指定距離以上を無視したRGB画像
            depth_mask_norm = (depth_mask / 255.0)
            color_filtered_image = (depth_mask_norm * color_image).astype(np.uint8)  # マスク済みRGB画像

            # 背景合成
            composite_image = bg_image
            composite_image = (composite_image.astype(np.float32) * (1 - depth_mask_norm)).astype(np.uint8)
            composite_image[0:HEIGHT, 0:WIDTH] += (color_image * depth_mask_norm).astype(np.uint8)

            # 表示`
            cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('demo', composite_image)  # 合成画像

            description1 = np.hstack((depth_color_image, depth_mask))
            description2 = np.hstack((color_image, color_filtered_image))
            description_image = np.vstack((description1, description2))
            cv2.namedWindow("demo2", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('demo2', description_image)

            if cv2.waitKey(1) & 0xff == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
