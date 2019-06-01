import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 60
THRESHOLD = 1.5  # これより遠い距離の画素を無視する
BG_PATH = "./image.png"  # 背景画像のパス


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
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # 指定距離以上を無視した深度画像
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_filtered_image = (depth_image < max_dist) * depth_image
            depth_gray_image = (depth_filtered_image * 255. / max_dist).reshape((HEIGHT, WIDTH)).astype(np.uint8)

            # RGB画像
            color_image = np.asanyarray(color_frame.get_data())

            # 指定距離以上を無視したRGB画像
            color_filtered_image = (depth_filtered_image.reshape((HEIGHT, WIDTH, 1)) > 0) * color_image

            # 背景合成
            background_masked_image = (depth_filtered_image.reshape((HEIGHT, WIDTH, 1)) == 0) * bg_image
            composite_image = background_masked_image + color_filtered_image

            # 表示
            cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('demo', composite_image)
            if cv2.waitKey(1) & 0xff == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
