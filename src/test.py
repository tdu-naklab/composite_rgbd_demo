import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    depth_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue
        
            color_image = np.asanyarray(color_frame.get_data())
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            images = np.hstack((color_image, depth_color_image))
            cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('demo', images)
            if cv2.waitKey(1) & 0xff == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()