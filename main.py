from ultralytics import YOLO 
if __name__ == "__main__":
    INPUT_VIDEO = "jabarismith.mp4"
    OUTPUT_VIDEO = "output_detection_pose.mp4"
    model = YOLO("models/player_detector.pt")
    results = model.predict(INPUT_VIDEO, save = True)
    for box in results[0].boxes:
        print(box)
    '''
    print("="*60)
    print("Basketball Shot Detector - YOLOv8-Pose Enhanced")
    print("="*60)
    print(f"Input: {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}\n")
    
    # Optional: visualize calibration points
    # visualize_frame_points(INPUT_VIDEO)
    
    # Run detection
    #H = detect_court_homography()
    #detect_basketball_objects(INPUT_VIDEO, H, OUTPUT_VIDEO)
    
    print("\n" + "="*60)
    print("Improvements in this version:")
    print("✓ YOLOv8-pose for keypoint detection")
    print("✓ Pose validation (standing vs sitting)")
    print("✓ Leg articulation check (playing vs static)")
    print("✓ Vertical span validation")
    print("✓ Multi-point court boundary check")
    print("✓ Size and aspect ratio filtering")
    print("="*60)
    print("\nTuning tips:")
    print("1. Adjust vertical_span threshold (currently 80px) based on your resolution")
    print("2. Adjust max_leg_bend threshold (currently 30px) for pose sensitivity")
    print("3. Use yolov8m-pose.pt or yolov8l-pose.pt for even better accuracy")
    print("4. Tweak court boundary coordinates if needed")
    print("="*60)
    '''