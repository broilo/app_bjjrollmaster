import cv2
import yt_dlp
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import subprocess

# Load the PoseNet model
# Download the model from TF Hub.
movenet = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/4")
model = movenet.signatures['serving_default']

# Initialize score variables
fighter_a_score = 0
fighter_b_score = 0

# Variables to track fighters
fighter_a_landmarks = None
fighter_b_landmarks = None


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two landmarks."""
    return np.linalg.norm([point1[0] - point2[0], point1[1] - point2[1]])


def assign_fighters(landmarks):
    """Assign fighters based on initial horizontal positions."""
    global fighter_a_landmarks, fighter_b_landmarks

    # Get center x-coordinates of landmarks (e.g., hips)
    left_hip_x = landmarks[11][1]
    right_hip_x = landmarks[12][1]
    center_x = (left_hip_x + right_hip_x) / 2

    # Assign fighters based on horizontal position
    if fighter_a_landmarks is None and fighter_b_landmarks is None:
        if center_x < 0.5:  # Left side of the frame
            fighter_a_landmarks = landmarks
        else:  # Right side of the frame
            fighter_b_landmarks = landmarks


def analyze_pose_and_score(landmarks, is_fighter_a):
    """Analyze pose landmarks and apply scoring rules for a specific fighter."""
    global fighter_a_score, fighter_b_score
    score = fighter_a_score if is_fighter_a else fighter_b_score
    feedback = []

    try:
        # Example key landmarks for PoseNet (indexes are based on PoseNet's body parts)
        left_knee = landmarks[25]  # PoseNet left knee index
        right_knee = landmarks[26]  # PoseNet right knee index
        left_hip = landmarks[11]  # PoseNet left hip index
        right_hip = landmarks[12]  # PoseNet right hip index

        # 1. Detect Takedown
        if (
            calculate_distance(left_hip, right_hip) > 0.5
            and abs(left_hip[0] - right_hip[0]) > 0.2
        ):
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Takedown detected!"
            )
            score += 2

        # 2. Detect Sweep
        if left_knee[0] < left_hip[0] and right_knee[0] < right_hip[0]:
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Sweep detected!"
            )
            score += 2

        # 3. Detect Guard Pass
        if calculate_distance(left_hip, right_hip) < 0.3 and left_hip[0] > left_knee[0]:
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Guard Pass detected!"
            )
            score += 3

        # 4. Detect Mount
        if (
            left_knee[0] < left_hip[0]
            and right_knee[0] < right_hip[0]
            and abs(left_knee[1] - right_knee[1]) > 0.3
        ):
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Mount position detected!"
            )
            score += 4

        # 5. Detect Back Control
        if (
            calculate_distance(left_knee, left_hip) < 0.3
            and calculate_distance(right_knee, right_hip) < 0.3
        ):
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Back Control detected!"
            )
            score += 4

    except Exception as e:
        feedback.append(
            f"Error in analyzing pose for {'Fighter A' if is_fighter_a else 'Fighter B'}: {e}"
        )

    if is_fighter_a:
        fighter_a_score = score
    else:
        fighter_b_score = score

    return feedback


# Load video from a YouTube link using yt-dlp
video_url = "https://www.youtube.com/watch?v=7Gy7WuGq1tw"  # Replace with your YouTube video link

# Fetch the video URL with yt-dlp
ydl_opts = {
    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Use best video and audio format
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)

    # Find the best quality video URL
    video_stream_url = None
    for format in info["formats"]:
        if (
            format.get("ext") == "mp4" and format.get("acodec") != "none"
        ):  # Ensure video and audio
            video_stream_url = format["url"]
            break

    if video_stream_url is None:
        raise ValueError("Unable to find a suitable video stream URL.")

# Use ffmpeg to open the video stream
command = [
    "ffmpeg",
    "-i",
    video_stream_url,  # Input stream URL
    "-f",
    "rawvideo",  # Output raw video
    "-pix_fmt",
    "bgr24",  # Output pixel format (BGR for OpenCV compatibility)
    "-an",  # Disable audio
    "-",
]

ffmpeg_process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Read and process video frames from ffmpeg's stdout
while True:
    raw_frame = ffmpeg_process.stdout.read(
        1920 * 1080 * 3
    )  # Adjust frame size to video resolution
    if not raw_frame:
        break

    # Convert raw frame to numpy array
    frame = np.frombuffer(raw_frame, np.uint8).reshape(
        (1080, 1920, 3)
    )  # Adjust resolution accordingly

    # Convert frame to RGB for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference with PoseNet
    input_image = tf.convert_to_tensor(rgb_frame)
    input_image = tf.image.resize(input_image, (192, 192))
    input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension
    outputs = model(input_image)

    # Extract keypoints (pose landmarks) from PoseNet output
    keypoints = outputs["output_0"].numpy()

    # Process pose landmarks and assign fighters
    if keypoints is not None:
        landmarks = keypoints[0, 0]  # Extract keypoints for the first frame

        assign_fighters(landmarks)

        # Analyze scoring separately for each fighter
        feedback_a = (
            analyze_pose_and_score(fighter_a_landmarks, is_fighter_a=True)
            if fighter_a_landmarks
            else []
        )
        feedback_b = (
            analyze_pose_and_score(fighter_b_landmarks, is_fighter_a=False)
            if fighter_b_landmarks
            else []
        )

        # Display feedback and scores
        feedback = feedback_a + feedback_b
        cv2.putText(
            frame,
            f"Fighter A: {fighter_a_score} | Fighter B: {fighter_b_score}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        for idx, line in enumerate(feedback):
            cv2.putText(
                frame,
                line,
                (10, 100 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    # Show the video frame
    cv2.imshow("Jiu-Jitsu Match Scoring", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
ffmpeg_process.stdout.close()
ffmpeg_process.stderr.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()
