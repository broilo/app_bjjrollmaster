import cv2
import yt_dlp
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import subprocess

# Load the PoseNet model from TF Hub
movenet = hub.load(
    "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/4"
)
model = movenet.signatures["serving_default"]

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
    """Assign fighters based on initial horizontal positions or update positions."""
    global fighter_a_landmarks, fighter_b_landmarks

    # Get center x-coordinates of landmarks (e.g., hips)
    left_hip_x = landmarks[11][1]
    right_hip_x = landmarks[12][1]
    center_x = (left_hip_x + right_hip_x) / 2
    print(f"Center X: {center_x}")

    # If fighters are already assigned, check if they need to be reassigned
    if fighter_a_landmarks is not None and fighter_b_landmarks is not None:
        # Reassign if fighter's center_x position crosses over
        if center_x < 0.5:
            if (
                fighter_b_landmarks
                and center_x
                > (fighter_b_landmarks[11][1] + fighter_b_landmarks[12][1]) / 2
            ):
                fighter_a_landmarks, fighter_b_landmarks = (
                    fighter_b_landmarks,
                    fighter_a_landmarks,
                )
        else:
            if (
                fighter_a_landmarks
                and center_x
                < (fighter_a_landmarks[11][1] + fighter_a_landmarks[12][1]) / 2
            ):
                fighter_a_landmarks, fighter_b_landmarks = (
                    fighter_b_landmarks,
                    fighter_a_landmarks,
                )
    else:
        # Assign fighters based on horizontal position if they are not yet assigned
        if center_x < 0.5:
            fighter_a_landmarks = landmarks
        else:
            fighter_b_landmarks = landmarks


def analyze_pose_and_score(landmarks, is_fighter_a):
    """Analyze pose landmarks and apply scoring rules for a specific fighter."""
    global fighter_a_score, fighter_b_score
    score = fighter_a_score if is_fighter_a else fighter_b_score
    feedback = []

    try:
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_hip = landmarks[11]
        right_hip = landmarks[12]

        print(f"Left Hip: {left_hip}, Right Hip: {right_hip}")
        print(f"Left Knee: {left_knee}, Right Knee: {right_knee}")

        # Checking takedown (example condition)
        if (
            calculate_distance(left_hip, right_hip) > 0.5
            and abs(left_hip[0] - right_hip[0]) > 0.2
        ):
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Takedown detected!"
            )
            score += 2

        # Checking sweep (example condition)
        if left_knee[0] < left_hip[0] and right_knee[0] < right_hip[0]:
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Sweep detected!"
            )
            score += 2

        # Checking guard pass (example condition)
        if calculate_distance(left_hip, right_hip) < 0.3 and left_hip[0] > left_knee[0]:
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Guard Pass detected!"
            )
            score += 3

        # Checking mount position (example condition)
        if (
            left_knee[0] < left_hip[0]
            and right_knee[0] < right_hip[0]
            and abs(left_knee[1] - right_knee[1]) > 0.3
        ):
            feedback.append(
                f"{'Fighter A' if is_fighter_a else 'Fighter B'}: Mount position detected!"
            )
            score += 4

        # Checking back control (example condition)
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
video_url = "https://www.youtube.com/watch?v=o28Uq3XuMQM"  # Replace with your YouTube video link

ydl_opts = {
    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)

    video_stream_url = None
    for format in info["formats"]:
        if format.get("ext") == "mp4" and format.get("acodec") != "none":
            video_stream_url = format["url"]
            break

    if video_stream_url is None:
        raise ValueError("Unable to find a suitable video stream URL.")

command = [
    "ffmpeg",
    "-i",
    video_stream_url,
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-an",
    "-",
]

ffmpeg_process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

while True:
    raw_frame = ffmpeg_process.stdout.read(1920 * 1080 * 3)
    if not raw_frame:
        break

    frame = np.frombuffer(raw_frame, np.uint8).reshape((1080, 1920, 3))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_image = tf.convert_to_tensor(rgb_frame)
    input_image = tf.image.resize(input_image, (192, 192))
    input_image = tf.expand_dims(input_image, axis=0)
    outputs = model(input_image)

    keypoints = outputs["output_0"].numpy()

    if keypoints is not None:
        landmarks = keypoints[0, 0]

        assign_fighters(landmarks)

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

    cv2.imshow("Jiu-Jitsu Match Scoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ffmpeg_process.stdout.close()
ffmpeg_process.stderr.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()

# Print final scores
print(f"Final Scores: Fighter A: {fighter_a_score}, Fighter B: {fighter_b_score}")
