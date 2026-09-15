from services.audio_processor import preprocess_audio


file_path = r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\Ai_video.mp3"

result = preprocess_audio(
    input_path=file_path,
    original_filename="test_audio"
)

print("\n===== PHASE 3 RESULT =====")

for key, value in result.items():
    print(f"{key}: {value}")