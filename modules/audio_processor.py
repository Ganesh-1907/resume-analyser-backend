from typing import Dict
import os

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper"""
        try:
            import whisper
            print("Loading Whisper model...")
            model = whisper.load_model("base")
            print("Transcribing audio...")
            result = model.transcribe(audio_path, language="en")
            return result["text"]
        except ImportError:
            print("⚠ Whisper not installed. Install with: pip install openai-whisper")
            return ""
        except Exception as e:
            print(f"Error transcribing: {e}")
            return ""
    
    def analyze_audio_quality(self, audio_path: str) -> Dict:
        """Analyze audio quality metrics"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Simple quality score based on duration and amplitude
            quality_score = min(100.0, (duration / 10) * 100)
            
            return {
                "duration_seconds": float(duration),
                "quality_score": quality_score
            }
        except ImportError:
            print("⚠ librosa not installed. Install with: pip install librosa")
            return {"duration_seconds": 0, "quality_score": 0}
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return {"duration_seconds": 0, "quality_score": 0}
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str = "temp_audio.wav") -> str:
        """Extract audio track from video file using FFmpeg"""
        try:
            # Use FFmpeg to extract audio
            cmd = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path} -y 2>/dev/null"
            os.system(cmd)
            
            if os.path.exists(output_audio_path):
                return output_audio_path
            return None
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None