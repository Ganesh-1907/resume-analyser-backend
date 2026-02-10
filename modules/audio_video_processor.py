from typing import Dict
import os
import cv2
import numpy as np
import tempfile

class AudioVideoProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.whisper_model = None
        
    def _load_whisper_model(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            try:
                import whisper
                print("🎤 Loading Whisper model for transcription...")
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper model loaded successfully")
            except ImportError:
                print("⚠️  Whisper not installed. Install with: pip install openai-whisper")
                return None
            except Exception as e:
                print(f"❌ Error loading Whisper: {e}")
                return None
        return self.whisper_model
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper - THIS IS THE KEY FUNCTION"""
        try:
            model = self._load_whisper_model()
            if model is None:
                return ""
            
            print(f"🎤 Transcribing audio from: {audio_path}")
            result = model.transcribe(audio_path, language="en", fp16=False)
            transcribed_text = result["text"].strip()
            
            print(f"✅ Transcription complete: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def transcribe_audio_realtime(self, audio_path: str) -> Dict:
        """Transcribe with word-level timestamps for real-time display"""
        try:
            model = self._load_whisper_model()
            if model is None:
                return {"text": "", "segments": []}
            
            print(f"🎤 Real-time transcribing: {audio_path}")
            result = model.transcribe(audio_path, language="en", fp16=False, word_timestamps=True)
            
            return {
                "text": result["text"].strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", "en")
            }
            
        except Exception as e:
            print(f"❌ Error in real-time transcription: {e}")
            return {"text": "", "segments": []}
    
    def analyze_audio_quality(self, audio_path: str) -> Dict:
        """Analyze audio quality metrics including clarity and confidence"""
        try:
            import librosa
            import soundfile as sf
            
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate audio energy (volume/clarity indicator)
            energy = np.sum(y ** 2) / len(y)
            
            # Calculate zero crossing rate (speech quality indicator)
            zcr = np.sum(librosa.zero_crossings(y)) / len(y)
            
            # Spectral centroid (tonal quality)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_spectral_centroid = np.mean(spectral_centroids)
            
            # RMS energy for volume consistency
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            
            # Calculate quality score (0-100)
            duration_score = min(50.0, (duration / 30) * 50)  # Up to 50 points
            clarity_score = min(30.0, avg_rms * 1000)  # Up to 30 points
            consistency_score = min(20.0, (1 - np.std(rms)) * 100)  # Up to 20 points
            quality_score = duration_score + clarity_score + consistency_score
            
            # Detect speaking pace
            speaking_pace = "normal"
            if duration > 0:
                # Estimate words per second based on audio features
                estimated_wps = (len(y) / sr / duration) * 0.5
                if estimated_wps > 3:
                    speaking_pace = "fast"
                elif estimated_wps < 1.5:
                    speaking_pace = "slow"
            
            return {
                "duration_seconds": float(duration),
                "quality_score": min(100.0, quality_score),
                "clarity_score": float(clarity_score),
                "speaking_pace": speaking_pace,
                "audio_energy": float(energy),
                "spectral_quality": float(avg_spectral_centroid),
                "volume_consistency": float(1 - np.std(rms))
            }
        except ImportError:
            print("⚠️  librosa not installed. Install with: pip install librosa soundfile")
            return self._get_default_audio_analysis()
        except Exception as e:
            print(f"❌ Error analyzing audio: {e}")
            return self._get_default_audio_analysis()
    
    def _get_default_audio_analysis(self):
        """Return default audio analysis when libraries unavailable"""
        return {
            "duration_seconds": 0,
            "quality_score": 50,  # Default medium score
            "clarity_score": 50,
            "speaking_pace": "normal",
            "audio_energy": 0,
            "spectral_quality": 0,
            "volume_consistency": 0.5
        }
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str = None) -> str:
        """Extract audio track from video file - CRITICAL FOR VIDEO PROCESSING"""
        if output_audio_path is None:
            # Create temp file for audio
            temp_dir = tempfile.gettempdir()
            output_audio_path = os.path.join(temp_dir, f"audio_{os.getpid()}.wav")
        
        try:
            # Use FFmpeg to extract audio with proper settings
            cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_audio_path}" -y 2>&1'
            print(f"🎬 Extracting audio from video...")
            result = os.system(cmd)
            
            if result == 0 and os.path.exists(output_audio_path):
                file_size = os.path.getsize(output_audio_path)
                print(f"✅ Audio extracted successfully: {output_audio_path} ({file_size} bytes)")
                return output_audio_path
            else:
                print("⚠️  FFmpeg extraction failed, trying moviepy...")
                return self._extract_audio_moviepy(video_path, output_audio_path)
                
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            # Try alternative method
            try:
                return self._extract_audio_moviepy(video_path, output_audio_path)
            except:
                return None
    
    def _extract_audio_moviepy(self, video_path: str, output_audio_path: str) -> str:
        """Fallback method using moviepy"""
        try:
            from moviepy.editor import VideoFileClip
            
            print("🎬 Using moviepy for audio extraction...")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("⚠️  Video has no audio track")
                video.close()
                return None
            
            video.audio.write_audiofile(output_audio_path, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
            video.close()
            
            if os.path.exists(output_audio_path):
                print(f"✅ Audio extracted with moviepy: {output_audio_path}")
                return output_audio_path
            return None
            
        except ImportError:
            print("⚠️  moviepy not installed. Install with: pip install moviepy")
            return None
        except Exception as e:
            print(f"❌ MoviePy extraction error: {e}")
            return None
    
    def analyze_video_confidence(self, video_path: str) -> Dict:
        """Analyze video for facial expressions and confidence indicators"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  Could not open video file: {video_path}")
                return self._get_default_video_analysis()
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames for analysis (every 1 second)
            frames_to_analyze = min(30, int(duration))
            frame_interval = max(1, frame_count // frames_to_analyze) if frames_to_analyze > 0 else 1
            
            face_detected_count = 0
            eye_contact_score = 0
            brightness_scores = []
            
            # Load face and eye detectors
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            frame_idx = 0
            analyzed_frames = 0
            
            print(f"👤 Analyzing {frames_to_analyze} frames from video...")
            
            while analyzed_frames < frames_to_analyze and frame_idx < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate brightness
                brightness = np.mean(gray)
                brightness_scores.append(brightness)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    face_detected_count += 1
                    
                    # Detect eyes within face region
                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y+h, x:x+w]
                        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                        if len(eyes) >= 2:
                            eye_contact_score += 1
                
                frame_idx += frame_interval
                analyzed_frames += 1
            
            cap.release()
            
            # Calculate confidence metrics
            face_visibility = (face_detected_count / analyzed_frames * 100) if analyzed_frames > 0 else 0
            eye_contact_percentage = (eye_contact_score / analyzed_frames * 100) if analyzed_frames > 0 else 0
            avg_brightness = np.mean(brightness_scores) if brightness_scores else 0
            
            # Lighting quality (optimal is around 100-150)
            lighting_quality = "good" if 80 < avg_brightness < 180 else "needs_improvement"
            
            # Overall confidence score (weighted)
            confidence_score = (face_visibility * 0.5 + eye_contact_percentage * 0.5)
            
            print(f"✅ Video analysis complete: Face {face_visibility:.0f}%, Eyes {eye_contact_percentage:.0f}%")
            
            return {
                "video_duration": float(duration),
                "frames_analyzed": analyzed_frames,
                "face_visibility_percentage": round(face_visibility, 2),
                "eye_contact_percentage": round(eye_contact_percentage, 2),
                "confidence_score": round(confidence_score, 2),
                "lighting_quality": lighting_quality,
                "average_brightness": round(avg_brightness, 2),
                "posture_quality": "good" if face_visibility > 70 else "needs_improvement"
            }
            
        except Exception as e:
            print(f"❌ Error analyzing video: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_video_analysis()
    
    def _get_default_video_analysis(self):
        """Return default video analysis when processing fails"""
        return {
            "video_duration": 0,
            "frames_analyzed": 0,
            "face_visibility_percentage": 50,
            "eye_contact_percentage": 50,
            "confidence_score": 50,
            "lighting_quality": "unknown",
            "average_brightness": 100,
            "posture_quality": "unknown"
        }
    
    def process_interview_response(self, video_path: str) -> Dict:
        """
        MAIN PROCESSING PIPELINE
        Complete processing pipeline for interview response
        Returns transcribed text, audio analysis, and video analysis
        """
        print("\n" + "="*70)
        print("🎬 PROCESSING INTERVIEW RESPONSE")
        print("="*70)
        
        results = {
            "transcription": "",
            "audio_analysis": {},
            "video_analysis": {},
            "overall_presentation_score": 0,
            "processing_status": "starting"
        }
        
        # Step 1: Extract audio from video
        print("\n📹 Step 1: Extracting audio from video...")
        audio_path = self.extract_audio_from_video(video_path)
        
        if audio_path and os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"✅ Audio file created: {file_size} bytes")
            
            # Step 2: Transcribe audio to text (CRITICAL STEP)
            print("\n🎤 Step 2: Transcribing audio to text...")
            transcription = self.transcribe_audio(audio_path)
            
            if transcription:
                results["transcription"] = transcription
                results["processing_status"] = "transcription_success"
                print(f"✅ Transcription: {transcription[:150]}...")
            else:
                results["processing_status"] = "transcription_failed"
                print("⚠️  Transcription failed or empty")
            
            # Step 3: Analyze audio quality
            print("\n📊 Step 3: Analyzing audio quality...")
            audio_analysis = self.analyze_audio_quality(audio_path)
            results["audio_analysis"] = audio_analysis
            print(f"✅ Audio quality score: {audio_analysis['quality_score']:.1f}/100")
            
            # Clean up temp audio file
            try:
                os.remove(audio_path)
                print(f"🗑️  Cleaned up temp audio file")
            except:
                pass
        else:
            print("⚠️  Could not extract audio from video")
            results["processing_status"] = "audio_extraction_failed"
            results["audio_analysis"] = self._get_default_audio_analysis()
        
        # Step 4: Analyze video for confidence
        print("\n👤 Step 4: Analyzing video for confidence indicators...")
        video_analysis = self.analyze_video_confidence(video_path)
        results["video_analysis"] = video_analysis
        print(f"✅ Video confidence score: {video_analysis['confidence_score']:.1f}/100")
        
        # Step 5: Calculate overall presentation score
        audio_score = results["audio_analysis"].get("quality_score", 50)
        video_score = results["video_analysis"].get("confidence_score", 50)
        
        # Weighted average: 50% audio quality, 50% video confidence
        overall_score = (audio_score * 0.5 + video_score * 0.5)
        results["overall_presentation_score"] = round(overall_score, 2)
        
        print(f"\n✅ PROCESSING COMPLETE!")
        print(f"   📝 Transcription: {'Success' if results['transcription'] else 'Failed'}")
        print(f"   🎤 Audio Score: {audio_score:.1f}/100")
        print(f"   👤 Video Score: {video_score:.1f}/100")
        print(f"   ⭐ Overall Presentation: {overall_score:.1f}/100")
        print("="*70 + "\n")
        
        return results