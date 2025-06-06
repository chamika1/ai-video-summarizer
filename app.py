from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os
import json
import hashlib
from werkzeug.utils import secure_filename
import time
import tempfile
import threading
from threading import Timer # Added Timer for background task
from datetime import datetime, timedelta
import shutil

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
CACHE_FOLDER = 'video_cache'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBt2dOiqJkMCwbqpzRwb1pFKuePUAEl9Fk" # Please replace with your actual key if this is a placeholder
genai.configure(api_key=GEMINI_API_KEY)

# Cleanup configuration
CLEANUP_THRESHOLD_HOURS = 6

# Global variables
upload_progress = {}
video_cache = {}  # Store processed video data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(filepath):
    """Generate MD5 hash of file for caching"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_video_cache(file_hash, video_data):
    """Save processed video data to cache"""
    cache_file = os.path.join(CACHE_FOLDER, f"{file_hash}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(video_data, f, ensure_ascii=False, indent=2)

def load_video_cache(file_hash):
    """Load processed video data from cache"""
    cache_file = os.path.join(CACHE_FOLDER, f"{file_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def process_video_with_gemini(filepath):
    """Process video with Gemini and extract comprehensive data"""
    try:
        # Upload video to Gemini
        video_file = genai.upload_file(filepath)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception("Video processing failed in Gemini")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Extract comprehensive video data in one go
        comprehensive_prompt = """
        Please analyze this video thoroughly and provide a very comprehensive analysis in JSON format.
        The goal is to extract as much information as possible, including a detailed narrative, all topics discussed, and the full dialogue script if present.
        The JSON structure should be:
        {
            "basic_info": {
                "duration_estimate": "approximate duration in HH:MM:SS format",
                "content_type": "e.g., educational, tutorial, conversation practice, entertainment, etc.",
                "main_topic": "A concise main topic or subject of the video."
            },
            "comprehensive_summary_multilingual": {
                "english": "A very detailed and comprehensive summary in English, covering all key aspects, narrative flow, examples given, and conclusions. This should be much more than a brief overview.",
                "sinhala": "ඉතා සවිස්තරාත්මක සහ සවිස්තරාත්මක සාරාංශයක් සිංහල භාෂාවෙන්, සියලුම ප්‍රධාන අංග, ආඛ්‍යාන ප්‍රවාහය, ලබා දී ඇති උදාහරණ සහ නිගමන ආවරණය කරයි. මෙය කෙටි දළ විශ්ලේෂණයකට වඩා බොහෝ සෙයින් වැඩි විය යුතුය.",
                "tamil": "ஆங்கிலத்தில் மிக விரிவான மற்றும் விரிவான சுருக்கம், அனைத்து முக்கிய அம்சங்கள், கதை ஓட்டம், கொடுக்கப்பட்ட எடுத்துக்காட்டுகள் மற்றும் முடிவுகளை உள்ளடக்கியது. இது ஒரு சுருக்கமான கண்ணோட்டத்தை விட மிக அதிகமாக இருக்க வேண்டும்."
            },
            "full_dialogue_script_multilingual": {
                "english": "The complete dialogue script or transcript of the video in English, if applicable. If no dialogue, state 'No dialogue present'.",
                "sinhala": "වීඩියෝවේ සම්පූර්ණ දෙබස් පිටපත හෝ පිටපත සිංහල භාෂාවෙන්, අදාළ නම්. දෙබස් නොමැති නම්, 'දෙබස් නොමැත' ಎಂದು සඳහන් කරන්න.",
                "tamil": "வீடியோவின் முழு உரையாடல் ஸ்கிரிப்ட் அல்லது டிரான்ஸ்கிரிப்ட் தமிழில், பொருந்தினால். உரையாடல் இல்லை என்றால், 'உரையாடல் இல்லை' எனக் குறிப்பிடவும்."
            },
            "key_points_multilingual": {
                "english": ["key point 1 in English", "key point 2 in English"],
                "sinhala": ["ප්‍රධාන කරුණ 1 සිංහලෙන්", "ප්‍රධාන කරුණ 2 සිංහලෙන්"],
                "tamil": ["முக்கிய புள்ளி 1 தமிழில்", "முக்கிய புள்ளி 2 தமிழில்"]
            },
            "topics_covered_multilingual": {
                "english": ["topic 1 in English", "topic 2 in English"],
                "sinhala": ["මාතෘකාව 1 සිංහලෙන්", "මාතෘකාව 2 සිංහලෙන්"],
                "tamil": ["தலைப்பு 1 தமிழில்", "தலைப்பு 2 தமிழில்"]
            },
            "visual_elements_description": "Detailed description of important visual content, scenes, and on-screen text.",
            "audio_elements_description": "Detailed description of audio elements like speech, music, sound effects.",
            "detailed_timestamps": [
                {
                    "time": "HH:MM:SS",
                    "description_english": "Detailed description of what happens at this time in English.",
                    "description_sinhala": "මෙම අවස්ථාවේදී සිදුවන දේ පිළිබඳ සවිස්තරාත්මක විස්තරය සිංහලෙන්.",
                    "description_tamil": "இந்த நேரத்தில் என்ன நடக்கிறது என்பதன் விரிவான விளக்கம் தமிழில்."
                }
            ],
            "keywords_multilingual": {
                "english": ["keyword1", "keyword2"],
                "sinhala": ["ප්‍රධාන වචනය1", "ප්‍රධාන වචනය2"],
                "tamil": ["முக்கியசொல்1", "முக்கியசொல்2"]
            },
            "scene_by_scene_analysis": [
                {
                    "scene_number": 1,
                    "start_time": "HH:MM:SS",
                    "end_time": "HH:MM:SS",
                    "description_english": "Detailed scene description in English.",
                    "description_sinhala": "සවිස්තරාත්මක දර්ශන විස්තරය සිංහලෙන්.",
                    "description_tamil": "விரிவான காட்சி விளக்கம் தமிழில்.",
                    "key_elements_english": ["element1", "element2"],
                    "key_elements_sinhala": ["මූලිකාංග1", "මූලිකාංග2"],
                    "key_elements_tamil": ["முக்கிய கூறுகள்1", "முக்கிய கூறுகள்2"]
                }
            ]
        }
        
        Ensure all text fields are populated comprehensively. If a section is not applicable (e.g., no dialogue), explicitly state that.
        The goal is to capture ALL relevant information from the video for detailed Q&A and summarization later.
        """
        
        response = model.generate_content([comprehensive_prompt, video_file])
        
        # Clean up uploaded file from Gemini
        genai.delete_file(video_file.name)
        
        # Parse JSON response
        try:
            actual_json_str = response.text
            
            # Attempt to remove markdown-style code block fences
            if '```json' in actual_json_str:
                actual_json_str = actual_json_str.split('```json', 1)[1].split('```', 1)[0]
            elif '```' in actual_json_str:
                 actual_json_str = actual_json_str.split('```',1)[1].split('```',1)[0]

            # Find the first '{' and last '}' to get the JSON object
            start_index = actual_json_str.find('{')
            end_index = actual_json_str.rfind('}')
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_content_to_parse = actual_json_str[start_index : end_index+1].strip()
                parsed_content = json.loads(json_content_to_parse)

                # Check if Gemini wrapped the result in an extra {"json": ...} layer
                if isinstance(parsed_content, dict) and "json" in parsed_content and len(parsed_content) == 1 and isinstance(parsed_content["json"], dict):
                    video_data = parsed_content["json"]
                else:
                    video_data = parsed_content
            else:
                raise ValueError("Could not find valid JSON object in Gemini response")

        except Exception as e_parse:
            print(f"Failed to parse Gemini JSON response: {str(e_parse)}")
            print(f"Raw Gemini response (first 500 chars): {response.text[:500]}...")
            # If JSON parsing fails, create a basic structure with raw text
            video_data = {
                "basic_info": {"main_topic": "Video content (JSON parsing failed)"},
                "comprehensive_summary_multilingual": {
                    "english": f"Raw analysis (JSON parsing failed for English summary): {response.text}",
                    "sinhala": f"සිංහල: අමු විශ්ලේෂණය (JSON විග්‍රහ කිරීම සාරාංශය සඳහා අසාර්ථක විය): {response.text}",
                    "tamil": f"தமிழ்: மூல பகுப்பாய்வு (JSON பாகுபடுத்தல் சுருக்கத்திற்கு தோல்வியடைந்தது): {response.text}"
                },
                 "full_dialogue_script_multilingual": {
                    "english": "Dialogue script not available due to parsing error.",
                    "sinhala": "දෙබස් පිටපත ලබා ගත නොහැක, විග්‍රහ කිරීමේ දෝෂයක් ඇත.",
                    "tamil": "உரையாடல் ஸ்கிரிப்ட் கிடைக்கவில்லை, பாகுபடுத்தல் பிழை."
                },
                "key_points_multilingual": {"english": ["Content analysis available from raw text"]},
                "raw_analysis": response.text # Store the raw text for debugging
            }
        
        return video_data
        
    except Exception as e:
        raise Exception(f"Video processing failed: {str(e)}")

class VideoUploadProgress:
    def __init__(self, filename):
        self.filename = filename
        self.progress = 0
        self.status = "uploading"
        self.start_time = time.time()
        self.estimated_time = 0

def update_progress(file_id, progress, status="uploading"):
    if file_id in upload_progress:
        upload_progress[file_id].progress = progress
        upload_progress[file_id].status = status
        elapsed_time = time.time() - upload_progress[file_id].start_time
        if progress > 0:
            upload_progress[file_id].estimated_time = elapsed_time / progress * (100 - progress)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        file_id = str(hash(filename + str(time.time())))
        upload_progress[file_id] = VideoUploadProgress(filename)
        
        temp_file_descriptor, temp_file_path = tempfile.mkstemp()
        os.close(temp_file_descriptor)

        try:
            file.save(temp_file_path)
            final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            def save_and_process_video(source_path, destination_path, current_file_id):
                try:
                    # Save file with progress tracking
                    with open(source_path, 'rb') as f_in:
                        f_in.seek(0, os.SEEK_END)
                        total_size = f_in.tell()
                        f_in.seek(0, os.SEEK_SET)
                        
                        with open(destination_path, 'wb') as f_out:
                            uploaded_bytes = 0
                            while True:
                                chunk = f_in.read(8192)
                                if not chunk:
                                    break
                                f_out.write(chunk)
                                uploaded_bytes += len(chunk)
                                
                                progress_percent = min((uploaded_bytes / total_size) * 100, 100) if total_size > 0 else 100
                                update_progress(current_file_id, progress_percent * 0.5)  # Upload is 50% of total progress
                                time.sleep(0.01)
                    
                    # Update progress to processing phase
                    update_progress(current_file_id, 50, "processing")
                    
                    # Generate file hash for caching
                    file_hash = get_file_hash(destination_path)
                    
                    # Check if already processed
                    cached_data = load_video_cache(file_hash)
                    
                    if cached_data:
                        # Use cached data
                        video_cache[filename] = cached_data
                        update_progress(current_file_id, 100, "completed")
                    else:
                        # Process with Gemini
                        update_progress(current_file_id, 60, "ai_processing")
                        video_data = process_video_with_gemini(destination_path)
                        
                        # Cache the results
                        save_video_cache(file_hash, video_data)
                        video_cache[filename] = video_data
                        
                        update_progress(current_file_id, 100, "completed")
                    
                except Exception as thread_exception:
                    print(f"Error in save_and_process_video: {thread_exception}")
                    update_progress(current_file_id, upload_progress[current_file_id].progress, "error")
                finally:
                    if os.path.exists(source_path):
                        os.remove(source_path)
            
            thread = threading.Thread(target=save_and_process_video, args=(temp_file_path, final_filepath, file_id))
            thread.start()
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'message': 'Upload and processing started'
            })
            
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/progress/<file_id>')
def get_progress(file_id):
    if file_id in upload_progress:
        progress_info = upload_progress[file_id]
        return jsonify({
            'progress': progress_info.progress,
            'status': progress_info.status,
            'estimated_time': progress_info.estimated_time,
            'filename': progress_info.filename
        })
    return jsonify({'error': 'File not found'}), 404

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.get_json()
        filename = data.get('filename')
        language = data.get('language', 'sinhala')
        
        if not filename:
            return jsonify({'error': 'Filename not provided'}), 400
        
        # Check if video data is cached
        if filename in video_cache:
            raw_video_data = video_cache[filename]
            
            # Handle potential {"json": actual_data} wrapper in older cache files
            if isinstance(raw_video_data, dict) and "json" in raw_video_data and len(raw_video_data) == 1 and isinstance(raw_video_data["json"], dict):
                video_data = raw_video_data["json"]
            else:
                video_data = raw_video_data

            # Get comprehensive summary in requested language
            summary_data = video_data.get('comprehensive_summary_multilingual', {})
            if language in summary_data:
                summary = summary_data[language]
            else:
                summary = summary_data.get('english', 'Comprehensive summary not available in the requested language or by default.')
            
            return jsonify({
                'success': True,
                'summary': summary,
                'cached': True
            })
        else:
            return jsonify({'error': 'Video not processed yet. Please wait for upload to complete.'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        filename = data.get('filename')
        language = data.get('language', 'sinhala')
        
        if not question or not filename:
            return jsonify({'error': 'Question and filename required'}), 400
        
        # Check if video data is cached
        if filename not in video_cache:
            return jsonify({'error': 'Video not processed yet. Please wait for upload to complete.'}), 400
        
        raw_video_data = video_cache[filename]

        # Handle potential {"json": actual_data} wrapper in older cache files
        if isinstance(raw_video_data, dict) and "json" in raw_video_data and len(raw_video_data) == 1 and isinstance(raw_video_data["json"], dict):
            video_data = raw_video_data["json"]
        else:
            video_data = raw_video_data
        
        # Use Gemini to answer based on cached data
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create context from cached video data, now using the new structure
        context_parts = []
        context_parts.append(f"Basic Info: {json.dumps(video_data.get('basic_info', {}), ensure_ascii=False, indent=2)}")
        
        summary_multilingual = video_data.get('comprehensive_summary_multilingual', {})
        if language in summary_multilingual:
            context_parts.append(f"Comprehensive Summary ({language}): {summary_multilingual[language]}")
        elif 'english' in summary_multilingual: # Fallback to English summary
             context_parts.append(f"Comprehensive Summary (english): {summary_multilingual['english']}")

        dialogue_multilingual = video_data.get('full_dialogue_script_multilingual', {})
        dialogue_script_for_context = ""
        if language in dialogue_multilingual:
            dialogue_script_for_context = dialogue_multilingual[language]
        elif 'english' in dialogue_multilingual: # Fallback to English dialogue
            dialogue_script_for_context = dialogue_multilingual['english']
        
        if dialogue_script_for_context and dialogue_script_for_context.lower() != 'no dialogue present':
             context_parts.append(f"Full Dialogue Script ({language if language in dialogue_multilingual else 'english'}):\n{dialogue_script_for_context}")


        key_points_multilingual = video_data.get('key_points_multilingual', {})
        if language in key_points_multilingual and key_points_multilingual[language]:
            context_parts.append(f"Key Points ({language}): {json.dumps(key_points_multilingual[language], ensure_ascii=False, indent=2)}")
        elif 'english' in key_points_multilingual and key_points_multilingual['english']:
            context_parts.append(f"Key Points (english): {json.dumps(key_points_multilingual['english'], ensure_ascii=False, indent=2)}")

        topics_multilingual = video_data.get('topics_covered_multilingual', {})
        if language in topics_multilingual and topics_multilingual[language]:
            context_parts.append(f"Topics Covered ({language}): {json.dumps(topics_multilingual[language], ensure_ascii=False, indent=2)}")
        elif 'english' in topics_multilingual and topics_multilingual['english']:
            context_parts.append(f"Topics Covered (english): {json.dumps(topics_multilingual['english'], ensure_ascii=False, indent=2)}")

        context_parts.append(f"Visual Elements Description: {video_data.get('visual_elements_description', 'Not available')}")
        context_parts.append(f"Audio Elements Description: {video_data.get('audio_elements_description', 'Not available')}")
        
        detailed_timestamps = video_data.get('detailed_timestamps', [])
        if detailed_timestamps:
            ts_strings = []
            for ts in detailed_timestamps:
                desc = ts.get(f"description_{language}", ts.get("description_english", "No description"))
                ts_strings.append(f"- At {ts.get('time', 'N/A')}: {desc}")
            context_parts.append(f"Detailed Timestamps ({language}):\n" + "\n".join(ts_strings))

        keywords_multilingual = video_data.get('keywords_multilingual', {})
        if language in keywords_multilingual and keywords_multilingual[language]:
            context_parts.append(f"Keywords ({language}): {json.dumps(keywords_multilingual[language], ensure_ascii=False, indent=2)}")
        elif 'english' in keywords_multilingual and keywords_multilingual['english']:
            context_parts.append(f"Keywords (english): {json.dumps(keywords_multilingual['english'], ensure_ascii=False, indent=2)}")

        scene_analysis = video_data.get('scene_by_scene_analysis', [])
        if scene_analysis:
            scene_strings = []
            for i, scene in enumerate(scene_analysis):
                desc = scene.get(f"description_{language}", scene.get("description_english", "No description"))
                elements_lang = scene.get(f"key_elements_{language}", scene.get("key_elements_english", []))
                elements_str = ", ".join(elements_lang) if elements_lang else "N/A"
                scene_strings.append(f"  Scene {scene.get('scene_number', i+1)} ({scene.get('start_time', 'N/A')} - {scene.get('end_time', 'N/A')}): {desc}. Key Elements: {elements_str}")
            context_parts.append(f"Scene-by-Scene Analysis ({language}):\n" + "\n".join(scene_strings))
        
        context = "\n\n".join(context_parts)
        
        # Special handling for "dialogue" or "දෙබස" or "සංවාදය" requests
        if "dialogue" in question.lower() or "දෙබස" in question or "සංවාදය" in question:
             dialogue_script_for_q = dialogue_multilingual.get(language, dialogue_multilingual.get('english', ''))
             if dialogue_script_for_q and dialogue_script_for_q.lower() != 'no dialogue present':
                 answer = f"මෙන්න වීඩියෝවේ {language} දෙබස:\n\n{dialogue_script_for_q}" if language == 'sinhala' else \
                          f" இதோ வீடியோவின் {language} உரையாடல்:\n\n{dialogue_script_for_q}" if language == 'tamil' else \
                          f"Here is the {language} dialogue from the video:\n\n{dialogue_script_for_q}"
                 return jsonify({'success': True, 'answer': answer, 'from_cache': True})
             else: # If no specific dialogue found but question implies it
                 answer = f"වීඩියෝවේ නිශ්චිත දෙබස් පිටපතක් {language} භාෂාවෙන් සොයාගත නොහැකි විය." if language == 'sinhala' else \
                          f"வீடியோவில் குறிப்பிட்ட உரையாடல் ஸ்கிரிப்ட் {language} மொழியில் கிடைக்கவில்லை." if language == 'tamil' else \
                          f"A specific dialogue script in {language} could not be found in the video analysis."
                 return jsonify({'success': True, 'answer': answer, 'from_cache': True})

        # Format prompt based on language for general questions
        if language == 'sinhala':
            prompt = f"""
            මෙම සවිස්තරාත්මක වීඩියෝ විශ්ලේෂණ දත්ත මත පදනම්ව පහත ප්‍රශ්නයට සිංහල භාෂාවෙන් උත්තර දෙන්න. හැකි සෑම විටම නිශ්චිත තොරතුරු (උදා: දෙබස්, දර්ශන විස්තර) උපුටා දක්වන්න:
            
            වීඩියෝ දත්ත:
            {context}
            
            ප්‍රශ්නය: {question}
            
            වීඩියෝවේ අන්තර්ගතය සමග සම්බන්ධ නම් පමණක් නිවැරදි උත්තරයක් දෙන්න.
            """
        elif language == 'tamil':
            prompt = f"""
            இந்த விரிவான வீடியோ பகுப்பாய்வு தரவின் அடிப்படையில் பின்வரும் கேள்விக்கு தமிழில் பதிலளிக்கவும். முடிந்தவரை குறிப்பிட்ட தகவல்களை (எ.கா., உரையாடல்கள், காட்சி விளக்கங்கள்) மேற்கோள் காட்டுங்கள்:
            
            வீடியோ தரவு:
            {context}
            
            கேள்வி: {question}
            
            வீடியோ உள்ளடக்கத்துடன் தொடர்புடையதாக இருந்தால் மட்டுமே சரியான பதிலை வழங்கவும்.
            """
        else:  # English
            prompt = f"""
            Based on this detailed video analysis data, answer the following question in English. Cite specific information (e.g., dialogues, scene descriptions) whenever possible:
            
            Video Data:
            {context}
            
            Question: {question}
            
            Only provide accurate answers if the question is related to the video content.
            """
        
        response = model.generate_content(prompt)
        answer = response.text
        
        return jsonify({
            'success': True,
            'answer': answer,
            'from_cache': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Question processing failed: {str(e)}'}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear video cache (admin function)"""
    try:
        # Clear memory cache
        video_cache.clear()
        
        # Clear file cache
        for filename in os.listdir(CACHE_FOLDER):
            if filename.endswith('.json'):
                os.remove(os.path.join(CACHE_FOLDER, filename))
        
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': f'Cache clearing failed: {str(e)}'}), 500

def clear_all_upload_and_cache_files():
    """Deletes ALL files from UPLOAD_FOLDER and CACHE_FOLDER."""
    deleted_files_count = 0
    errors = []
    folders_to_clear = [UPLOAD_FOLDER, CACHE_FOLDER]

    for folder in folders_to_clear:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    deleted_files_count += 1
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # Remove directory and all its contents
                    deleted_files_count += 1 # Consider a dir as one item for count, or count files inside
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                error_msg = f"Error deleting {file_path}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
    
    # Also clear the in-memory video_cache
    video_cache.clear()
    print("In-memory video_cache cleared.")
    
    return deleted_files_count, errors

def cleanup_old_files():
    """Deletes files older than CLEANUP_THRESHOLD_HOURS from UPLOAD_FOLDER and CACHE_FOLDER."""
    now = datetime.now()
    threshold = timedelta(hours=CLEANUP_THRESHOLD_HOURS)
    deleted_files_count = 0
    errors = []

    folders_to_clean = [UPLOAD_FOLDER, CACHE_FOLDER]

    for folder in folders_to_clean:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if now - file_mod_time > threshold:
                        os.remove(file_path)
                        deleted_files_count += 1
                        print(f"Deleted old file: {file_path}")
                elif os.path.isdir(file_path): # Optional: clean up empty old subdirectories if any
                    # For now, only focusing on files as per original request
                    pass
            except Exception as e:
                error_msg = f"Error deleting {file_path}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
    
    return deleted_files_count, errors

@app.route('/analyze_and_clean', methods=['POST'])
def analyze_and_clean_route():
    try:
        # This route now clears ALL files, not just old ones.
        deleted_count, errors = clear_all_upload_and_cache_files()
        if errors:
            return jsonify({
                'success': False, 
                'message': f'Full cleanup partially completed. Cleared {deleted_count} items.',
                'errors': errors
            }), 500
        return jsonify({
            'success': True, 
            'message': f'Successfully cleared {deleted_count} items from upload and cache folders.'
        })
    except Exception as e:
        return jsonify({'error': f'Cleanup process failed: {str(e)}'}), 500

# --- Automatic Cleanup Scheduler ---
def scheduled_cleanup_task():
    """Runs the cleanup task and schedules the next run."""
    print(f"[{datetime.now()}] Running scheduled cleanup of old files...")
    try:
        deleted_count, errors = cleanup_old_files()
        if errors:
            print(f"[{datetime.now()}] Scheduled cleanup completed with errors: {errors}")
        else:
            print(f"[{datetime.now()}] Scheduled cleanup completed. Deleted {deleted_count} files.")
    except Exception as e:
        print(f"[{datetime.now()}] Error during scheduled cleanup: {str(e)}")
    
    # Reschedule the task
    cleanup_interval_seconds = CLEANUP_THRESHOLD_HOURS * 60 * 60
    Timer(cleanup_interval_seconds, scheduled_cleanup_task).start()
    print(f"[{datetime.now()}] Next cleanup scheduled in {CLEANUP_THRESHOLD_HOURS} hours.")

if __name__ == '__main__':
    # Start the background cleanup task when the app starts
    # Run the first cleanup immediately (or after a short delay) and then schedule
    print(f"[{datetime.now()}] Initializing automatic file cleanup task...")
    # Optional: Run first cleanup after a short delay to allow server to fully start
    # Timer(10, scheduled_cleanup_task).start() 
    # Or run immediately and then schedule:
    scheduled_cleanup_task() 
    
    print("Optimized Video Analysis Web App Starting...")
    print("Optimized Video Analysis Web App Starting...")
    print("Features:")
    print("- Video caching for faster responses")
    print("- Comprehensive video analysis")
    print("- No repeated Gemini API calls")
    print("Make sure to set your GEMINI_API_KEY!")
    app.run(debug=True, host='0.0.0.0', port=5000)
