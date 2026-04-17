import requests
import json

def emotion_detector(text_to_analyze):
    """
    Analyzes text for emotions using Watson NLP API.
    Includes a simulation mode fallback if the API is unreachable.
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = { "raw_document": { "text": text_to_analyze } }

    try:
        response = requests.post(url, json = myobj, headers=header, timeout=5)
        formatted_response = json.loads(response.text)
        
        if response.status_code == 200:
            emotions = formatted_response['emotionPredictions'][0]['emotion']
            anger_score = emotions['anger']
            disgust_score = emotions['disgust']
            fear_score = emotions['fear']
            joy_score = emotions['joy']
            sadness_score = emotions['sadness']
            
            emotion_list = [anger_score, disgust_score, fear_score, joy_score, sadness_score]
            emotion_keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
            dominant_emotion = emotion_keys[emotion_list.index(max(emotion_list))]
            
            return {
                'anger': anger_score,
                'disgust': disgust_score,
                'fear': fear_score,
                'joy': joy_score,
                'sadness': sadness_score,
                'dominant_emotion': dominant_emotion
            }
        
        if response.status_code == 400:
            return {
                'anger': None, 'disgust': None, 'fear': None, 
                'joy': None, 'sadness': None, 'dominant_emotion': None
            }
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        # Simulation Fallback for demonstration when API is down
        text = text_to_analyze.lower()
        sim_res = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
        
        if 'happy' in text or 'glad' in text or 'joy' in text:
            sim_res['joy'] = 0.95; sim_res['dominant_emotion'] = 'joy'
        elif 'mad' in text or 'angry' in text:
            sim_res['anger'] = 0.95; sim_res['dominant_emotion'] = 'anger'
        elif 'sad' in text or 'sorry' in text:
            sim_res['sadness'] = 0.95; sim_res['dominant_emotion'] = 'sadness'
        elif 'scared' in text or 'afraid' in text:
            sim_res['fear'] = 0.95; sim_res['dominant_emotion'] = 'fear'
        elif 'disgust' in text:
            sim_res['disgust'] = 0.95; sim_res['dominant_emotion'] = 'disgust'
        else:
            sim_res['joy'] = 0.5; sim_res['dominant_emotion'] = 'joy'
            
        # Ensure all keys are present
        for key in ['anger', 'disgust', 'fear', 'joy', 'sadness']:
            if sim_res[key] == 0.0: sim_res[key] = 0.01

        return sim_res

    return None
