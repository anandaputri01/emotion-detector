from EmotionDetection.emotion_detection import emotion_detector
import unittest
from unittest.mock import patch
import json

class TestEmotionDetector(unittest.TestCase):
    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector(self, mock_post):
        # Helper to setup mock response
        def set_mock(text, anger, disgust, fear, joy, sadness):
            response_data = {
                'emotionPredictions': [{
                    'emotion': {
                        'anger': anger,
                        'disgust': disgust,
                        'fear': fear,
                        'joy': joy,
                        'sadness': sadness
                    }
                }]
            }
            mock_post.return_value.text = json.dumps(response_data)
            mock_post.return_value.status_code = 200

        # Test case for joy
        set_mock('I am glad this happened', 0.1, 0.0, 0.0, 0.9, 0.0)
        result_1 = emotion_detector('I am glad this happened')
        self.assertEqual(result_1['dominant_emotion'], 'joy')
        
        # Test case for anger
        set_mock('I am really mad about this', 0.9, 0.0, 0.0, 0.1, 0.0)
        result_2 = emotion_detector('I am really mad about this')
        self.assertEqual(result_2['dominant_emotion'], 'anger')
        
        # Test case for disgust
        set_mock('I feel disgusted just hearing about this', 0.1, 0.9, 0.0, 0.0, 0.0)
        result_3 = emotion_detector('I feel disgusted just hearing about this')
        self.assertEqual(result_3['dominant_emotion'], 'disgust')
        
        # Test case for fear
        set_mock('I am so afraid that this will happen', 0.0, 0.0, 0.9, 0.0, 0.1)
        result_4 = emotion_detector('I am so afraid that this will happen')
        self.assertEqual(result_4['dominant_emotion'], 'fear')
        
        # Test case for sadness
        set_mock('I am so sad about this', 0.1, 0.0, 0.0, 0.0, 0.9)
        result_5 = emotion_detector('I am so sad about this')
        self.assertEqual(result_5['dominant_emotion'], 'sadness')

if __name__ == '__main__':
    unittest.main()
