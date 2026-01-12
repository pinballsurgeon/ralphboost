
import unittest
from unittest.mock import patch, MagicMock
from ralphboost.gemini_client import generate_response
from ralphboost.config import Settings

class TestGeminiClient(unittest.TestCase):
    @patch('ralphboost.gemini_client._require_genai')
    def test_generate_response_success(self, mock_require):
        # Mock genai module structure
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_require.return_value = (mock_genai, mock_types)
        
        # Mock client and response
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_resp = MagicMock()
        mock_resp.text = "response text"
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 20
        mock_resp.usage_metadata.total_token_count = 30
        
        mock_client.models.generate_content.return_value = mock_resp
        
        settings = Settings(api_key="fake", model="test-model")
        
        res = generate_response("prompt", settings)
        
        self.assertEqual(res["text"], "response text")
        self.assertEqual(res["model"], "test-model")
        self.assertEqual(res["usage"]["total_tokens"], 30)
        
        # Verify config usage
        mock_types.GenerateContentConfig.assert_called()
        mock_client.models.generate_content.assert_called()
        
    @patch('ralphboost.gemini_client._require_genai')
    def test_generate_response_empty(self, mock_require):
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_require.return_value = (mock_genai, mock_types)
        
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_resp = MagicMock()
        mock_resp.text = None # Empty response
        mock_resp.usage_metadata = None
        
        mock_client.models.generate_content.return_value = mock_resp
        
        settings = Settings(api_key="fake", model="test-model")
        
        res = generate_response("prompt", settings)
        
        self.assertEqual(res["text"], "")
        self.assertEqual(res["usage"]["total_tokens"], 0)

if __name__ == '__main__':
    unittest.main()
