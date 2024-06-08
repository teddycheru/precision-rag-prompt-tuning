import unittest
from unittest.mock import patch
from src.prompt_generation import generate_prompts, main

class TestPromptGeneration(unittest.TestCase):

    def test_generate_prompts(self):
        description = "Explain the concept of RAG in AI"
        scenarios = ["in a business context", "for educational purposes", "for technical documentation"]
        target_audience = "non-experts"
        tone = "friendly"
        additional_instructions = "Include examples where relevant."

        expected_prompts = [
            "Explain the concept of RAG in AI in the context of in a business context for non-experts with a friendly tone. Include examples where relevant.",
            "Explain the concept of RAG in AI in the context of for educational purposes for non-experts with a friendly tone. Include examples where relevant.",
            "Explain the concept of RAG in AI in the context of for technical documentation for non-experts with a friendly tone. Include examples where relevant."
        ]

        generated_prompts = generate_prompts(description, scenarios, target_audience, tone, additional_instructions)

        self.assertEqual(generated_prompts, expected_prompts)

    @patch('prompt_generation.rag_generate')
    @patch('builtins.print')
    def test_main(self, mock_print, mock_rag_generate):
        mock_rag_generate.return_value = "Generated response"

        main()

        self.assertEqual(mock_print.call_count, 12) 
        mock_rag_generate.assert_called()

if __name__ == "__main__":
    unittest.main()
