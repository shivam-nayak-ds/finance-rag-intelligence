import logging
import os
import sys
from typing import List, Optional

sys.path.append(os.getcwd())
from dotenv import load_dotenv

from src.generation.prompt_templates import build_rag_prompt, FINANCE_SYSTEM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

class LLMGenerationError(Exception):
    """Custom exception raised when LLM generation fails."""
    pass

class LLMChain:
    """
    Core Generation Engine for Finance RAG.
    Orchestrates prompting and interaction with external inference providers (Groq/Gemini).
    """

    def __init__(self) -> None:
        """Initializes the LLM Chain and establishes connections with cloud providers."""
        self.client = None
        self.mode: Optional[str] = None
        self.model_name: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Dynamically detects available API keys and initializes the primary 
        or fallback inference models.
        """
        # Primary Backend: Groq (Low latency inference)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=groq_key)
                self.mode = "groq"
                self.model_name = "llama-3.3-70b-versatile"
                logger.info(f"LLM Provider Initialized: Groq (Model: {self.model_name})")
                return
            except ImportError:
                logger.error("Groq SDK is not installed. Run `pip install groq`.")
            except Exception as e:
                logger.warning(f"Groq setup failed: {e}")

        # Fallback Backend: Google Gemini
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
                self.mode = "gemini"
                self.model_name = "gemini-1.5-flash"
                logger.info(f"LLM Provider Initialized: Gemini (Model: {self.model_name})")
                return
            except ImportError:
                logger.error("Google GenerativeAI SDK is not installed.")
            except Exception as e:
                logger.warning(f"Gemini setup failed: {e}")

        # Critical Failure if no providers are available
        logger.error("Initialization Failed: Neither GROQ_API_KEY nor GOOGLE_API_KEY found in environment.")

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates a grounded answer based on the retrieved financial context.

        Args:
            query (str): The user's financial question.
            context_chunks (List[str]): Reranked document chunks to ground the LLM.

        Returns:
            str: The generated response.

        Raises:
            LLMGenerationError: If inference fails or no backend is activated.
        """
        if not self.mode:
            raise LLMGenerationError("No LLM backend available. Ensure API keys are configured correctly.")

        # Constructing grounded prompt
        user_prompt = build_rag_prompt(query, context_chunks)
        logger.info(f"Executing Inference Request | Mode: {self.mode.upper()} | Context Size: {len(context_chunks)} chunks")

        if self.mode == "groq":
            return self._generate_groq(user_prompt)
        elif self.mode == "gemini":
            return self._generate_gemini(user_prompt)
        
        raise LLMGenerationError("Invalid LLM mode encountered.")

    def _generate_groq(self, user_prompt: str) -> str:
        """Executes inference via Groq's Chat Completions endpoint."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": FINANCE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,  # Strict adherence to facts
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API Inference Error: {e}", exc_info=True)
            raise LLMGenerationError(f"Groq Inference Failed: {str(e)}") from e

    def _generate_gemini(self, user_prompt: str) -> str:
        """Executes inference via Google Generative AI endpoint."""
        try:
            full_prompt = f"{FINANCE_SYSTEM_PROMPT}\n\n{user_prompt}"
            response = self.client.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API Inference Error: {e}", exc_info=True)
            raise LLMGenerationError(f"Gemini Inference Failed: {str(e)}") from e


if __name__ == "__main__":
    # Integration test block
    TEST_QUERY = "What is Wipro's revenue in Q4 FY24?"
    MOCK_CHUNKS = [
        "Wipro's operating margin in Q4 FY24 stood at 16.1%, up 30 bps sequentially.",
        "Wipro IT Services revenue was $2.63 billion in Q4 FY24, down 4.4% YoY."
    ]
    
    chain = LLMChain()
    try:
        answer = chain.generate(query=TEST_QUERY, context_chunks=MOCK_CHUNKS)
        print("\n" + "="*70)
        print(f"QUESTION: {TEST_QUERY}")
        print("="*70)
        print(f"ANSWER:\n{answer}")
        print("="*70 + "\n")
    except LLMGenerationError as err:
        logger.error(f"Test Failed: {str(err)}")
