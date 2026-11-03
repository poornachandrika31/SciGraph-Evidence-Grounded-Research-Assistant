import os
import json
import logging
from typing import List, Dict, Any, Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================
# GEMINI INITIALIZATION
# ============================================================

_gemini_initialized = False

try:
    import google.generativeai as genai

    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _gemini_initialized = True
        logger.info("Google Generative AI initialized successfully.")
    else:
        logger.warning("GEMINI_API_KEY is not configured.")

except Exception as e:
    logger.warning(
        "Google Generative AI initialization warning: %s",
        e
    )


# ============================================================
# LLM SERVICE
# ============================================================

class LLMService:
    """
    Provider-agnostic LLM abstraction layer.

    Current provider:
        Google Gemini

    Responsibilities:
        - Gemini configuration
        - Text generation
        - Structured JSON generation
        - Response parsing
        - Error reporting

    IMPORTANT:
        This service does NOT silently return unrelated demo data
        when Gemini fails.
    """

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    @classmethod
    def is_configured(cls) -> bool:
        """Return True when a Gemini API key is available."""
        return bool(settings.GEMINI_API_KEY)

    # --------------------------------------------------------
    # Text Generation
    # --------------------------------------------------------

    @classmethod
    async def generate_completion(
        cls,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4
    ) -> str:
        """
        Generate a text completion using Gemini.

        If Gemini fails, raise an explicit error instead of
        returning hardcoded scientific demo content.

        This is important because the old fallback returned
        microplastic hypotheses even for unrelated questions.
        """

        # ----------------------------------------------------
        # Check configuration
        # ----------------------------------------------------

        if not cls.is_configured():
            raise RuntimeError(
                "Gemini API is not configured. "
                "Please set GEMINI_API_KEY in the backend environment."
            )

        try:
            import google.generativeai as genai

            # Configure Gemini
            genai.configure(
                api_key=settings.GEMINI_API_KEY
            )

            # Use configured model or default
            model_name = (
                settings.GEMINI_MODEL
                or "gemini-3.5-flash"
            )

            logger.info(
                "Calling Gemini model: %s",
                model_name
            )

            # ------------------------------------------------
            # Create Gemini model
            # ------------------------------------------------

            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )

            # ------------------------------------------------
            # Generate response
            # ------------------------------------------------

            response = await model.generate_content_async(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )

            # ------------------------------------------------
            # Validate response
            # ------------------------------------------------

            if not response:
                raise RuntimeError(
                    "Gemini returned no response."
                )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            logger.info(
                "Gemini response received successfully."
            )

            return response.text

        except Exception as e:

            # Log the COMPLETE traceback in the backend
            # terminal so we can diagnose the actual problem.
            logger.exception(
                "Gemini API call failed."
            )

            # Do NOT fall back to unrelated hardcoded
            # scientific examples.
            raise RuntimeError(
                f"Gemini API call failed: {str(e)}"
            ) from e

    # --------------------------------------------------------
    # Structured JSON Generation
    # --------------------------------------------------------

    @classmethod
    async def generate_structured_json(
        cls,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from Gemini.

        The LLM is explicitly instructed to return raw JSON.
        The method also attempts to recover JSON if Gemini
        accidentally wraps it in markdown.
        """

        prompt_with_instructions = (
            f"{user_prompt}\n\n"
            "IMPORTANT: Return ONLY valid JSON format "
            "matching the requested schema. "
            "Do not include markdown code block formatting "
            "(like ```json). "
            "Return raw JSON only."
        )

        # ----------------------------------------------------
        # Ask Gemini
        # ----------------------------------------------------

        raw_text = await cls.generate_completion(
            system_prompt=system_prompt,
            user_prompt=prompt_with_instructions,
            temperature=0.2
        )

        # ----------------------------------------------------
        # Clean response
        # ----------------------------------------------------

        cleaned = raw_text.strip()

        # Remove ```json
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        # Remove ```
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        # Remove closing ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # ----------------------------------------------------
        # Direct JSON parsing
        # ----------------------------------------------------

        try:
            return json.loads(cleaned)

        except Exception as e:

            logger.warning(
                "Failed to parse JSON directly from Gemini: %s",
                e
            )

        # ----------------------------------------------------
        # Attempt object extraction
        # ----------------------------------------------------

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")

        if start_idx != -1 and end_idx != -1:

            try:
                extracted = cleaned[
                    start_idx:end_idx + 1
                ]

                return json.loads(extracted)

            except Exception as e:

                logger.warning(
                    "Object JSON extraction failed: %s",
                    e
                )

        # ----------------------------------------------------
        # Attempt array extraction
        # ----------------------------------------------------

        start_arr = cleaned.find("[")
        end_arr = cleaned.rfind("]")

        if start_arr != -1 and end_arr != -1:

            try:
                extracted = cleaned[
                    start_arr:end_arr + 1
                ]

                return json.loads(extracted)

            except Exception as e:

                logger.warning(
                    "Array JSON extraction failed: %s",
                    e
                )

        # ----------------------------------------------------
        # Nothing worked
        # ----------------------------------------------------

        raise ValueError(
            "Could not parse valid JSON from Gemini response. "
            f"Response preview: {cleaned[:500]}"
        )

    # --------------------------------------------------------
    # Optional Safe Fallback
    # --------------------------------------------------------

    @classmethod
    def _domain_fallback_completion(
        cls,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Legacy fallback retained for compatibility.

        IMPORTANT:
        This method is intentionally NOT called by
        generate_completion().

        It is kept temporarily so that other parts of the
        existing application do not break if they reference it.
        """

        logger.warning(
            "Legacy domain fallback requested. "
            "This fallback should not be used for production "
            "scientific hypothesis generation."
        )

        return json.dumps({
            "error": (
                "LLM generation is currently unavailable. "
                "Please verify Gemini configuration."
            )
        })
