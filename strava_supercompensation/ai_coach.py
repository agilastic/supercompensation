"""
AI Training Coach using Ollama (local) or Claude API for personalized training recommendations.

This module provides an intelligent coaching assistant that analyzes your current
training state, wellness metrics, and planned workouts to give personalized advice.
"""

import os
import requests
from datetime import datetime
from typing import Dict, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AITrainingCoach:
    """AI-powered training coach using Ollama (local) or Claude API."""

    def __init__(self, use_ollama: bool = True, model: str = "llama3.1:8b", api_key: Optional[str] = None):
        """
        Initialize the AI coach.

        Args:
            use_ollama: If True, use local Ollama. If False, use Claude API.
            model: Model name for Ollama (default: llama3.1:8b) or Claude
            api_key: Anthropic API key (only needed if use_ollama=False)
        """
        self.use_ollama = use_ollama
        self.model = model

        if use_ollama:
            # Check if Ollama is running
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=2)
                if response.status_code != 200:
                    raise ConnectionError("Ollama is not responding")
            except requests.exceptions.RequestException:
                raise ConnectionError(
                    "Ollama is not running. Start it with: ollama serve\n"
                    "Or install with: brew install ollama"
                )
        else:
            # Use Claude API
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )

            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not found. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                )

            self.client = anthropic.Anthropic(api_key=self.api_key)
            if not model.startswith('claude'):
                self.model = "claude-3-5-sonnet-20241022"

    def get_recommendation(
        self,
        user_message: str,
        training_context: Dict
    ) -> str:
        """
        Get AI coach recommendation based on user question and training context.

        Args:
            user_message: User's question or concern (e.g., "I feel tired, should I rest?")
            training_context: Dict containing:
                - ctl: Chronic Training Load (fitness)
                - atl: Acute Training Load (fatigue)
                - tsb: Training Stress Balance (form)
                - hrv: Heart Rate Variability (ms)
                - sleep_score: Sleep quality (0-100)
                - rhr: Resting Heart Rate (bpm)
                - planned_workout: Today's planned workout description
                - recent_summary: Summary of last 7 days training
                - risk_state: Current overtraining risk state
                - readiness: Overall readiness percentage

        Returns:
            AI coach recommendation as formatted text
        """

        # Build context summary
        context_summary = self._format_context(training_context)

        # Create the system prompt
        system_prompt = """You are an expert endurance training coach specializing in:
- Heart rate variability (HRV) analysis
- Training load management (CTL/ATL/TSB)
- Periodization and recovery optimization
- Overtraining prevention
- Sport science and exercise physiology

You analyze athletes' objective data to provide evidence-based recommendations.
You understand both German and English workout terminology.
Be concise, actionable, and prioritize athlete safety."""

        # Create the user prompt
        user_prompt = f"""Analyze this athlete's current situation and provide a recommendation.

{context_summary}

Athlete's Question/Concern:
"{user_message}"

Provide your analysis in this format:

💡 Recommendation:
[Clear recommendation: PROCEED / MODIFY / REST]

🏃 Suggested Workout:
[Specific workout alternative if modifying, or "Rest day" if recommending rest]

📊 Reasoning:
[2-3 sentences explaining why based on the data]

⚠️ Warnings:
[Any concerns about overtraining, injury risk, or metrics to watch]

Be direct and actionable. Focus on the athlete's safety and long-term progress."""

        if self.use_ollama:
            return self._get_ollama_response(system_prompt, user_prompt)
        else:
            return self._get_claude_response(system_prompt, user_prompt)

    def _get_ollama_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from local Ollama."""
        try:
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model,
                    'prompt': full_prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 800
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result['response']
            else:
                return f"❌ Error: Ollama returned status {response.status_code}"

        except requests.exceptions.Timeout:
            return "❌ Error: Ollama request timed out. The model might still be loading. Try again in a moment."
        except requests.exceptions.ConnectionError:
            return "❌ Error: Cannot connect to Ollama. Make sure it's running with: ollama serve"
        except Exception as e:
            return f"❌ Error getting AI recommendation: {str(e)}"

    def _get_claude_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            return message.content[0].text

        except Exception as e:
            return f"❌ Error getting AI recommendation: {str(e)}\n\nPlease check your API key and internet connection."

    def _format_context(self, context: Dict) -> str:
        """Format training context into readable text for the AI."""

        # Interpret TSB status
        tsb = context.get('tsb', 0)
        if tsb < -30:
            tsb_status = "Very High Fatigue ⚠️"
        elif tsb < -10:
            tsb_status = "High Fatigue"
        elif tsb < 5:
            tsb_status = "Moderate Fatigue"
        elif tsb < 15:
            tsb_status = "Fresh"
        else:
            tsb_status = "Very Fresh/Peaked"

        # Interpret HRV
        hrv = context.get('hrv')
        hrv_str = f"{hrv} ms" if hrv else "No data"
        if hrv and hrv < 35:
            hrv_str += " (Low - possible stress/fatigue)"
        elif hrv and hrv > 60:
            hrv_str += " (Good recovery)"

        # Format sleep
        sleep = context.get('sleep_score')
        sleep_str = f"{sleep}/100" if sleep else "No data"
        if sleep and sleep < 60:
            sleep_str += " (Poor)"
        elif sleep and sleep > 80:
            sleep_str += " (Good)"

        # Risk state
        risk = context.get('risk_state', 'OPTIMAL_TRAINING')
        risk_emoji = {
            'OPTIMAL_TRAINING': '✅',
            'FUNCTIONAL_OVERREACHING': '🟡',
            'HIGH_STRAIN': '🟠',
            'NON_FUNCTIONAL_OVERREACHING': '🔴'
        }

        context_text = f"""Current Physiological State:
• Fitness (CTL): {context.get('ctl', 'N/A'):.1f} - Long-term training load
• Fatigue (ATL): {context.get('atl', 'N/A'):.1f} - Short-term training load
• Form (TSB): {context.get('tsb', 'N/A'):.1f} - {tsb_status}
• HRV (Heart Rate Variability): {hrv_str}
• Sleep Score: {sleep_str}
• Resting Heart Rate: {context.get('rhr', 'N/A')} bpm
• Overall Readiness: {context.get('readiness', 'N/A'):.0f}%
• Risk State: {risk_emoji.get(risk, '❓')} {risk.replace('_', ' ').title()}

Planned Workout Today:
{context.get('planned_workout', 'No workout planned')}

Recent Training (Last 7 Days):
{context.get('recent_summary', 'No recent data')}

Activity Log (Last 7 Days):
{context.get('activity_log', 'No activities logged')}"""

        # Add conversation history if available
        if context.get('conversation_history'):
            context_text += f"\n\nPrevious Conversation:\n{context['conversation_history']}"

        return context_text


def check_ollama_available() -> bool:
    """Check if Ollama is available and running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False


def check_anthropic_available() -> bool:
    """Check if anthropic package is installed."""
    return ANTHROPIC_AVAILABLE
