from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionProvideRiskInfo(Action):
    def name(self) -> str:
        return "action_provide_risk_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        message = tracker.latest_message.get('text', '')
        metadata = tracker.latest_message.get('metadata', {})
        lang = metadata.get('language', 'fr')
        translations = {
            'fr': "Je n'ai pas d'informations récentes sur votre score de risque. Veuillez lancer une prédiction d'abord.",
            'en': "I don't have recent information on your risk score. Please run a prediction first."
        }
        if any(keyword in message.lower() for keyword in ['risk', 'score', 'biomarkers', 'risque', 'biomarqueurs']):
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text=translations.get(lang, translations['fr']))
        return []