import google.generativeai as genai

genai.configure(api_key="") 
model = genai.GenerativeModel('gemini-2.0-flash-lite')
chat = model.start_chat(history=[])

def get_gemini_response(prompt):
    """Sends a prompt to Gemini and returns the response."""
    response = chat.send_message(prompt)
    return response.text

from collections import deque
from .detection_model import detect_from_window, FEATURE_COLS, WINDOW_SIZE

_window_buffer = deque(maxlen=WINDOW_SIZE)

def anomaly_detection(sensor_reading):
    reading_for_model = {k: sensor_reading[k] for k in FEATURE_COLS}

    _window_buffer.append(reading_for_model)

    if len(_window_buffer) < WINDOW_SIZE:
        return 0, {}   

    is_anom, score = detect_from_window(list(_window_buffer))

    if is_anom:
        latest = reading_for_model.copy()
        latest["score"] = score
        return 1, latest

    return 0, {}


def report_generator(anamoly_readings):
    readings_text = ", ".join([f"{k}: {v}" for k, v in anamoly_readings.items()])
    
    prompt = f"""
    An anomaly has been detected in a machine's sensor readings. 
    The current anomalous readings are: **{readings_text}**.

    Please generate a detailed anomaly report that includes the following sections:

    1.  **Anomaly Summary:** Clearly state which sensor readings are outside the normal range.
    2.  **Severity Assessment:** Assign a severity level (e.g., Low, Medium, High, Critical) and justify the choice based on the values.
    3.  **Potential Causes:** Provide a list of 2-3 likely root causes for these specific high readings (e.g., component failure, environmental factor, calibration error).
    4.  **Recommended Action:** Suggest immediate steps that a technician should take to address the issue.

    Structure the output as a formal report.
    """
    gemini_answer = get_gemini_response(prompt)
    print(gemini_answer)

def main():
    anamoly,anamoly_report=anamoly_dectection(65,10,20,12)
    if(anamoly==1):
        report_generator(anamoly_report)

main()
