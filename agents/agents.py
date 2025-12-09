import google.generativeai as genai

genai.configure(api_key="")

model = genai.GenerativeModel("gemini-2.0-flash-lite")

def get_gemini_response(prompt):
    """Fresh call (no history) to save tokens."""
    response = model.generate_content(prompt)
    return response.text.strip()

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
    
    if severity == "low":
        urgency = "monitor"
        notify_app = True
        notify_whatsapp = False
        trigger_voice_call = False
        recommended_sla_hours = 72   # service within 3 days 
    elif severity == "medium":
        urgency = "schedule_soon"
        notify_app = True
        notify_whatsapp = True
        trigger_voice_call = False
        recommended_sla_hours = 24   # service within 24 hours
    else:  
        urgency = "immediate_service"
        notify_app = True
        notify_whatsapp = True
        trigger_voice_call = True
        recommended_sla_hours = 4    # service ASAP (e.g., 4 hours)

    latest = reading_for_model.copy()
    latest.update({
        "score": score,
        "severity": severity,
        "urgency": urgency,
        "notify_app": notify_app,
        "notify_whatsapp": notify_whatsapp,
        "trigger_voice_call": trigger_voice_call,
        "recommended_sla_hours": recommended_sla_hours,
    })

    return 1, latest

def report_generator(anamoly_readings):
    readings_text = ", ".join([f"{k}: {v}" for k, v in anamoly_readings.items()])
    
    prompt = f"""
    An anomaly has been detected.

    Readings: {readings_text}

    Generate a short anomaly report with:
    - Summary
    - Severity
    - Causes
    - Recommended Actions
    """

    return "very serious issue"

def severaity_check(anamoly_readings):
    readings_text = ", ".join([f"{k}: {v}" for k, v in anamoly_readings.items()])

    prompt = f"""
    Readings: {readings_text}
    On a scale of 1–10, output ONLY a single number indicating severity.
    """

    ans = get_gemini_response(prompt)

    # Extract number safely
    try:
        return int(ans.strip().split()[0])
    except:
        return 5  # fallback

def master_agent(anamoly_readings):
    report = report_generator(anamoly_readings)
    print(report)
    print("Report sent to backend.")

def main():
    anomaly, readings = anamoly_dectection(60, 8, 7, 90)
    if anomaly:
        master_agent(readings)

main()
