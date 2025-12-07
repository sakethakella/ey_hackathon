import google.generativeai as genai

genai.configure(api_key="")

model = genai.GenerativeModel("gemini-2.0-flash-lite")

def get_gemini_response(prompt):
    """Fresh call (no history) to save tokens."""
    response = model.generate_content(prompt)
    return response.text.strip()

def anamoly_dectection(temperature, pressure, humidity, tire_pressure):
    if (temperature > 60 or pressure > 2 or humidity > 60 or tire_pressure > 3):
        return 1, {
            'temp': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'tire_pressure': tire_pressure
        }
    return 0, {}

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
