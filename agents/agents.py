import google.generativeai as genai

genai.configure(api_key="AIzaSyAdDBAagSL-AizkJy7nFhsoLVHPAgbDbaY") 
model = genai.GenerativeModel('gemini-2.0-flash-lite')
chat = model.start_chat(history=[])

def get_gemini_response(prompt):
    """Sends a prompt to Gemini and returns the response."""
    response = chat.send_message(prompt)
    return response.text

def anamoly_dectection(temperature,pressure,humidity,tire_pressure):
    if(temperature>60 or pressure>2 or humidity>60 or tire_pressure>3):
        return 1,{'temp':temperature,'humidity':humidity,'pressure':pressure,'tire_pressure':tire_pressure}
    else:
        return 0,{}

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
