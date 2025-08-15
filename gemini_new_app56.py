# I apologize in advance for the messy code and lack of comments

import streamlit as st
import os
import tempfile
import whisper
import json
import time
from datetime import timedelta, datetime
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO
import re
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import torch
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import matplotlib.pyplot as plt
import tempfile
import requests

def founder_form():
    # Constants
    TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    VC_TRAITS = ["Vision", "Grit", "Coachability", "Charisma", "Execution", "Emotional Stability"]
    LABEL_MEAN = np.array([0.56628148, 0.52273139, 0.47614642, 0.54818132, 0.52028646])
    LABEL_STD = np.array([0.14697756, 0.1552065, 0.15228453, 0.13637365, 0.15353347])
    HARMONIC_API_KEY = os.getenv("HARMONIC_API_KEY")

    # def enrich_company(company_identifier: dict, api_key: str):
    #     """
    #     Call Harmonic API to enrich company info.

    #     company_identifier: dict with exactly one key like {"website_domain": "harmonic.ai"}
    #     api_key: your Harmonic API key

    #     Returns: dict of company info or {"status":"pending"} or {"status":"error"}
    #     """
    #     url = "https://api.harmonic.ai/companies"
    #     headers = {
    #         "Content-Type": "application/json",
    #         "accept": "application/json",
    #         "apikey": api_key
    #     }

    #     if len(company_identifier) != 1:
    #         raise ValueError("Provide exactly one company identifier.")

    #     try:
    #         response = requests.post(url, headers=headers, json=company_identifier, timeout=15)
    #         if response.status_code == 404:
    #             # Enrichment pending; Harmonic recommends retrying later
    #             return {"status": "pending", "message": response.json()}
    #         response.raise_for_status()
    #         return response.json()
    #     except requests.RequestException as e:
    #         return {"status": "error", "message": str(e)}

    def query_harmonic(domain: str, api_key: str):
        url = "https://api.harmonic.ai/companies"
        headers = {
            "accept": "application/json",
            "apikey": api_key
        }
        params = {
            "website_domain": domain
        }

        response = requests.post(url, headers=headers, params=params)
        print(response)
        response.raise_for_status()
        data = response.json()
        # print("Harmonic JSON:", json.dumps(data, indent=2))
        return data

    # def parse_company_info(data):
    #     if data.get("status") == "pending":
    #         return {"message": "Enrichment in progress, try again later."}
    #     if data.get("status") == "error":
    #         return {"error": data.get("message")}

    #     info = {
    #         "Name": data.get("name"),
    #         "Description": data.get("description") or data.get("short_description") or data.get("external_description"),
    #         "Stage": data.get("stage"),
    #         "Headcount": data.get("headcount"),
    #         "Location": data.get("location", {}).get("display"),
    #         "Website": data.get("website", {}).get("url"),
    #         "Socials": ", ".join(f"{k}: {v}" for k, v in (data.get("socials") or {}).items() if v),
    #         "Tags": ", ".join(data.get("tags", [])),
    #         "Founders": ", ".join([p.get("name") for p in data.get("people", []) if p.get("role") == "Founder"]),
    #         "Funding": data.get("funding", {}).get("display"),
    #     }
    #     # Remove empty fields
    #     print(info)
    #     return {k: v for k, v in info.items() if v}
    #     # Only remove None or empty strings

    def parse_company_info(data):
        print("Parsing")
        #status = data.get("status")
        #if status == "pending":
        #    return {"message": "Enrichment in progress, try again later."}
        #if status == "error":
        #    return {"error": data.get("message")}
        
        # print("location and website")
        # location = data.get('location') or {}
        # city = location.get('city', '')
        # region = location.get('region', '')
        # country = location.get('country', '')
        # website_info = data.get('website') or {}
        
        print("Now info")
        info = {
            "Name": data.get("name"),
            "Description": data.get("description"),
            "Stage": data.get("stage"),
            "Headcount": data.get("headcount"),
            "Ownership": data.get("ownership_status")
            # "Location": f"{city} {region} {country}".strip(),
            # "URL": website_info.get('url') or "",
            # "domain": website_info.get('domain') or ""
        }
        
        print("done info")
        cleaned_info = {k: v for k, v in info.items() if v is not None and v != ""}
        print(cleaned_info)
        return cleaned_info

    # Example usage with retry logic
    def get_harmonic_insights(company_identifier, api_key, retries=3, wait_seconds=10):
        for attempt in range(retries):
            data = query_harmonic(company_identifier, api_key)
            print(data.get("status"))
            if data.get("status") == "pending":
                if attempt < retries - 1:
                    time.sleep(wait_seconds)  # wait before retrying
                    continue
                else:
                    return {"error": "Enrichment still pending after retries."}
            elif data.get("status") == "error":
                return {"error": data.get("message")}
            else:
                return parse_company_info(data)
        return {"error": "Unknown error in enrichment."}

    def normalize_field_value(value):
        if isinstance(value, list):
            formatted_items = []
            for item in value:
                if isinstance(item, dict):
                    # Recursively format dictionary items in a list
                    formatted_parts = []
                    for k, v in item.items():
                        # For nested dictionaries or lists, call normalize_field_value recursively
                        # Otherwise, just append key: value
                        if isinstance(v, (dict, list)):
                            formatted_parts.append(f"{k}: {normalize_field_value(v)}")
                        else:
                            formatted_parts.append(f"{k}: {str(v).strip()}")
                    formatted_items.append(", ".join(formatted_parts)) # Join key:value pairs with comma
                else:
                    # Handle simple list items (strings, numbers, etc.)
                    formatted_items.append(str(item).strip(' "\''))
            return "\n".join(formatted_items)
        elif isinstance(value, dict):
            # Format a single dictionary
            formatted_parts = []
            for k, v in value.items():
                # For nested dictionaries or lists, call normalize_field_value recursively
                # Otherwise, just append key: value
                if isinstance(v, (dict, list)):
                    formatted_parts.append(f"{k}: {normalize_field_value(v)}")
                else:
                    formatted_parts.append(f"{k}: {str(v).strip()}")
            return ", ".join(formatted_parts) # Join key:value pairs with comma
        elif value is None:
            return ""
        else:
            return str(value).strip()

    # Set initial page
    if "page" not in st.session_state:
        st.session_state.page = "form"

    # Streamlit UI
    # st.set_page_config(page_title="Y+ Founder Submission", layout="wide")
    # st.title("🚀 Founder Inquiry Submission")

    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    def run_analysis(deck, video, text_pitch):
            # Transcribe video if provided
        transcript = ""
        temp_video_name = None
        if video is not None:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_bytes = video.read()
            temp_video.write(video_bytes)
            temp_video.close()
            temp_video_name = temp_video.name
            whisper_model = whisper.load_model("base")
            transcript = whisper_model.transcribe(temp_video_name)["text"]
            st.session_state["video_transcript"] = transcript
        if text_pitch:
            transcript += "\n" + text_pitch

        st.session_state["transcript"] = transcript
        st.session_state["temp_video_name"] = temp_video_name
        st.session_state["text_pitch"] = text_pitch  # For use in PDF generation later
        st.session_state["deck_bytes"] = deck.read()  # For use in file upload later
        deck.seek(0)

        # Convert PDF to images
        all_images = convert_from_bytes(st.session_state["deck_bytes"], dpi=200)

        # Prepare prompt fields (for info only)
        basic_fields = [
            "Founder Name", "Email", "Phone", "Founder LinkedIn", "Company Name",
            "Company Website (Please only provide the domain; i.e., company.com)", "Customer Facing (B2B, B2C, B2B2C, or GovTech)", "Industry",
            "Stage (Angel, Pre-Seed, Seed, Series A, Growth Stage, or Pre-IPO)", "One-Line Pitch"
        ]
        extended_fields = [
            "Co-Founder 1 Name", "Co-Founder 1 LinkedIn", "Co-Founder 2 Name", "Co-Founder 2 LinkedIn",
            "List Other Co-founders' Name and LinkedIn if there are more than 3",
            "Co-founder Backstory (How did you and your co-founder(s) meet and decide to team up?)",
            "Team & Advisors (Bio/LinkedIn of key hires & advisors if any)",
            "Links to Demo or Prototype",
            "Problem Statement (What pain are you solving—and for whom?)",
            "Solution Overview (Brief description of product/service)",
            "Why You Care (What drives you to solve this problem? Why does it matter so much?)",
            "Target Market & TAM",
            "Traction Metrics (e.g. users, revenue, pilots, LOIs—anything quantifiable to date. Input N/A if you don't have any yet. We understand it takes time!)",
            "Business Model / Monetization (Subscription, Licensing, Transaction Fees, Advertisements, Others, or Haven't figured out yet but eager to learn!)",
            "If choose Others, explain your business model",
            "Funding to Date - How Much Have you Raised (It helps us to understand the funding history of your start-up. Input 0 if you have not raised any yet. We are excited to potentially be your first check too!)",
            "Stakeholders (Who did you raise from? If your answer to the last question is above 0)",
            "Runway/Burn Rate",
            "Amount Seeking (How much capital do you target to raise now)",
            "Proposed Post-Money Valuation (What is the valuation in your mind for this round?)",
            "Referral & How Heard About Us (We would like to thank them!)"
        ]

        transcript_context = transcript[:3000] if transcript else ""

        prompt = f"""
    You are analyzing a startup pitch deck composed of multiple slides.

    Your job is to extract the following fields as structured JSON from the **slide visuals** (sent as images). If a field is missing, use best-guess inference. If you still do not have an answer, explictly write "N/A" for the field.

    Do not include explanations, commentary, or citations. Return only **valid JSON** using double quotes for all strings. For the keys, use the exact field name including parenthetical statements.

    Fields to extract:
    {json.dumps(basic_fields + extended_fields, indent=2)}

    Partial transcript for context:
    {transcript_context}
    """


        try:
            response = model.generate_content([prompt] + all_images)

            # Extract JSON substring from Gemini response
            match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
            if not match:
                st.warning("⚠️ Gemini response could not be parsed as JSON.")
                st.stop()

            parsed_json = json.loads(match.group(0))

            # Define all expected fields including both combined and nested versions
            EXPECTED_FIELDS = [
                "Founder Name", "Email", "Phone", "Founder LinkedIn", "Company Name", "Company Website (Please only provide the domain; i.e., company.com)",
                "Customer Facing (B2B, B2C, B2B2C, or GovTech)", "Industry",
                "Stage (Angel, Pre-Seed, Seed, Series A, Growth Stage, or Pre-IPO)", "One-Line Pitch",
                "Co-Founder 1 Name", "Co-Founder 1 LinkedIn", "Co-Founder 2 Name", "Co-Founder 2 LinkedIn",
                "List Other Co-founders' Name and LinkedIn if there are more than 3",
                "Co-founder Backstory (How did you and your co-founder(s) meet and decide to team up?)",
                "Team & Advisors (Bio/LinkedIn of key hires & advisors if any)",
                "Links to Demo or Prototype",
                "Problem Statement (What pain are you solving—and for whom?)",
                "Solution Overview (Brief description of product/service)",
                "Why You Care (What drives you to solve this problem? Why does it matter so much?)",
                "Target Market & TAM",
                "Traction Metrics (e.g. users, revenue, pilots, LOIs—anything quantifiable to date. Input N/A if you don't have any yet. We understand it takes time!)",
                "Business Model / Monetization (Subscription, Licensing, Transaction Fees, Advertisements, Others, or Haven't figured out yet but eager to learn!)",
                "If choose Others, explain your business model",
                "Funding to Date - How Much Have you Raised (It helps us to understand the funding history of your start-up. Input 0 if you have not raised any yet. We are excited to potentially be your first check too!)",
                "Stakeholders (Who did you raise from? If your answer to the last question is above 0)",
                "Runway/Burn Rate",
                "Amount Seeking (How much capital do you target to raise now)",
                "Proposed Post-Money Valuation (What is the valuation in your mind for this round?)",
                "Referral & How Heard About Us (We would like to thank them!)"
            ]

            # 1. Normalize values directly from the parsed JSON using the predefined list of fields.
            #    This avoids creating new fields for sub-keys.
            normalized_output = {
                field: normalize_field_value(parsed_json.get(field, ""))
                for field in EXPECTED_FIELDS
            }
            
            st.session_state["extracted_data"] = normalized_output
            st.session_state["show_review_form"] = True

            # # 2. Show extracted fields clearly to user for review.
            # #    This will now strictly follow the order and content of EXPECTED_FIELDS.
            # st.subheader("✅ Review Auto-Filled Submission")
            # for key, value in normalized_output.items():
            #     st.markdown(f"**{key}**: {value if value else '*Not provided*'}")

            # # Optional: Raw JSON debug view
            # with st.expander("See Raw Gemini JSON"):
            #     st.json(parsed_json)

        except Exception as e:
            st.error(f"❌ Gemini analysis failed: {e}")
            st.stop()

    # Page: Submission Form
    if st.session_state.page == "form":
        st.title("🚀 Y+ Founder Inquiry Submission")

        deck = st.file_uploader("📁 Upload your pitch deck (PDF).", type="pdf")
        video = st.file_uploader("🎙️ Optional: Short elevator pitch recording (MP4). Please ensure only one person is present.", type="mp4")
        text_pitch = st.text_area("📝 Optional: Paste your elevator pitch (text).")
        
        extract_disabled = deck is None

        if st.button("Extract information", disabled=extract_disabled):
            st.session_state.deck = deck
            st.session_state.video = video
            st.session_state.text_pitch = text_pitch
            st.session_state.page = "loading"
            st.rerun()

    # Page: Loading Page
    elif st.session_state.page == "loading":
        st.markdown("""
            <style>
            .loading-screen {
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                background-color: #0f1116;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                font-weight: bold;
                color: white;
                font-family: 'Source Sans', monospace;
            }

            .wavy span {
                display: inline-block;
                animation: wave 2.0s infinite ease-in-out;
            }

            /* Delay each letter’s animation for a wave effect */
            .wavy span:nth-child(1) { animation-delay: 0s; }
            .wavy span:nth-child(2) { animation-delay: 0.1s; }
            .wavy span:nth-child(3) { animation-delay: 0.2s; }
            .wavy span:nth-child(4) { animation-delay: 0.3s; }
            .wavy span:nth-child(5) { animation-delay: 0.4s; }
            .wavy span:nth-child(6) { animation-delay: 0.5s; }
            .wavy span:nth-child(7) { animation-delay: 0.6s; }
            .wavy span:nth-child(8) { animation-delay: 0.7s; }
            .wavy span:nth-child(9) { animation-delay: 0.8s; }
            .wavy span:nth-child(10) { animation-delay: 0.9s; }
            .wavy span:nth-child(11) { animation-delay: 1.0s; }
            .wavy span:nth-child(12) { animation-delay: 1.1s; }
            .wavy span:nth-child(13) { animation-delay: 1.2s; }
            .wavy span:nth-child(14) { animation-delay: 1.3s; }
            .wavy span:nth-child(15) { animation-delay: 1.4s; }

            @keyframes wave {
                0%, 100% {
                    transform: translateY(0);
                }
                50% {
                    transform: translateY(-10px);
                }
            }
                    
            /* Fade-in animation for the note */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
       
            .loading-note {
                font-size: 18px;
                font-weight: normal;
                margin-top: 20px;
                color: white;
                opacity: 0;                    /* start invisible */
                animation: fadeIn 2s forwards; /* fade in over 2 seconds */
                animation-delay: 1s;           /* wait 1s before starting */
            }
            </style>

            <div class="loading-screen">
                <div class="wavy">
                    <span>⏳</span>
                    <span>&nbsp;</span>
                    <span>E</span><span>x</span><span>t</span><span>r</span><span>a</span><span>c</span><span>t</span><span>i</span><span>n</span><span>g</span><span>.</span><span>.</span><span>.</span>
                </div>
                <div class="loading-note">This can take a minute.</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("*This can take a minute.*")

        # Run analysis only once
        if "analysis_done" not in st.session_state:
            run_analysis(
                st.session_state.deck,
                st.session_state.video,
                st.session_state.text_pitch,
            )
            st.session_state.analysis_done = True
            st.session_state.page = "review"
            st.rerun()

    # -- PAGE: Review Auto-Filled Submission --
    elif st.session_state.page == "review":
        st.subheader("🧾 Review Auto-Filled Submission")

        # Initialize fields only once
        if "fields" not in st.session_state:
            st.session_state["fields"] = {
                k: normalize_field_value(st.session_state["extracted_data"].get(k, ""))
                for k in st.session_state["extracted_data"]
            }

        st.markdown("##### Please review the extracted fields below:")

        for field_key, original_value in st.session_state["fields"].items():
            widget_key = f"input_{field_key}"

            # Initialize editable field state if not present
            if widget_key not in st.session_state:
                st.session_state[widget_key] = original_value

            # Show the editable text area directly (no form)
            user_input = st.text_area(field_key, key=widget_key)

            # Show red flag only if value is still "N/A"
            if user_input.strip().upper() == "N/A":
                st.markdown(
                    f"<span style='color: red; font-weight: bold;'>⚠️ Above field flagged as N/A. Please ensure this is the desired response.</span>",
                    unsafe_allow_html=True
                )

        # Display buttons side-by-side on the left
        with st.container():
            col1, col2, _ = st.columns([2, 2, 10])  # two small columns and one wide spacer

            with col1:
                if st.button("⬅️ Back to Upload"):
                    st.session_state.clear()
                    st.rerun()

            with col2:
                if st.button("✅ Submit Form"):
                    # Save updated fields
                    for field_key in st.session_state["fields"]:
                        widget_key = f"input_{field_key}"
                        st.session_state["fields"][field_key] = st.session_state[widget_key]

                    st.session_state.page = "loading2"
                    st.rerun()

    # -- PAGE: Loading screen before final analysis --
    elif st.session_state.page == "loading2":
        st.markdown("""
            <style>
            .loading-screen {
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                background-color: #0f1116;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                font-weight: bold;
                color: white;
                font-family: 'Source Sans', monospace;
            }

            .wavy span {
                display: inline-block;
                animation: wave 1.9s infinite ease-in-out;
            }

            /* Delay each letter’s animation for a wave effect */
            .wavy span:nth-child(1) { animation-delay: 0s; }
            .wavy span:nth-child(2) { animation-delay: 0.1s; }
            .wavy span:nth-child(3) { animation-delay: 0.2s; }
            .wavy span:nth-child(4) { animation-delay: 0.3s; }
            .wavy span:nth-child(5) { animation-delay: 0.4s; }
            .wavy span:nth-child(6) { animation-delay: 0.5s; }
            .wavy span:nth-child(7) { animation-delay: 0.6s; }
            .wavy span:nth-child(8) { animation-delay: 0.7s; }
            .wavy span:nth-child(9) { animation-delay: 0.8s; }
            .wavy span:nth-child(10) { animation-delay: 0.9s; }
            .wavy span:nth-child(11) { animation-delay: 1.0s; }
            .wavy span:nth-child(12) { animation-delay: 1.1s; }
            .wavy span:nth-child(13) { animation-delay: 1.2s; }
            .wavy span:nth-child(14) { animation-delay: 1.3s; }

            @keyframes wave {
                0%, 100% {
                    transform: translateY(0);
                }
                50% {
                    transform: translateY(-10px);
                }
            }
                    
            /* Fade-in animation for the note */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
       
            .loading-note {
                font-size: 18px;
                font-weight: normal;
                margin-top: 20px;
                color: white;
                opacity: 0;                    /* start invisible */
                animation: fadeIn 2s forwards; /* fade in over 2 seconds */
                animation-delay: 1s;           /* wait 1s before starting */
            }
            </style>

            <div class="loading-screen">
                <div class="wavy">
                    <span>⏳</span>
                    <span>&nbsp;</span>
                    <span>U</span><span>p</span><span>l</span><span>o</span><span>a</span><span>d</span><span>i</span><span>n</span><span>g</span><span>.</span><span>.</span><span>.</span>
                </div>
                <div class="loading-note">This can take a minute.</div>
            </div>
            """, unsafe_allow_html=True)

        # Only run this once on load
        if "report_uploaded" not in st.session_state:
            try:
                # Define the scopes for Google Sheets and Drive access
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

                # Load the service account info from Streamlit secrets
                # The service account JSON content should be stored as a string in st.secrets
                try:
                    service_account_info = json.loads(st.secrets["google_sheets"]["service_account"])
                    creds = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=scope
                    )
                    client = gspread.authorize(creds)

                    # Open the Google Sheet and append the row
                    sheet = client.open_by_key("1g5g7wZoh7a1xcJh-G8oL7F3fvA3z_gfpWPHNTKXcu9k")
                    row_main = [st.session_state["fields"].get(k, "") for k in st.session_state["fields"]]
                    sheet.worksheet("Submissions").append_row(row_main)

                except KeyError:
                    st.error("❌ Google Sheets service account credentials not found in Streamlit secrets. Please configure `st.secrets`.")
                except Exception as e:
                    st.error(f"❌ Failed to connect to Google Sheets: {e}")

                temp_video_name = st.session_state.get("temp_video_name", None)

                if temp_video_name is not None:
                    # Run personality model (black-box backend)
                    from founder_personality_analysis import run_personality_analysis
                    transcript = st.session_state.get("video_transcript", "")
                    video_path = temp_video_name
                    analysis = run_personality_analysis(video_path, transcript, st.session_state["fields"].get("Company Name", ""))

                    def get_big_five_flags(preds, mean, std):
                        z_scores = (preds - mean) / std
                        flags = []
                        for trait, z in zip(TRAITS, z_scores):
                            if abs(z) >= 2:
                                flags.append(f"Outlier in {trait}")
                            elif 1 <= abs(z) < 2:
                                flags.append(f"Less Common in {trait}")
                        return "; ".join(flags) if flags else ""

                    company_name = st.session_state["fields"].get("Company Name", "")
                    audio_summary = analysis.get("audio_summary", "")
                    emotions_list = analysis.get("emotions", [])
                    big_five_preds = np.array(analysis.get("big_five", []))
                    big_five_flags = get_big_five_flags(big_five_preds, LABEL_MEAN, LABEL_STD)
                    vc_traits = analysis.get("vc_traits", {})
                    qualitative_report = analysis.get("qualitative", "")
                    timestamp_iso = datetime.now().isoformat()

                    temp_face_image_paths = []
                    if "face_images" in analysis:
                        for img in analysis["face_images"]:
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            img.save(temp_file.name)
                            temp_face_image_paths.append(temp_file.name)
                    else:
                        temp_face_image_paths = None

                    st.session_state["founder_analysis"] = {
                        "video_transcript": st.session_state.get("video_transcript", ""),
                        "audio_summary": audio_summary,
                        "emotions": emotions_list,
                        "big_five": big_five_preds,
                        "vc_traits": vc_traits,
                        "qualitative": qualitative_report,
                        "video_frames": temp_face_image_paths
                    }

                    vc_scores = [vc_traits.get(trait, (None, ""))[0] for trait in VC_TRAITS]
                    vc_reasons = [vc_traits.get(trait, (None, ""))[1] for trait in VC_TRAITS]

                    row_personality = [
                        company_name, transcript, audio_summary, ", ".join(emotions_list),
                        f"{big_five_preds[0]:.2f}", f"{big_five_preds[1]:.2f}", f"{big_five_preds[2]:.2f}",
                        f"{big_five_preds[3]:.2f}", f"{big_five_preds[4]:.2f}", big_five_flags,
                        *[f"{s:.2f}" if s is not None else "" for s in vc_scores],
                        *vc_reasons, qualitative_report, timestamp_iso
                    ]

                    sheet.worksheet("Founder Personality Analysis").append_row(row_personality)
                

                def sanitize_filename(name):
                    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip()
                
                def make_radar_chart(title, labels, values, color, filename):
                    # print(f"make_radar_chart called with {len(labels)} labels and {len(values)} values")
                    # print(f"Labels: {labels}")
                    # print(f"Values before extension: {values}")

                    num_vars = len(labels)
                    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                    values = list(values)          # convert numpy array to list first
                    values = values + values[:1]   # concatenate by creating a new list
                    angles = angles + angles[:1]

                    # print(f"Values after extension: {values}")
                    # print(f"Angles after extension: {angles}")

                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                    ax.set_ylim(0, 1)  # set radius limits from 0 to 1

                    ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
                    ax.set_yticks(ticks)  # set where the ticks are
                    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)  # set labels

                    ax.plot(angles, values, color=color, linewidth=2)
                    ax.fill(angles, values, color=color, alpha=0.25)
                    ax.set_yticklabels([])
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(labels, fontsize=8)
                    ax.set_title(title, size=12, y=1.08)
                    ax.grid(True)

                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches='tight')
                    plt.close()

                def create_full_pdf(fields, analysis, text_pitch=None, has_video=False, harmonic_data=None, output_path_suffix="submission_report"):
                    pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))

                    company_name = fields.get("Company Name", "unknown_company")
                    safe_company_name = sanitize_filename(company_name)
                    output_path = f"{safe_company_name}_{output_path_suffix}.pdf"

                    doc = SimpleDocTemplate(output_path, pagesize=letter,
                                            rightMargin=50, leftMargin=50,
                                            topMargin=50, bottomMargin=50)

                    styles = getSampleStyleSheet()
                    normal_style = ParagraphStyle(
                        name='NormalDejaVu',
                        parent=styles['Normal'],
                        fontName='DejaVu',
                        fontSize=12,
                        leading=15,
                    )
                    title_style = ParagraphStyle(
                        name='Title',
                        parent=styles['Heading2'],
                        fontName='DejaVu',
                        fontSize=14,
                        leading=18,
                        spaceAfter=12,
                    )

                    story = []
                    temp_images = []

                    def add_section_title(title):
                        story.append(Paragraph(f"<b>{title}</b>", title_style))
                        story.append(Spacer(1, 12))

                    def add_label_content(label, content):
                        if not content:
                            return
                        story.append(Paragraph(f"<b>{label}:</b>", normal_style))
                        content = str(content).replace("\n", "<br/>")
                        story.append(Paragraph(content, normal_style))
                        story.append(Spacer(1, 10))

                    # Section 1: Founder Submission
                    #add_section_title("📋 Founder Submission")
                    add_section_title("-- Founder Submission --")
                    for key, value in fields.items():
                        add_label_content(key, value)

                    # Section 1.5: Optional Text Pitch
                    if text_pitch:
                        #add_section_title("💬 Text Elevator Pitch")
                        add_section_title("-- Text Elevator Pitch --")
                        add_label_content("Pitch", text_pitch)

                    # Only include analysis section if video was uploaded
                    if has_video:
                        story.append(PageBreak())
                        #add_section_title("🧠 Founder Personality Analysis")
                        add_section_title("-- Founder Personality Analysis --")

                        add_label_content("Video Transcript", analysis.get("video_transcript", ""))
                        add_label_content("Audio Summary", analysis.get("audio_summary", ""))
                        video_frames = analysis.get("video_frames", "")

                        if len(video_frames) > 0:
                            story.append(PageBreak())
                            # add_section_title("🎥 Sample Video Frames Used for Emotion Analysis")

                            # Used to be 1.5, 1.2
                            max_image_width = 0.8 * inch
                            max_image_height = 0.64 * inch  # maintain aspect ratio ~ 4:3
                            # Limit to first 16 frames
                            max_frames = 16
                            frame_images = [
                                RLImage(path, width=max_image_width, height=max_image_height)
                                for path in video_frames[:max_frames]
                            ]

                            # Arrange into rows of 8
                            row_length = 8
                            image_rows = [frame_images[i:i+row_length] for i in range(0, len(frame_images), row_length)]

                            # Ensure all rows have equal columns
                            for row in image_rows:
                                while len(row) < row_length:
                                    row.append("")  # Empty cell to maintain structure

                            table = Table(image_rows, hAlign='LEFT', colWidths=[max_image_width]*row_length)
                            table.setStyle(TableStyle([
                                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                                ("TOPPADDING", (0,0), (-1,-1), 6),
                                # Optional: Uncomment below to add borders
                                # ("BOX", (0,0), (-1,-1), 0.25, colors.grey),
                                # ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                            ]))

                            story.append(table)

                        add_label_content("Detected Emotions", ", ".join(analysis.get("emotions", [])))

                        big_five = analysis.get("big_five", [])
                        if len(big_five) > 0:
                            big_five_text = "<br/>".join([f"{trait}: {score:.2f}" for trait, score in zip(TRAITS, big_five)])
                            add_label_content("Big Five Scores", big_five_text)

                            z_scores = (np.array(big_five) - LABEL_MEAN) / LABEL_STD
                            outlier_lines = []
                            for trait, z in zip(TRAITS, z_scores):
                                if abs(z) >= 2:
                                    outlier_lines.append(f"Outlier in {trait}")
                                elif 1 <= abs(z) < 2:
                                    outlier_lines.append(f"Less Common in {trait}")
                            if outlier_lines:
                                add_label_content("Notable Personality Deviations", "<br/>".join(outlier_lines))

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                                make_radar_chart("Big Five Traits", TRAITS, big_five, 'blue', tmp_chart.name)
                                img = RLImage(tmp_chart.name, width=250, height=250)
                                story.append(KeepTogether([
                                    Paragraph("<b>Big Five Radar Chart</b>", normal_style),
                                    Spacer(1, 6), img, Spacer(1, 20)
                                ]))
                                temp_images.append(tmp_chart.name)

                        vc_traits = analysis.get("vc_traits", {})
                        if len(vc_traits) > 0:
                            vc_text = "<br/>".join([
                                f"{trait}: {score:.2f} — {reason}"
                                for trait, (score, reason) in vc_traits.items()
                            ])
                            add_label_content("VC Trait Scores", vc_text)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                                vc_trait_names = list(vc_traits.keys())
                                vc_trait_scores = [vc_traits[t][0] for t in vc_trait_names]
                                make_radar_chart("VC-Relevant Founder Traits", vc_trait_names, vc_trait_scores, 'green', tmp_chart.name)
                                img = RLImage(tmp_chart.name, width=250, height=250)
                                story.append(KeepTogether([
                                    Paragraph("<b>VC Trait Radar Chart</b>", normal_style),
                                    Spacer(1, 6), img, Spacer(1, 20)
                                ]))
                                temp_images.append(tmp_chart.name)

                        add_label_content("Qualitative Summary", analysis.get("qualitative", ""))

                    add_section_title("-- Harmonic Insights --")
                    for label, content in harmonic_data.items():
                        add_label_content(label, content)

                    doc.build(story)

                    for path in temp_images:
                        os.unlink(path)

                    return output_path

                def create_full_pdf_in_memory(fields, analysis, text_pitch=None, has_video=False, harmonic_data=None):
                    #pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf')) # You need to handle font embedding without a file path, which can be complex. For a simple solution, stick to standard fonts or pre-load them.
                    # from reportlab.lib.colors import colors # If you use colors
                    
                    # ✅ MODIFICATION: Use BytesIO instead of a filepath
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter,
                                                    rightMargin=50, leftMargin=50,
                                                    topMargin=50, bottomMargin=50)

                    # ... (Your existing code to build the 'story' list remains the same)
                    # The image creation part `make_radar_chart` still writes to a temp file, which is fine
                    # as long as you clean them up immediately after adding them to the PDF.
                    # The logic below this point for story list creation does not change.
                    
                    styles = getSampleStyleSheet()
                    normal_style = ParagraphStyle(
                        name='Normal', # Using standard font to avoid issues with DejaVu
                        parent=styles['Normal'],
                        fontSize=12,
                        leading=15,
                    )
                    title_style = ParagraphStyle(
                        name='Title',
                        parent=styles['Heading2'],
                        fontSize=14,
                        leading=18,
                        spaceAfter=12,
                    )

                    story = []
                    temp_images = []

                    def add_section_title(title):
                        story.append(Paragraph(f"<b>{title}</b>", title_style))
                        story.append(Spacer(1, 12))

                    def add_label_content(label, content):
                        if not content:
                            return
                        story.append(Paragraph(f"<b>{label}:</b>", normal_style))
                        content = str(content).replace("\n", "<br/>")
                        story.append(Paragraph(content, normal_style))
                        story.append(Spacer(1, 10))

                    # Section 1: Founder Submission
                    add_section_title("-- Founder Submission --")
                    for key, value in fields.items():
                        add_label_content(key, value)

                    # Section 1.5: Optional Text Pitch
                    if text_pitch:
                        add_section_title("-- Text Elevator Pitch --")
                        add_label_content("Pitch", text_pitch)

                    # Only include analysis section if video was uploaded
                    if has_video:
                        story.append(PageBreak())
                        add_section_title("-- Founder Personality Analysis --")

                        add_label_content("Video Transcript", analysis.get("video_transcript", ""))
                        add_label_content("Audio Summary", analysis.get("audio_summary", ""))
                        video_frames = analysis.get("video_frames", "")

                        if len(video_frames) > 0:
                            story.append(PageBreak())
                            max_image_width = 0.8 * inch
                            max_image_height = 0.64 * inch
                            max_frames = 16
                            frame_images = [
                                RLImage(path, width=max_image_width, height=max_image_height)
                                for path in video_frames[:max_frames]
                            ]
                            
                            row_length = 8
                            image_rows = [frame_images[i:i+row_length] for i in range(0, len(frame_images), row_length)]
                            for row in image_rows:
                                while len(row) < row_length:
                                    row.append("")
                            
                            from reportlab.lib import colors
                            table = Table(image_rows, hAlign='LEFT', colWidths=[max_image_width]*row_length)
                            table.setStyle(TableStyle([
                                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                                ("TOPPADDING", (0,0), (-1,-1), 6),
                            ]))

                            story.append(table)

                        add_label_content("Detected Emotions", ", ".join(analysis.get("emotions", [])))

                        big_five = analysis.get("big_five", [])
                        if len(big_five) > 0:
                            big_five_text = "<br/>".join([f"{trait}: {score:.2f}" for trait, score in zip(TRAITS, big_five)])
                            add_label_content("Big Five Scores", big_five_text)

                            z_scores = (np.array(big_five) - LABEL_MEAN) / LABEL_STD
                            outlier_lines = []
                            for trait, z in zip(TRAITS, z_scores):
                                if abs(z) >= 2:
                                    outlier_lines.append(f"Outlier in {trait}")
                                elif 1 <= abs(z) < 2:
                                    outlier_lines.append(f"Less Common in {trait}")
                            if outlier_lines:
                                add_label_content("Notable Personality Deviations", "<br/>".join(outlier_lines))

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                                make_radar_chart("Big Five Traits", TRAITS, big_five, 'blue', tmp_chart.name)
                                img = RLImage(tmp_chart.name, width=250, height=250)
                                story.append(KeepTogether([
                                    Paragraph("<b>Big Five Radar Chart</b>", normal_style),
                                    Spacer(1, 6), img, Spacer(1, 20)
                                ]))
                                temp_images.append(tmp_chart.name)

                        vc_traits = analysis.get("vc_traits", {})
                        if len(vc_traits) > 0:
                            vc_text = "<br/>".join([
                                f"{trait}: {score:.2f} — {reason}"
                                for trait, (score, reason) in vc_traits.items()
                            ])
                            add_label_content("VC Trait Scores", vc_text)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                                vc_trait_names = list(vc_traits.keys())
                                vc_trait_scores = [vc_traits[t][0] for t in vc_trait_names]
                                make_radar_chart("VC-Relevant Founder Traits", vc_trait_names, vc_trait_scores, 'green', tmp_chart.name)
                                img = RLImage(tmp_chart.name, width=250, height=250)
                                story.append(KeepTogether([
                                    Paragraph("<b>VC Trait Radar Chart</b>", normal_style),
                                    Spacer(1, 6), img, Spacer(1, 20)
                                ]))
                                temp_images.append(tmp_chart.name)

                        add_label_content("Qualitative Summary", analysis.get("qualitative", ""))

                    add_section_title("-- Harmonic Insights --")
                    for label, content in harmonic_data.items():
                        add_label_content(label, content)

                    doc.build(story)

                    for path in temp_images:
                        os.unlink(path)

                    # ✅ MODIFICATION: Move the buffer's cursor to the beginning
                    buffer.seek(0)
                    return buffer
                
                def upload_in_memory_file_to_drive(
                    file_stream,
                    filename,
                    parent_folder_id,
                    mimetype,
                    service_account_info_string=st.secrets["google_sheets"]["service_account"]
                ):
                    if service_account_info_string is None:
                        raise ValueError("You must provide the service account JSON string.")

                    # Load the service account info from the string
                    service_account_info = json.loads(service_account_info_string)

                    creds = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/drive"]
                    )

                    service = build("drive", "v3", credentials=creds)

                    file_metadata = {
                        "name": filename,
                        "parents": [parent_folder_id]
                    }

                    media = MediaIoBaseUpload(file_stream, mimetype=mimetype, resumable=True)

                    uploaded_file = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True
                    ).execute()

                    return uploaded_file.get("id")

                print("Starting Harmonic enrichment")
                api_key = HARMONIC_API_KEY
                company_name = st.session_state["fields"].get("Company Name", "Unknown_Company")
                print(company_name)
                domain = st.session_state["fields"].get("Company Website (Please only provide the domain; i.e., company.com)", "Unknown_Domain")
                print(domain)
                identifier = domain

                harmonic_data = get_harmonic_insights(identifier, api_key)

                if "error" in harmonic_data:
                    print("Error getting Harmonic data:", harmonic_data["error"])
                else:
                    print("Harmonic company insights:")
                    for k, v in harmonic_data.items():
                        print(f"{k}: {v}")
                # Pass harmonic_data to your PDF generation to add a "Harmonic Insights" section.
                print("Harmonic enrichment done", harmonic_data)

                # Generate the report
                print("Creating PDF")
                pdf_stream = create_full_pdf_in_memory(
                    fields=st.session_state.get("fields", {}),
                    analysis=st.session_state.get("founder_analysis") or {},
                    text_pitch=st.session_state.get("text_pitch", None),
                    has_video=bool(st.session_state.get("temp_video_name")),
                    harmonic_data=harmonic_data
                )
                print("Done creating PDF")
                
                def upload_file_to_drive(
                    filepath,
                    parent_folder_id,
                    desired_name=None,
                    service_account_info_string=st.secrets["google_sheets"]["service_account"]
                ):
                    if service_account_info_string is None:
                        raise ValueError("You must provide the service account JSON string.")

                    # Load credentials from string
                    service_account_info = json.loads(service_account_info_string)
                    creds = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/drive"]
                    )

                    # Build Drive service
                    service = build("drive", "v3", credentials=creds)

                    # Determine filename
                    file_name = desired_name if desired_name else os.path.basename(filepath)

                    # Set metadata
                    file_metadata = {
                        "name": file_name,
                        "parents": [parent_folder_id]
                    }

                    # Guess MIME type
                    ext = os.path.splitext(filepath)[1].lower()
                    if ext == ".pdf":
                        mimetype = "application/pdf"
                    elif ext in [".mp4", ".mov", ".avi"]:
                        mimetype = "video/mp4"
                    else:
                        mimetype = "application/octet-stream"

                    # Prepare media
                    media = MediaFileUpload(filepath, mimetype=mimetype)

                    # Upload
                    uploaded_file = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True
                    ).execute()

                    return uploaded_file.get("id")
                
                def create_drive_folder(
                    folder_name,
                    parent_id,
                    service_account_info_string=st.secrets["google_sheets"]["service_account"]
                ):
                    if service_account_info_string is None:
                        raise ValueError("You must provide the service account JSON string.")

                    # Load credentials from string
                    service_account_info = json.loads(service_account_info_string)
                    creds = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/drive"]
                    )

                    # Build Drive service
                    service = build("drive", "v3", credentials=creds)

                    # Folder metadata
                    folder_metadata = {
                        "name": folder_name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [parent_id]
                    }

                    # Create folder
                    folder = service.files().create(
                        body=folder_metadata,
                        fields="id",
                        supportsAllDrives=True
                    ).execute()

                    return folder.get("id")

                # Parent folder in your Google Drive
                PARENT_FOLDER_ID = "1evRwL2yafIdMfyXwuqcHz44YNnjN8x3Q"
                company_name = st.session_state["fields"].get("Company Name", "Unknown_Company")
                safe_company_name = sanitize_filename(company_name)

                # Create subfolder with company name
                print("Creating folder")
                company_folder_id = create_drive_folder(company_name, PARENT_FOLDER_ID)

                # Upload files to this folder
                print("Uploading PDF")
                report_filename = f"{safe_company_name}_submission_report.pdf"
                uploaded_report_id = upload_in_memory_file_to_drive(pdf_stream, report_filename, company_folder_id, "application/pdf")
                
                print("Uploading deck")
                deck_name = f"{safe_company_name}_PitchDeck.pdf"
                deck_bytes = st.session_state.get("deck_bytes", None)
                if deck_bytes:
                    uploaded_deck_id = upload_in_memory_file_to_drive(
                        file_stream=BytesIO(deck_bytes),
                        filename=deck_name,
                        parent_folder_id=company_folder_id,
                        mimetype="application/pdf"
                )
                    
                temp_video_name = st.session_state.get("temp_video_name", None)
                if temp_video_name and os.path.exists(temp_video_name):
                    video_name = f"{company_name}_ElevatorPitch.mp4"
                    upload_file_to_drive(temp_video_name, company_folder_id, desired_name=video_name)
                    os.remove(temp_video_name)

                st.session_state["uploaded_file_id"] = uploaded_report_id
                st.session_state["report_uploaded"] = True
            except Exception as e:
                st.error(f"❌ Failed during uploading: {e}")
                st.stop()

            # Now navigate to the final page with success message
            st.session_state.page = "submitted"
            st.rerun()

    # -- PAGE: Final success confirmation --
    elif st.session_state.page == "submitted":
        uploaded_file_id = st.session_state.get("uploaded_file_id", "unknown")
        st.success(f"✅ Submission successfully uploaded")

        if "founder_analysis" in st.session_state:
            for path in st.session_state["founder_analysis"].get("video_frames", []):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"Cleanup error: {e}")

        if st.button("⬅️ Back to Start"):
            # Reset everything
            st.session_state.clear()
            st.rerun()