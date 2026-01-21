import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os

# Set page configuration
st.set_page_config(page_title="Competitor Content Gap Analyzer", layout="wide")

# --- CSS / AESTHETICS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #fafafa;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        background-color: #262730;
        color: #fafafa;
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
    .report-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure your LLM provider settings here.")

provider = st.sidebar.selectbox("Select Provider", ["Groq", "OpenRouter", "Other"])

# Set defaults based on provider
if provider == "Groq":
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "llama3-8b-8192"
elif provider == "OpenRouter":
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "meta-llama/llama-3-8b-instruct"
else:
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-3.5-turbo"

api_key = st.sidebar.text_input("API Key", type="password", help=f"Enter your {provider} API Key")
base_url = st.sidebar.text_input("Base URL", value=default_base_url)
model_name = st.sidebar.text_input("Model Name", value=default_model)

# --- FUNCTIONS ---

def scrape_content(url):
    """Scrapes the main text content from a given URL."""
    if not url:
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        return text[:15000] # Truncate to avoid context limit issues initially
        
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return None

def analyze_gap(keyword, own_content, comp_contents, client, model):
    """Generates the content gap analysis using the LLM."""
    
    competitor_text = ""
    for i, content in enumerate(comp_contents):
        if content:
            competitor_text += f"\n--- COMPETITOR {i+1} CONTENT ---\n{content}\n"

    prompt = f"""
    You are a senior SEO content strategist performing a competitor-based content gap analysis. You must rely only on the content provided to you and not on external knowledge, assumptions, or rankings. The purpose of this analysis is to identify what is missing, underdeveloped, or misaligned in the target page compared to competitor pages that already perform well for the same topic.

    The primary keyword provided is used only to understand topic intent and context. It must not be treated as a keyword-matching requirement. Do not evaluate keyword density or exact-match usage. Focus instead on topical coverage, intent satisfaction, entities, and structural completeness.

    You are given two datasets. The first dataset represents combined content from top competitor pages. These pages define the semantic scope, intent coverage, and depth expectations for this topic. The second dataset represents the content of the target landing page.

    Your task is to analyze differences between these two datasets and identify meaningful gaps. A gap exists when competitors consistently cover a subtopic, concept, intent, entity, or section that is missing or significantly weaker in the target page. Do not invent gaps that are not supported by competitor evidence.

    You must group gaps by intent type where applicable, such as informational, commercial, or transactional intent, but only when the distinction is clear from competitor content. You must also identify structural gaps, such as missing sections, poor content sequencing, or insufficient depth compared to competitors.

    Your output must be practical and decision-oriented. Clearly explain what is missing, why it matters in the context of competitor coverage, and how it affects the completeness of the target page. When appropriate, suggest specific section ideas or headings that could address the gap, but do not write full content.

    Do not claim ranking outcomes, traffic impact, or guarantees. This analysis is strictly a semantic and structural comparison based on the provided content.

    ### PRIMARY KEYWORD
    {keyword}

    ### DATASET 1: COMPETITOR CONTENT (Combined)
    {competitor_text}

    ### DATASET 2: TARGET LANDING PAGE CONTENT
    {own_content}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior SEO strategist. STRICTLY grounded in the provided context. Do not invent facts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

def generate_outline(keyword, comp_contents, client, model):
    """Generates a master content outline based on competitor data."""
    
    competitor_text = ""
    for i, content in enumerate(comp_contents):
        if content:
            competitor_text += f"\n--- COMPETITOR {i+1} CONTENT ---\n{content}\n"

    prompt = f"""
    You are an expert SEO Content Architect.
    
    ### GOAL
    Create the **Ultimate Content Outline** for the keyword: "{keyword}".
    Your goal is to design a structure that is superior to all competitors by combining their best sections and filling the identifying gaps.

    ### COMPETITOR DATA
    {competitor_text}

    ### INSTRUCTIONS
    1.  **Analyze Structure**: Look at the H2/H3 structures of the competitors.
    2.  **Synthesize**: Create a "Master Outline" that covers the topic comprehensively.
    3.  **Format**:
        -   **H1**: Optimized Title
        -   **H2**: Main Sections (Order them logically for the user journey)
        -   **H3**: Sub-topics (What specific points to cover)
        -   **Notes**: Briefly explain *why* this section is included (e.g., "Competitor 1 & 2 cover this, crucial for intent").

    Output in Markdown.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior SEO strategist. Crreate an outline based ONLY on the provided competitor data. Do not hallucinate sections not relevant to the topic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating outline: {e}")
        return None

# --- MAIN UI ---

st.title("🏆 Competitor Analysis Tool")
st.markdown("Reverse engineer top-ranking content or plan your next blog post.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Mode", ["Content Gap Analysis", "New Blog Outline Generator"])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Details")
    target_keyword = st.text_input("Target Keyword", placeholder="e.g., best running shoes 2024")
    
    own_url = None
    if mode == "Content Gap Analysis":
        st.markdown("### Own Content")
        own_url = st.text_input("Your URL", placeholder="https://mysite.com/post")
    else:
        st.info("ℹ️ Generating outline based on competitors only.")
    
    st.markdown("### Competitors")
    comp_url_1 = st.text_input("Competitor URL 1", placeholder="https://competitor1.com")
    comp_url_2 = st.text_input("Competitor URL 2", placeholder="https://competitor2.com")
    comp_url_3 = st.text_input("Competitor URL 3", placeholder="https://competitor3.com")
    
    generate_outline_check = True
    if mode == "Content Gap Analysis":
        generate_outline_check = st.checkbox("Also Generate Optimized Outline", value=True)

    display_btn_text = "Analyze Content Gap" if mode == "Content Gap Analysis" else "Generate Outline"
    analyze_btn = st.button(display_btn_text)

with col2:
    if analyze_btn:
        if not api_key:
            st.warning("Please enter your API Key in the sidebar first.")
        elif not target_keyword:
            st.warning("Please enter a keyword.")
        elif mode == "Content Gap Analysis" and not own_url:
             st.warning("Please enter your URL for Gap Analysis.")
        else:
            with st.spinner("Scraping content..."):
                own_text = scrape_content(own_url) if own_url else None
                
                comp_texts = []
                # Store tuples of (url, text) to track which competitor is which
                comp_data = [] 
                
                if comp_url_1: 
                    t1 = scrape_content(comp_url_1)
                    comp_data.append(t1)
                    if t1: 
                        if len(t1) < 300:
                            st.warning(f"⚠️ Competitor 1 content is very short ({len(t1)} chars). It might be blocked or require Login. Check Debug Info below.")
                        comp_texts.append(t1)
                    else:
                        st.warning(f"⚠️ Could not scrape Competitor 1 ({comp_url_1}). Skipping.")
                    
                if comp_url_2: 
                    t2 = scrape_content(comp_url_2)
                    comp_data.append(t2)
                    if t2: 
                        if len(t2) < 300:
                            st.warning(f"⚠️ Competitor 2 content is very short ({len(t2)} chars). It might be blocked or require Login. Check Debug Info below.")
                        comp_texts.append(t2)
                    else:
                        st.warning(f"⚠️ Could not scrape Competitor 2 ({comp_url_2}). Skipping.")

                if comp_url_3: 
                    t3 = scrape_content(comp_url_3)
                    comp_data.append(t3)
                    if t3: 
                        if len(t3) < 300:
                            st.warning(f"⚠️ Competitor 3 content is very short ({len(t3)} chars). It might be blocked or require Login. Check Debug Info below.")
                        comp_texts.append(t3)
                    else:
                        st.warning(f"⚠️ Could not scrape Competitor 3 ({comp_url_3}). Skipping.")
                
            # --- DEBUGGING VIEW ---
            with st.expander("🔍 View Scraped Content (Debug Info)"):
                st.write(f"**My Content Length:** {len(own_text) if own_text else 0} chars")
                if own_text: st.code(own_text[:500] + "...", language="text")
                
                for i, txt in enumerate(comp_data):
                    st.write(f"**Competitor {i+1} Length:** {len(txt) if txt else 0} chars")
                    if txt:
                        st.code(txt[:500] + "...", language="text")
                    else:
                        st.error(f"Competitor {i+1} returned no text. (May be blocked or JS-rendered)")

            # Check if we have enough data (Gap Analysis needs own_text, Outline only needs competitors)
            valid_to_proceed = False
            if mode == "Content Gap Analysis":
                valid_to_proceed = own_text and comp_texts
            else:
                valid_to_proceed = bool(comp_texts)

            if valid_to_proceed:
                try:
                    client = OpenAI(api_key=api_key, base_url=base_url)
                    
                    # 1. Gap Analysis (Only if Mode is Gap Analysis)
                    if mode == "Content Gap Analysis":
                        with st.spinner("Analyzing Gaps with Llama-3..."):
                            analysis = analyze_gap(target_keyword, own_text, comp_texts, client, model_name)
                            
                            if analysis:
                                st.success("Analysis Complete!")
                                st.subheader("📊 Content Gap Report")
                                st.markdown(f'<div class="report-card">{analysis}</div>', unsafe_allow_html=True)
                    
                    # 2. Outline Generation (Always run for Generator mode, optional for Gap mode)
                    run_outline = True
                    if mode == "Content Gap Analysis" and not generate_outline_check:
                         run_outline = False
                    
                    if run_outline:
                        with st.spinner("Generating Master Outline..."):
                            outline = generate_outline(target_keyword, comp_texts, client, model_name)
                            if outline:
                                st.subheader("📝 Optimized Content Outline")
                                st.markdown(f'<div class="report-card">{outline}</div>', unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"Failed to initialize client or run analysis: {e}")
            else:
                st.error("Could not scrape enough content. Check URLs and Debug View.")

    else:
        st.info("👈 Enter your details and click Analyze to start.")
        st.markdown("""
        ### How it works:
        1.  **Enter Keyword**: The main term you want to rank for.
        2.  **Enter URLs**: Your page and up to 3 top competitors.
        3.  **Configure API**: Add your API key (Groq, OpenRouter, etc.) in the sidebar.
        4.  **Get Results**: The tool scrapes all pages and uses **Llama-3** to tell you exactly what you're missing.
        """)
