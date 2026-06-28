import os
import re
import io
import time
import json
import hashlib
from pathlib import Path
from numbers import Real
from xml.sax.saxutils import escape

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env.local")

st.set_page_config(
    page_title="게임 상태 진단 AI",
    layout="wide",
)

EXPECTED_COLUMNS = [
    "일자",
    "접속유저", "신규유저", "복귀유저", "기존유저", "누적유저수",
    "최고동접", "평균동접", "플레이시간", "총매출", "개인매출",
    "PU", "PUR", "ARPPU"
]

NUMERIC_COLUMNS = [
    "접속유저", "신규유저", "복귀유저", "기존유저", "누적유저수",
    "최고동접", "평균동접", "플레이시간", "총매출", "개인매출",
    "PU", "PUR", "ARPPU"
]

THRESHOLDS = {
    "접속유저": {"warning": -5, "critical": -10},
    "신규유저": {"warning": -5, "critical": -10},
    "복귀유저": {"warning": -5, "critical": -10},
    "기존유저": {"warning": -3, "critical": -7},
    "평균동접": {"warning": -5, "critical": -10},
    "최고동접": {"warning": -5, "critical": -10},
    "플레이시간": {"warning": -5, "critical": -10},
    "총매출": {"warning": -5, "critical": -10},
    "개인매출": {"warning": -3, "critical": -7},
    "PU": {"warning": -5, "critical": -10},
    "PUR": {"warning": -3, "critical": -7},
    "ARPPU": {"warning": -3, "critical": -7},
}

AI_STATE_KEYS = [
    "analyst_result",
    "risk_result",
    "improvement_result",
    "insight_result",
]

CARD_METRICS = ["접속유저", "신규유저", "복귀유저", "총매출", "PUR", "ARPPU"]
HIGHLIGHT_METRICS = ["접속유저", "신규유저", "복귀유저", "총매출", "PU", "PUR", "ARPPU"]
CHART_METRICS = ["접속유저", "신규유저", "복귀유저", "총매출", "PU", "PUR", "ARPPU", "플레이시간"]


def inject_custom_css():
    st.markdown(
        """
<style>
:root {
    --box-border-color: rgba(120, 120, 120, 0.14);
    --box-border: 1px solid var(--box-border-color);
    --box-radius: 18px;
    --box-bg: rgba(255,255,255,0.02);
    --text-muted: rgba(127, 127, 127, 0.96);
    --danger-border: rgba(255, 99, 132, 0.28);
    --danger-bg: rgba(255, 99, 132, 0.045);
    --warning-border: rgba(255, 193, 7, 0.28);
    --warning-bg: rgba(255, 193, 7, 0.045);
    --normal-border: rgba(40, 167, 69, 0.22);
    --normal-bg: rgba(40, 167, 69, 0.035);
    --hover-shadow: 0 6px 14px rgba(0,0,0,0.04);
}

.block-container {
    max-width: 1420px;
    padding-top: 1.15rem;
    padding-bottom: 3rem;
}

.pretty-page-hero,
.pretty-card,
.pretty-simple-card,
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: var(--box-border) !important;
    border-radius: var(--box-radius) !important;
    background: var(--box-bg) !important;
    box-shadow: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 14px 16px 20px 16px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] > *:first-child {
    margin-top: 0 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] > *:last-child {
    margin-bottom: 0 !important;
}

.pretty-page-hero {
    padding: 18px 20px;
    margin-bottom: 0.7rem;
}

.pretty-page-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.35rem;
}

.pretty-page-desc {
    font-size: 0.98rem;
    color: var(--text-muted);
    margin-bottom: 0;
}

.pretty-section-title {
    font-size: 1.18rem;
    font-weight: 800;
    margin-bottom: 0.12rem;
    letter-spacing: -0.01em;
}

.pretty-section-desc {
    font-size: 0.92rem;
    color: var(--text-muted);
    margin-bottom: 0.62rem;
    line-height: 1.35;
}

.pretty-card,
.pretty-simple-card {
    padding: 16px 16px;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    box-shadow: none;
    line-height: 1.3 !important;
}

.pretty-card:hover,
.pretty-simple-card:hover {
    transform: translateY(-1px);
    box-shadow: var(--hover-shadow) !important;
}

.pretty-card {
    min-height: 152px;
}

.pretty-simple-card {
    min-height: 124px;
}

.pretty-card-title,
.pretty-simple-card-title {
    font-size: 0.94rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.pretty-card-title {
    display: flex;
    align-items: center;
    gap: 8px;
}

.pretty-card-main,
.pretty-simple-card-main {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.12;
    margin-bottom: 8px;
    color: #353848;
    letter-spacing: -0.02em;
}

.pretty-simple-card-main {
    font-size: 1.48rem;
    margin-bottom: 0;
}

.pretty-card-trend {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 6px !important;
}

.pretty-dot {
    width: 14px;
    height: 14px;
    border-radius: 999px;
    display: inline-block;
    flex-shrink: 0;
}

.dot-danger {
    background: radial-gradient(circle at 30% 30%, #ff8aa0, #df3b5c 70%);
}

.dot-warning {
    background: radial-gradient(circle at 30% 30%, #ffd86e, #d89e00 70%);
}

.dot-normal {
    background: radial-gradient(circle at 30% 30%, #88f0ab, #2eaf63 70%);
}

.pretty-card-danger,
.pretty-simple-card-danger {
    border-color: var(--danger-border) !important;
    background: var(--danger-bg) !important;
}

.pretty-card-warning,
.pretty-simple-card-warning {
    border-color: var(--warning-border) !important;
    background: var(--warning-bg) !important;
}

.pretty-card-normal,
.pretty-simple-card-normal {
    border-color: var(--normal-border) !important;
    background: var(--normal-bg) !important;
}

.pretty-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid rgba(120, 120, 120, 0.16);
    margin-top: 6px;
}

.chip-red {
    background: rgba(255, 99, 132, 0.12);
    color: #d14b67;
}

.chip-yellow {
    background: rgba(255, 193, 7, 0.14);
    color: #b27b00;
}

.chip-green {
    background: rgba(40, 167, 69, 0.12);
    color: #238a45;
}

hr.pretty-divider {
    border: none;
    border-top: 1px solid var(--box-border-color);
    margin: 0.28rem 0 0.2rem 0 !important;
}

[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] {
    visibility: hidden;
    position: relative;
    min-height: 1.5rem;
}

[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"]::after {
    content: "파일을 여기에 드래그하거나 클릭하여 업로드하세요.";
    visibility: visible;
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: inherit;
    font-size: 1rem;
    font-weight: 500;
    text-align: center;
    padding: 0 1rem;
}

[data-testid="stFileUploader"] small {
    visibility: hidden;
    position: relative;
    display: inline-block;
    min-height: 1.2rem;
}

[data-testid="stFileUploader"] small::after {
    content: "형태: CSV 파일, 용량: 최대 200MB";
    visibility: visible;
    position: absolute;
    left: 0;
    top: 0;
    white-space: nowrap;
}

details {
    border-radius: 14px !important;
}

summary {
    font-weight: 700 !important;
}

div[data-testid="stExpander"] details {
    border: 1px solid rgba(120, 120, 120, 0.12) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.015) !important;
}

div[data-testid="stExpander"] details summary {
    padding-top: 0.16rem !important;
    padding-bottom: 0.16rem !important;
}

div[data-testid="stExpanderDetails"] {
    padding-top: 0.55rem !important;
    padding-bottom: 0.25rem !important;
}

div[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] > *:first-child {
    margin-top: 0 !important;
}

div[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] > *:last-child {
    margin-bottom: 0 !important;
}

div[data-testid="stExpanderDetails"] p {
    margin-bottom: 0.22rem !important;
}

div[data-testid="stExpanderDetails"] strong {
    display: inline-block;
    margin-bottom: 0.06rem !important;
}

div[data-testid="stExpanderDetails"] ul {
    margin-top: 0.06rem !important;
    margin-bottom: 0.06rem !important;
}

div[data-testid="stExpanderDetails"] li {
    margin-bottom: 0.06rem !important;
    line-height: 1.3 !important;
}

div[data-testid="stDataFrame"] {
    margin-top: 0.1rem !important;
    margin-bottom: 0.1rem !important;
}

div[data-testid="stDataFrame"] [role="columnheader"] {
    background: rgba(120, 120, 120, 0.075) !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(120, 120, 120, 0.12) !important;
}

div[data-testid="stDataFrame"] [role="gridcell"] {
    line-height: 1.3 !important;
}

div[data-testid="stMarkdownContainer"] p {
    line-height: 1.35 !important;
    margin-top: 0 !important;
    margin-bottom: 0.25rem !important;
}

div[data-testid="stMarkdownContainer"] ul,
div[data-testid="stMarkdownContainer"] ol {
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
    padding-left: 1.2rem !important;
}

div[data-testid="stMarkdownContainer"] li {
    line-height: 1.32 !important;
    margin-top: 0 !important;
    margin-bottom: 0.15rem !important;
}

div[data-testid="stMarkdownContainer"] li:last-child {
    margin-bottom: 0 !important;
}


div[data-testid="stAlert"] {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    margin-top: 0 !important;
}

div[data-testid="stAlert"] > div {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stAlert"] > div > div {
    background-color: #f3f4f6 !important;
    border-radius: 16px !important;
    color: #111827 !important;

    display: flex !important;
    align-items: center !important;

    min-height: 48px !important;
    padding: 0 20px !important;
}

div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.4 !important;
}

div[data-testid="stAlert"] svg {
    display: none !important;
}


div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    position: relative !important;
    top: -5px !important;
}



div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    width: 100% !important;
    height: 50px !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    line-height: 1 !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease, border-color 0.16s ease !important;
}

div[data-testid="stMarkdownContainer"] h3 {
    margin-bottom: 4px !important;
}


div.stButton > button > div,
div[data-testid="stDownloadButton"] > button > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
}

div.stButton > button span,
div[data-testid="stDownloadButton"] > button span,
div.stButton > button p,
div[data-testid="stDownloadButton"] > button p {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    text-align: center !important;
    margin: 0 !important;
    line-height: 1 !important;
}

button[kind="primary"] {
    background: #2f6fed !important;
    color: white !important;
    border: 1px solid #2f6fed !important;
}

button[kind="primary"]:hover {
    background: #2459bd !important;
    border-color: #2459bd !important;
    color: white !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(47, 111, 237, 0.16) !important;
}

button[kind="primary"]:focus {
    box-shadow: 0 0 0 0.2rem rgba(47, 111, 237, 0.18) !important;
    color: white !important;
}

button[kind="secondary"] {
    background: #f1f3f5 !important;
    color: #495057 !important;
    border: 1px solid #dee2e6 !important;
}

button[kind="secondary"]:hover {
    background: #e9ecef !important;
    border-color: #ced4da !important;
    color: #343a40 !important;
}

div[data-testid="stDownloadButton"] > button {
    background: #2fbf71 !important;
    color: white !important;
    border: 1px solid #2fbf71 !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    background: #249a5b !important;
    border-color: #249a5b !important;
    color: white !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(47, 191, 113, 0.16) !important;
}

div[data-testid="stDownloadButton"] > button:focus {
    box-shadow: 0 0 0 0.2rem rgba(47, 191, 113, 0.18) !important;
    color: white !important;
}

button:disabled {
    opacity: 0.72 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}


.pretty-section-desc:empty {
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
}


.pretty-chart-title {
    font-size: 0.95rem;
    font-weight: 700;
    text-align: center;
    width: 100%;
    display: block;
    margin: 0 0 0.28rem 0;
    color: #353848;
    letter-spacing: -0.01em;
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_openai_client():
    from openai import OpenAI

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY를 불러오지 못했습니다. 로컬(.env.local) 또는 배포 secrets 설정을 확인해주세요."
        )

    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_matplotlib_context():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    from matplotlib import font_manager as fm

    font_file_candidates = [
        Path("fonts/NotoSansKR-Regular.ttf"),
        Path("fonts/NanumGothic.ttf"),
        Path("fonts/NanumBarunGothic.ttf"),
    ]

    selected_font = None

    for font_path in font_file_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            selected_font = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = selected_font
            break

    if not selected_font:
        selected_font = "DejaVu Sans"
        plt.rcParams["font.family"] = selected_font

    plt.rcParams["axes.unicode_minus"] = False

    return {
        "plt": plt,
        "mdates": mdates,
        "FuncFormatter": FuncFormatter,
        "selected_font": selected_font,
    }


@st.cache_resource(show_spinner=False)
def get_pdf_font_name():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        Path("fonts/NotoSansKR-Regular.ttf"),
        Path("fonts/NanumGothic.ttf"),
        Path("fonts/NanumBarunGothic.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            font_name = f"pdf_{font_path.stem}"
            try:
                pdfmetrics.getFont(font_name)
            except Exception:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name

    return "Helvetica"


def section_space():
    st.markdown("<div style='margin-top: 2.15rem;'></div>", unsafe_allow_html=True)


def sub_section_space():
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)


def render_page_header():
    st.markdown(
        """
<div class="pretty-page-hero">
    <div class="pretty-page-title">게임 상태 진단 AI</div>
    <div class="pretty-page-desc">
        게임 KPI를 기반으로 현재 서비스 상태를 진단하고, 리스크와 개선 방향을 한눈에 제공해드립니다.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, desc: str):
    st.markdown(f"<div class='pretty-section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pretty-section-desc'>{desc}</div>", unsafe_allow_html=True)


def init_session_state():
    for key in AI_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None

    if "last_file_hash" not in st.session_state:
        st.session_state.last_file_hash = None

    if "ai_pdf_bytes" not in st.session_state:
        st.session_state.ai_pdf_bytes = None

    if "show_data_preview" not in st.session_state:
        st.session_state.show_data_preview = False


def reset_ai_results():
    for key in AI_STATE_KEYS:
        st.session_state[key] = None

    st.session_state.ai_pdf_bytes = None


def has_cached_ai_results():
    return all(st.session_state.get(key) for key in AI_STATE_KEYS)


def update_file_session(file_hash: str):
    if st.session_state.get("last_file_hash") != file_hash:
        st.session_state.last_file_hash = file_hash
        reset_ai_results()


def ask_ai(system_role: str, user_prompt: str) -> str:
    last_error = None

    try:
        client = get_openai_client()
    except Exception as e:
        return f"분석 생성 실패: {e}"

    for attempt in range(2):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "system",
                        "content": (
                            f"{system_role}\n"
                            "당신은 형식 준수에 매우 엄격해야 합니다. "
                            "사용자가 지정한 섹션 제목과 출력 순서를 정확히 지키고, "
                            "불필요한 장식, 가로줄, 코드블록, 표를 생성하지 마세요."
                        )
                    },
                    {"role": "user", "content": user_prompt}
                ]
            )

            text = getattr(response, "output_text", "") or ""
            text = text.strip()

            if not text:
                raise ValueError("AI 응답이 비어 있습니다.")

            return text

        except Exception as e:
            last_error = e
            if attempt < 1:
                time.sleep(1.2 * (attempt + 1))
            else:
                return f"분석 생성 실패: {last_error}"


def ai_format_rules(section_titles: list[str]) -> str:
    titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(section_titles)])

    return f"""
반드시 아래 형식을 정확히 지켜 작성하세요.

[출력 형식 규칙]
- 아래에 제시한 섹션 제목을 정확히 그대로 사용하세요.
- 각 최상위 섹션 제목은 반드시 '숫자. 제목' 형식으로 한 줄 단독 작성하세요.
- 최상위 섹션 제목 앞뒤에 다른 장식 문자(###, **, -, :, 괄호 등)를 붙이지 마세요.
- 가로줄(---, ***, ___)은 절대 출력하지 마세요.
- 코드블록(```)은 절대 사용하지 마세요.
- 표는 사용하지 마세요.
- 각 섹션 아래 내용은 불릿포인트(-) 또는 하위 불릿(  - ) 형태로만 작성하세요.
- 최상위 번호는 한 번만 사용하고, 같은 제목을 반복하지 마세요.
- 섹션 순서는 반드시 아래 순서를 따르세요.
- 불필요한 서론/결론 문장은 쓰지 마세요.
- 숫자 해석은 간결하게 쓰고, PM이 바로 이해할 수 있는 실행 관점으로 작성하세요.

[반드시 사용할 섹션 제목]
{titles_text}
"""


def format_ai_section_text(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    formatted_lines = []
    section_started = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            formatted_lines.append("")
            continue

        if stripped in ("---", "***", "___"):
            continue

        if re.match(r"^\d+\.\s+.+", stripped) and raw_line == raw_line.lstrip():
            if section_started:
                formatted_lines.append("---")
            formatted_lines.append(f"**{stripped}**")
            section_started = True
            continue

        formatted_lines.append(line)

    return "\n\n".join(formatted_lines)


def _clean_ai_lines_for_pdf(text: str) -> list[str]:
    cleaned = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if stripped in ("---", "***", "___"):
            continue
        cleaned.append(stripped)
    return cleaned


def _append_ai_block_to_story(story, block_title, text, styles, colors):
    from reportlab.platypus import Paragraph, Spacer, HRFlowable

    story.append(Paragraph(escape(block_title), styles["block_title"]))
    story.append(Spacer(1, 8))

    section_started = False
    for line in _clean_ai_lines_for_pdf(text):
        if not line:
            story.append(Spacer(1, 4))
            continue

        if re.match(r"^\d+\.\s+.+", line):
            if section_started:
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.7,
                        color=colors.HexColor("#D9DDE5"),
                        spaceBefore=2,
                        spaceAfter=3,
                    )
                )
            story.append(Paragraph(escape(line), styles["subheading"]))
            story.append(Spacer(1, 2))
            section_started = True
            continue

        if line.startswith("- "):
            bullet_text = escape(line[2:].strip())
            story.append(Paragraph(f"• {bullet_text}", styles["bullet"]))
            continue

        story.append(Paragraph(escape(line), styles["body"]))

    story.append(Spacer(1, 10))


@st.cache_data(show_spinner=False)
def build_ai_pdf_bytes(analyst_result: str, risk_result: str, improvement_result: str, insight_result: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=36,
        title="게임 상태 진단 결과",
        author="게임 상태 진단 AI",
    )

    base_styles = getSampleStyleSheet()
    font_name = get_pdf_font_name()

    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base_styles["Title"],
            fontName=font_name,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#2E3440"),
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "desc": ParagraphStyle(
            "desc",
            parent=base_styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=14,
            wordWrap="CJK",
        ),
        "block_title": ParagraphStyle(
            "block_title",
            parent=base_styles["Heading2"],
            fontName=font_name,
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#374151"),
            spaceBefore=4,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "subheading": ParagraphStyle(
            "subheading",
            parent=base_styles["Heading3"],
            fontName=font_name,
            fontSize=11.3,
            leading=15,
            textColor=colors.HexColor("#2E3440"),
            spaceBefore=1,
            spaceAfter=2,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base_styles["BodyText"],
            fontName=font_name,
            fontSize=10.2,
            leading=16,
            textColor=colors.HexColor("#374151"),
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base_styles["BodyText"],
            fontName=font_name,
            fontSize=10.2,
            leading=16,
            textColor=colors.HexColor("#374151"),
            leftIndent=10,
            firstLineIndent=0,
            spaceAfter=3,
            wordWrap="CJK",
        ),
    }

    story = []
    story.append(Paragraph("게임 상태 진단 결과", styles["title"]))
    story.append(Paragraph("지표 해석, 위험 신호, 개선 방안, 종합 인사이트를 한 번에 정리한 PDF입니다.", styles["desc"]))

    _append_ai_block_to_story(story, "1. 각 지표 분석 결과", analyst_result, styles, colors)
    _append_ai_block_to_story(story, "2. 위험 신호 정리", risk_result, styles, colors)
    _append_ai_block_to_story(story, "3. 개선 방안", improvement_result, styles, colors)
    _append_ai_block_to_story(story, "4. 종합 인사이트", insight_result, styles, colors)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def detect_pur_mode(series: pd.Series) -> str:
    clean = pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce").dropna()
    if clean.empty:
        return "ratio"
    return "ratio" if clean.max() <= 1 else "percent"


def normalize_pur(series: pd.Series, mode: str) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")
    if mode == "percent":
        return values / 100
    return values


def safe_pct_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return ((current - previous) / previous) * 100


def is_real_number(val):
    return isinstance(val, Real) and not isinstance(val, bool)


def format_with_commas(val, decimals_if_needed: int = 2) -> str:
    if val is None or pd.isna(val):
        return "N/A"

    if is_real_number(val):
        if float(val).is_integer():
            return f"{int(val):,}"
        return f"{float(val):,.{decimals_if_needed}f}"

    return str(val)


def format_pct(val):
    if val is None or pd.isna(val):
        return "계산 불가"
    return f"{float(val):.2f}%"


def format_num(val):
    return format_with_commas(val, decimals_if_needed=2)


def display_pct(metric: str, val):
    if val is None or pd.isna(val):
        return "계산 불가"
    return f"{float(val):.2f}%"


def display_num(metric: str, val):
    if val is None or pd.isna(val):
        return "N/A"

    if metric == "PUR":
        return f"{float(val) * 100:.2f}%"
    
    if is_real_number(val):
        return f"{round(float(val)):,}"

    return str(val)


def trend_arrow(change):
    if change is None or pd.isna(change):
        return "-"
    if change > 0:
        return "▲"
    if change < 0:
        return "▼"
    return "-"


def trend_text(change):
    if change is None or pd.isna(change):
        return "- 계산 불가"
    arrow = "▲" if change > 0 else "▼" if change < 0 else "•"
    return f"{arrow} {float(change):.2f}%"


def format_overview_date_display(date_val):
    if date_val is None or pd.isna(date_val):
        return "N/A"
    if isinstance(date_val, pd.Timestamp):
        return date_val.strftime("%Y-%m-%d")
    return str(date_val)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "일자" in df.columns:
        df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
        df = df.sort_values("일자").reset_index(drop=True)

    pur_mode = None
    if "PUR" in df.columns:
        pur_mode = detect_pur_mode(df["PUR"])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if col == "PUR":
                df[col] = normalize_pur(df[col], pur_mode or "ratio")
            else:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ""),
                    errors="coerce"
                )

    df.attrs["pur_mode"] = pur_mode or "ratio"
    return df


def read_csv_flexible_from_bytes(raw: bytes) -> pd.DataFrame:
    tried = []

    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as e:
            tried.append(f"{encoding}: {e}")

    raise ValueError(
        "CSV를 읽지 못했습니다. 인코딩을 확인해주세요.\n" + "\n".join(tried)
    )


@st.cache_data(show_spinner=False)
def load_and_clean_csv(file_bytes: bytes) -> pd.DataFrame:
    raw_df = read_csv_flexible_from_bytes(file_bytes)
    return clean_dataframe(raw_df)


def load_uploaded_dataframe(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    update_file_session(file_hash)
    df = load_and_clean_csv(file_bytes)

    return df, file_hash


def data_quality_check(df: pd.DataFrame):
    issues = []
    infos = []

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"누락 컬럼: {', '.join(missing_cols)}")

    if "일자" in df.columns:
        missing_dates = df["일자"].isna().sum()
        if missing_dates > 0:
            issues.append(f"해석 불가능한 일자 값 {missing_dates}건이 있습니다.")

        duplicated_dates = df["일자"].duplicated().sum()
        if duplicated_dates > 0:
            issues.append(f"중복된 일자 {duplicated_dates}건이 있습니다.")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            null_cnt = df[col].isna().sum()
            if null_cnt > 0:
                issues.append(f"{col}: 숫자 변환 실패 또는 결측 {null_cnt}건")
            if (df[col].dropna() < 0).any():
                issues.append(f"{col}: 음수 값이 포함되어 있습니다.")

    if "PUR" in df.columns:
        pur_mode = df.attrs.get("pur_mode", "ratio")
        if pur_mode == "ratio":
            infos.append("PUR은 0~1 비율값으로 인식되어 내부 계산됩니다.")
        else:
            infos.append("PUR은 퍼센트값으로 인식되어 내부적으로 0~1 비율로 변환되었습니다.")

    return issues, infos


def make_metric_summary(df: pd.DataFrame, recent_days: int = 7) -> dict:
    summary = {}

    if "일자" not in df.columns or df["일자"].isna().all():
        recent_df = df.tail(recent_days).copy()
        prev_df = (
            df.iloc[:-recent_days].tail(recent_days).copy()
            if len(df) > recent_days
            else df.head(max(len(df) - 1, 1)).copy()
        )
    else:
        latest_date = df["일자"].max()
        recent_start = latest_date - pd.Timedelta(days=recent_days - 1)
        prev_end = recent_start - pd.Timedelta(days=1)
        prev_start = prev_end - pd.Timedelta(days=recent_days - 1)

        recent_df = df[(df["일자"] >= recent_start) & (df["일자"] <= latest_date)].copy()
        prev_df = df[(df["일자"] >= prev_start) & (df["일자"] <= prev_end)].copy()

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        valid_recent = recent_df[col].dropna()
        latest = valid_recent.iloc[-1] if len(valid_recent) > 0 else None
        earliest_recent = valid_recent.iloc[0] if len(valid_recent) > 0 else None
        current_avg = recent_df[col].mean() if len(recent_df) > 0 else None
        previous_avg = prev_df[col].mean() if len(prev_df) > 0 else None

        summary[col] = {
            "latest": latest,
            "recent_avg": current_avg,
            "previous_avg": previous_avg,
            "recent_avg_vs_prev_avg_pct": safe_pct_change(current_avg, previous_avg),
            "latest_vs_recent_start_pct": safe_pct_change(latest, earliest_recent),
            "recent_min": recent_df[col].min() if len(recent_df) > 0 else None,
            "recent_max": recent_df[col].max() if len(recent_df) > 0 else None,
        }

    return summary


def priority_from_change(metric: str, change):
    if change is None or metric not in THRESHOLDS:
        return "⚪ 참고"

    if change <= THRESHOLDS[metric]["critical"]:
        return "🔴 즉시 대응"
    if change <= THRESHOLDS[metric]["warning"]:
        return "🟡 주의"
    return "🟢 정상"


def severity_icon(metric: str, change):
    if change is None or metric not in THRESHOLDS:
        return "🟢"
    if change <= THRESHOLDS[metric]["critical"]:
        return "🔴"
    if change <= THRESHOLDS[metric]["warning"]:
        return "🟡"
    return "🟢"


def severity_style(priority):
    if priority == "🔴 즉시 대응":
        return "danger"
    if priority == "🟡 주의":
        return "warning"
    return "normal"


def dot_class_from_priority(priority):
    style = severity_style(priority)
    if style == "danger":
        return "dot-danger"
    if style == "warning":
        return "dot-warning"
    return "dot-normal"


def status_chip_class(priority):
    if priority == "🔴 즉시 대응":
        return "chip-red"
    if priority == "🟡 주의":
        return "chip-yellow"
    return "chip-green"


def make_summary_text(summary: dict) -> str:
    lines = []
    for metric, values in summary.items():
        lines.append(
            f"- {metric}: "
            f"최신값={format_num(values['latest'])}, "
            f"최근평균={format_num(values['recent_avg'])}, "
            f"이전평균={format_num(values['previous_avg'])}, "
            f"최근평균 대비 증감률={format_pct(values['recent_avg_vs_prev_avg_pct'])}, "
            f"최근 시작 대비 최신 증감률={format_pct(values['latest_vs_recent_start_pct'])}"
        )
    return "\n".join(lines)


def action_recommendation_for_metric(metric: str, priority: str):
    if priority == "🔴 즉시 대응":
        actions = {
            "접속유저": "접속 감소 원인과 유입 채널 변화를 즉시 점검",
            "신규유저": "UA 채널 성과 및 광고·스토어 노출 변화를 우선 점검",
            "복귀유저": "복귀 캠페인 반응과 재방문 동선을 중심으로 점검",
            "기존유저": "이탈 구간과 콘텐츠 소진 흐름을 우선 점검",
            "평균동접": "동접 감소 시간대와 주요 콘텐츠 참여율 점검",
            "최고동접": "피크 타임 유입 요인과 이벤트 효과 점검",
            "플레이시간": "플레이 시간 감소 원인과 콘텐츠 구조 점검",
            "총매출": "매출 하락 원인과 결제 이벤트 구조 점검",
            "개인매출": "1인당 결제 감소 요인과 고가 상품 반응 점검",
            "PU": "과금 유저 감소 원인과 결제 이벤트 참여율 점검",
            "PUR": "결제 전환 흐름과 첫 구매 경험 구조 점검",
            "ARPPU": "고과금 유저 결제 패턴과 상품 매력도 점검",
        }
    else:
        actions = {
            "접속유저": "접속 추이와 유입 흐름 지속 모니터링",
            "신규유저": "신규 유입 채널 효율 추이 모니터링",
            "복귀유저": "복귀 유저 반응과 재방문 흐름 확인",
            "기존유저": "잔존율과 콘텐츠 소비 흐름 확인",
            "평균동접": "시간대별 동접 추이 확인",
            "최고동접": "피크 타임 이벤트 반응 확인",
            "플레이시간": "세션 길이와 반복 플레이 흐름 확인",
            "총매출": "매출 구조와 상품별 기여도 확인",
            "개인매출": "유저당 결제 금액 변화 확인",
            "PU": "과금 유저 수 추이 확인",
            "PUR": "결제 전환률 흐름 확인",
            "ARPPU": "고과금 상품 구매 추이 확인",
        }

    return actions.get(metric, "관련 지표 흐름 점검")


def detect_metric_risks(summary: dict):
    candidates = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")
        if change is None:
            continue

        priority = priority_from_change(metric, change)

        if priority in ("🔴 즉시 대응", "🟡 주의"):
            candidates.append({
                "지표": metric,
                "change": change,
                "변화율": format_pct(change),
                "base_priority": priority,
            })

    candidates.sort(key=lambda row: row["change"])

    rows = []

    for idx, item in enumerate(candidates):
        metric = item["지표"]

        if item["base_priority"] == "🔴 즉시 대응" and idx < 2:
            final_priority = "🔴 즉시 대응"
        else:
            final_priority = "🟡 주의"

        rows.append({
            "지표": metric,
            "변화율": item["변화율"],
            "우선순위": final_priority,
            "액션": action_recommendation_for_metric(metric, final_priority)
        })

    return rows



def build_grouped_action_items(metric_risks):
    groups = {
        "유입·복귀 구조 점검": [],
        "과금 구조 점검": [],
        "잔존·콘텐츠 구조 점검": [],
        "동접·플레이 흐름 점검": [],
    }

    for risk in metric_risks:
        metric = risk.get("지표")

        if metric in ("신규유저", "복귀유저", "접속유저"):
            groups["유입·복귀 구조 점검"].append(metric)
        elif metric in ("총매출", "개인매출", "PU", "PUR", "ARPPU"):
            groups["과금 구조 점검"].append(metric)
        elif metric in ("기존유저",):
            groups["잔존·콘텐츠 구조 점검"].append(metric)
        elif metric in ("평균동접", "최고동접", "플레이시간"):
            groups["동접·플레이 흐름 점검"].append(metric)

    action_messages = []

    if groups["유입·복귀 구조 점검"]:
        action_messages.append({
            "점검 영역": "유입·복귀 구조 점검",
            "관련 지표": f"{', '.join(groups['유입·복귀 구조 점검'])} 감소",
            "실행 포인트": "UA 채널, 광고·스토어 노출, 복귀 캠페인, 재방문 동선을 함께 점검"
        })

    if groups["과금 구조 점검"]:
        action_messages.append({
            "점검 영역": "과금 구조 점검",
            "관련 지표": f"{', '.join(groups['과금 구조 점검'])} 하락",
            "실행 포인트": "결제 전환 흐름, 첫 구매 경험, 결제 이벤트 참여율, 상품 매력도를 함께 점검"
        })

    if groups["잔존·콘텐츠 구조 점검"]:
        action_messages.append({
            "점검 영역": "잔존·콘텐츠 구조 점검",
            "관련 지표": f"{', '.join(groups['잔존·콘텐츠 구조 점검'])} 감소",
            "실행 포인트": "이탈 구간, 콘텐츠 소진 속도, 업데이트 반응을 함께 점검"
        })

    if groups["동접·플레이 흐름 점검"]:
        action_messages.append({
            "점검 영역": "동접·플레이 흐름 점검",
            "관련 지표": f"{', '.join(groups['동접·플레이 흐름 점검'])} 하락",
            "실행 포인트": "시간대별 동접 변화, 세션 길이, 핵심 콘텐츠 참여율을 함께 점검"
        })

    return action_messages


def render_grouped_action_section(metric_risks):
    grouped_actions = build_grouped_action_items(metric_risks)

    if not grouped_actions:
        return

    st.markdown("**실행 묶음 액션**")
    st.dataframe(
        pd.DataFrame(grouped_actions),
        use_container_width=True,
        hide_index=True
    )



def detect_combined_risks(summary: dict):
    def ch(metric):
        return summary.get(metric, {}).get("recent_avg_vs_prev_avg_pct")

    dau = ch("접속유저")
    new = ch("신규유저")
    ret = ch("복귀유저")
    existing = ch("기존유저")
    revenue = ch("총매출")
    pu = ch("PU")
    pur = ch("PUR")
    arppu = ch("ARPPU")
    playtime = ch("플레이시간")
    avg_ccu = ch("평균동접")

    risk_items = []

    def add_risk(priority: int, key: str, message: str):
        risk_items.append({
            "priority": priority,
            "key": key,
            "message": message,
        })

    if pu is not None and pu <= -5 and pur is not None and pur <= -5:
        add_risk(
            1,
            "monetization_conversion",
            "과금 유저 수와 결제 전환률이 동시에 하락해 BM 구조 점검이 필요합니다."
        )

    if revenue is not None and revenue <= -5 and arppu is not None and arppu <= -5:
        add_risk(
            1,
            "monetization_value",
            "총매출과 ARPPU가 함께 하락해 결제 가치 제안이 약해졌을 수 있습니다."
        )

    if dau is not None and revenue is not None and dau <= -8 and revenue <= -8:
        add_risk(
            2,
            "service_revenue_drop",
            "접속유저와 총매출이 함께 하락해 서비스 전반의 활력 저하 가능성이 큽니다."
        )

    if new is not None and new <= -8 and existing is not None and existing <= -5:
        add_risk(
            3,
            "user_acquisition_retention",
            "신규유저 유입 감소와 기존유저 감소가 함께 발생해 유입과 정착 모두 약화되고 있습니다."
        )

    if ret is not None and ret <= -8 and dau is not None and dau <= -5:
        add_risk(
            3,
            "returning_user_drop",
            "복귀유저 감소가 접속 감소와 함께 나타나 재방문 유도 장치가 약해졌을 수 있습니다."
        )

    if avg_ccu is not None and avg_ccu <= -8 and playtime is not None and playtime <= -8:
        add_risk(
            4,
            "engagement_drop",
            "평균동접과 플레이시간이 함께 감소해 핵심 콘텐츠 몰입도 저하 신호가 있습니다."
        )

    if pu is not None and pu <= -8 and arppu is not None and arppu > 0:
        add_risk(
            4,
            "whale_dependency",
            "과금 유저 수는 줄었지만 1인당 결제는 유지되어 소수 고과금 의존 가능성이 있습니다."
        )

    merge_rules = [
        {
            "keys": {"monetization_conversion", "monetization_value"},
            "priority": 1,
            "message": "과금 유저 수, 결제 전환률, ARPPU가 함께 약화되어 BM 구조와 결제 가치 제안을 우선 점검해야 합니다."
        },
        {
            "keys": {"service_revenue_drop", "monetization_conversion"},
            "priority": 1,
            "message": "접속유저와 총매출이 함께 하락했고 결제 전환 지표도 약화되어 서비스 활력과 BM 구조를 함께 점검해야 합니다."
        },
        {
            "keys": {"user_acquisition_retention", "returning_user_drop"},
            "priority": 3,
            "message": "신규·기존·복귀 유저 흐름이 동시에 약화되어 유입, 정착, 재방문 구조 전반을 점검해야 합니다."
        },
    ]

    active_keys = {item["key"] for item in risk_items}
    merged_items = []
    used_keys = set()

    for rule in merge_rules:
        if rule["keys"].issubset(active_keys):
            merged_items.append({
                "priority": rule["priority"],
                "message": rule["message"],
            })
            used_keys.update(rule["keys"])

    for item in risk_items:
        if item["key"] not in used_keys:
            merged_items.append({
                "priority": item["priority"],
                "message": item["message"],
            })

    unique_messages = []
    seen_messages = set()

    for item in sorted(merged_items, key=lambda x: x["priority"]):
        message = item["message"]
        if message in seen_messages:
            continue
        seen_messages.add(message)
        unique_messages.append(message)

    return unique_messages


def extract_action_items(summary: dict):
    actions = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")
        priority = priority_from_change(metric, change)

        if priority == "🔴 즉시 대응":
            actions.append({
                "지표": metric,
                "변화율": format_pct(change),
                "액션": f"{metric} 급락 → 원인 분석 및 즉시 대응 필요"
            })

    return actions


def final_priority_for_metric(metric: str, change, critical_metrics: set):
    base_priority = priority_from_change(metric, change)

    if base_priority == "🔴 즉시 대응":
        if metric in critical_metrics:
            return "🔴 즉시 대응"
        return "🟡 주의"

    return base_priority


def make_summary_table(summary: dict):
    critical_candidates = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")
        priority = priority_from_change(metric, change)

        if priority == "🔴 즉시 대응":
            critical_candidates.append({
                "metric": metric,
                "change": change,
            })

    # 하락폭 기준 정렬
    critical_candidates.sort(key=lambda row: row["change"])

    # 상위 2개만 선택
    critical_metrics = {item["metric"] for item in critical_candidates[:2]}

    rows = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")

        rows.append({
            "지표": metric,
            "전주 평균": display_num(metric, values.get("previous_avg")),
            "금주 평균": display_num(metric, values.get("recent_avg")),
            "전주 대비": display_pct(metric, change),
            "판단": final_priority_for_metric(metric, change, critical_metrics)
        })

    return pd.DataFrame(rows)


def make_highlight_summary(summary: dict):
    highlight_lines = []

    for metric in HIGHLIGHT_METRICS:
        if metric not in summary:
            continue

        change = summary[metric].get("recent_avg_vs_prev_avg_pct")
        recent_avg = summary[metric].get("recent_avg")
        severity = severity_icon(metric, change)
        priority = priority_from_change(metric, change)

        line = (
            f"{severity} **{metric}** | "
            f"평균 {display_num(metric, recent_avg)} | "
            f"{display_pct(metric, change)}"
        )


        highlight_lines.append(line)

    return highlight_lines


def build_status_message(action_items, metric_risks, combined_risks):
    urgent_count = len(action_items)
    risk_count = len(metric_risks)

    if urgent_count >= 3:
        return "🚨 현재 서비스 상태: 즉각 대응이 필요합니다.", "error"
    if urgent_count >= 1 or len(combined_risks) >= 1:
        return "⚠️ 현재 서비스 상태: 주의가 필요합니다.", "warning"
    if risk_count == 0:
        return "✅ 현재 서비스 상태: 전반적으로 안정적입니다.", "success"
    return "ℹ️ 현재 서비스 상태: 일부 변동이 있어 모니터링이 필요합니다.", "info"



@st.cache_data(show_spinner=False)
def build_dashboard_cache(df: pd.DataFrame):
    summary = make_metric_summary(df, recent_days=7)
    summary_text = make_summary_text(summary)[:2500]
    metric_risks = detect_metric_risks(summary)
    combined_risks = detect_combined_risks(summary)
    action_items = extract_action_items(summary)
    summary_df = make_summary_table(summary)
    highlight_lines = make_highlight_summary(summary)

    return {
        "summary": summary,
        "summary_text": summary_text,
        "metric_risks": metric_risks,
        "combined_risks": combined_risks,
        "action_items": action_items,
        "summary_df": summary_df,
        "highlight_lines": highlight_lines,
    }


@st.cache_data(show_spinner=False)
def build_data_quality_cache(df: pd.DataFrame):
    return data_quality_check(df)


def make_chart_payload(df: pd.DataFrame, metric: str):
    if metric not in df.columns or "일자" not in df.columns:
        return None

    chart_df = df[["일자", metric]].dropna().copy()
    if chart_df.empty:
        return None

    chart_df = chart_df.sort_values("일자").reset_index(drop=True)
    chart_df["일자"] = pd.to_datetime(chart_df["일자"]).dt.strftime("%Y-%m-%d")

    return {
        "metric": metric,
        "dates": chart_df["일자"].tolist(),
        "values": chart_df[metric].astype(float).tolist(),
    }


@st.cache_data(show_spinner=False)
def build_chart_payloads(df: pd.DataFrame, metrics: tuple):
    payloads = {}
    for metric in metrics:
        payload = make_chart_payload(df, metric)
        if payload is not None:
            payloads[metric] = payload
    return payloads


@st.cache_data(show_spinner=False)
def render_chart_image_from_payload(payload_json: str) -> bytes:
    mpl = get_matplotlib_context()
    plt = mpl["plt"]
    mdates = mpl["mdates"]
    FuncFormatter = mpl["FuncFormatter"]

    payload = json.loads(payload_json)
    metric = payload["metric"]

    chart_df = pd.DataFrame({
        "일자": pd.to_datetime(payload["dates"]),
        metric: payload["values"]
    })

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(chart_df["일자"], chart_df[metric], marker="o", linewidth=2)
    ax.set_xlabel("일자")
    ax.set_ylabel(metric)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    if metric == "총매출":
        ax.set_ylabel("총매출 (억)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1e8:.1f}억"))
    elif metric == "PUR":
        ax.set_ylabel("PUR (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x*100:.1f}%"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))

    if len(chart_df) >= 14:
        recent_df = chart_df.tail(7)
        prev_df = chart_df.iloc[-14:-7]
        recent_avg = recent_df[metric].mean()
        prev_avg = prev_df[metric].mean()

        ax.axhline(
            y=recent_avg,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"최근 7일 평균: {recent_avg/1e8:.1f}억" if metric == "총매출"
                  else f"최근 7일 평균: {recent_avg*100:.1f}%" if metric == "PUR"
                  else f"최근 7일 평균: {recent_avg:,.0f}"
        )
        ax.axhline(
            y=prev_avg,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label=f"이전 7일 평균: {prev_avg/1e8:.1f}억" if metric == "총매출"
                  else f"이전 7일 평균: {prev_avg*100:.1f}%" if metric == "PUR"
                  else f"이전 7일 평균: {prev_avg:,.0f}"
        )

    metric_mean = chart_df[metric].mean()
    if metric_mean and not pd.isna(metric_mean):
        upper_threshold = metric_mean * 1.15
        lower_threshold = metric_mean * 0.85
        outliers = chart_df[
            (chart_df[metric] >= upper_threshold) |
            (chart_df[metric] <= lower_threshold)
        ]
        if len(outliers) > 0:
            ax.scatter(outliers["일자"], outliers[metric], s=80, zorder=5, label="이상치")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def build_risk_table_text(metric_risks):
    if not metric_risks:
        return "없음"
    return pd.DataFrame(metric_risks).to_string(index=False)


def build_combined_risk_text(combined_risks):
    if not combined_risks:
        return "없음"
    return chr(10).join(combined_risks)


def generate_analyst_result(summary_text):
    return ask_ai(
        "당신은 모바일 게임 KPI를 해석하는 시니어 데이터 분석가입니다.",
        f"""
아래는 게임 KPI의 사전 요약입니다.

[지표 요약]
{summary_text}

요청:
1. 각 지표에 대한 핵심 해석
2. 특히 의미 있는 상승/하락 지표 강조
3. 숫자 나열보다 PM 관점 해석 중심
4. 각 섹션은 간결하고 명확하게 작성

{ai_format_rules(["지표별 분석", "핵심 변화 요약"])}
"""
    )


def generate_risk_result(summary_text, metric_risks, combined_risks):
    return ask_ai(
        "당신은 라이브 게임 운영 리스크를 점검하는 시니어 운영 전략가입니다.",
        f"""
아래는 KPI 요약과 Python이 감지한 위험 신호입니다.

[지표 요약]
{summary_text}

[지표별 위험]
{build_risk_table_text(metric_risks)}

[지표 조합 위험]
{build_combined_risk_text(combined_risks)}

{ai_format_rules(["지금 당장 액션 필요", "주의", "정상"])}
"""
    )


def generate_improvement_result(summary_text, risk_result):
    return ask_ai(
        "당신은 게임 사업 PM이자 라이브 운영 전략 전문가입니다.",
        f"""
아래는 KPI 분석 및 위험 신호입니다.

[지표 요약]
{summary_text}

[위험 신호]
{risk_result}

{ai_format_rules(["단기 개선안", "중기 개선안", "바로 실행할 과제"])}
"""
    )


def generate_insight_result(summary_text, analyst_result, risk_result, improvement_result):
    return ask_ai(
        "당신은 게임 사업 PM 총괄입니다. 팀 공유용으로 인사이트를 정리하세요.",
        f"""
아래는 게임 KPI 분석 결과입니다.

[지표 요약]
{summary_text}

[지표 분석]
{analyst_result}

[위험 신호]
{risk_result}

[개선 방안]
{improvement_result}

{ai_format_rules(["서비스 상태 진단", "종합 인사이트", "주목 KPI", "한 줄 결론"])}
"""
    )


def run_ai_analysis_pipeline(summary_text, metric_risks, combined_risks):
    if has_cached_ai_results():
        return "이미 분석된 결과입니다. 기존 결과를 표시합니다.", "info"

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.info("데이터를 읽고 있습니다…")
        progress_bar.progress(10)

        status_text.info("지표를 해석하는 중입니다…")
        progress_bar.progress(30)
        analyst_result = generate_analyst_result(summary_text)

        status_text.info("위험 신호를 분석하는 중입니다…")
        progress_bar.progress(55)
        risk_result = generate_risk_result(summary_text, metric_risks, combined_risks)

        status_text.info("개선 방향을 도출하는 중입니다…")
        progress_bar.progress(80)
        improvement_result = generate_improvement_result(summary_text, risk_result)

        status_text.info("최종 인사이트를 정리하고 있습니다…")
        progress_bar.progress(95)
        insight_result = generate_insight_result(summary_text, analyst_result, risk_result, improvement_result)

        st.session_state.analyst_result = analyst_result
        st.session_state.risk_result = risk_result
        st.session_state.improvement_result = improvement_result
        st.session_state.insight_result = insight_result
        st.session_state.ai_pdf_bytes = None

        progress_bar.progress(100)
        status_text.success("분석이 완료되었습니다. 아래 결과를 확인해 주세요.")
        return None, None

    except Exception as e:
        status_text.error(f"분석 중 오류가 발생했습니다: {e}")
        return f"분석 중 오류가 발생했습니다: {e}", "error"


def render_status_banner(action_items, metric_risks, combined_risks):
    status_message, status_level = build_status_message(
        action_items=action_items,
        metric_risks=metric_risks,
        combined_risks=combined_risks
    )

    if status_level == "error":
        st.error(status_message)
    elif status_level == "warning":
        st.warning(status_message)
    elif status_level == "success":
        st.success(status_message)
    else:
        st.info(status_message)


def render_simple_card(title: str, value: str, status: str = "normal"):
    st.markdown(
        f"""
<div class="pretty-simple-card pretty-simple-card-{status}">
    <div class="pretty-simple-card-title">{title}</div>
    <div class="pretty-simple-card-main">{value}</div>
</div>
""",
        unsafe_allow_html=True
    )


def render_overview_section(df, action_items, metric_risks, combined_risks, summary):

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    render_section_header(
        "서비스 현황 요약",
        ""
    )

    status_message, status_level = build_status_message(
        action_items=action_items,
        metric_risks=metric_risks,
        combined_risks=combined_risks
    )

    top_risk = None
    for risk in metric_risks:
        if risk.get("우선순위") == "🔴 즉시 대응":
            top_risk = risk
            break

    clean_status = (
        status_message
        .replace("🚨 ", "")
        .replace("⚠️ ", "")
        .replace("✅ ", "")
        .replace("ℹ️ ", "")
        .replace("현재 서비스 상태: ", "")
    )


    if top_risk:
        message = f"{top_risk['지표']} 감소 → {top_risk['액션']} 필요"
        st.info(message)


    top1, top2, top3, top4 = st.columns(4)

    with top1:
        if "일자" in df.columns and not df["일자"].isna().all():
            start_date = df["일자"].min()
            end_date = df["일자"].max()
            value = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        else:
            value = "일자 없음"

        render_simple_card("분석 기간", value, status="normal")

    with top2:
        if "일자" in df.columns and not df["일자"].isna().all():
            start_date = df["일자"].min()
            end_date = df["일자"].max()
            value = f"{(end_date - start_date).days + 1}일"
        else:
            value = "계산 불가"

        render_simple_card("분석 일수", value, status="normal")

    with top3:
        urgent_status = "danger" if len(action_items) >= 1 else "normal"
        render_simple_card("긴급 액션", f"{len(action_items)}건", status=urgent_status)

    with top4:
        risk_count = len(metric_risks) + len(combined_risks)
        risk_status = "danger" if risk_count >= 5 else "warning" if risk_count >= 1 else "normal"
        render_simple_card("위험 신호", f"{risk_count}건", status=risk_status)

    sub_section_space()



def render_risk_section(metric_risks, combined_risks):
    section_space()
    render_section_header(
        "위험 신호 요약",
        "단일 지표 기준 위험과 지표 조합 기준 위험을 함께 점검합니다."
    )

    with st.container(border=True):
        st.markdown("**지표별 위험 신호**")
        if metric_risks:
            st.dataframe(pd.DataFrame(metric_risks), use_container_width=True, hide_index=True)
        else:
            st.success("임계치 기준 위험 신호가 감지되지 않았습니다.")

        if metric_risks:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_grouped_action_section(metric_risks)


        st.markdown("**지표 조합 기반 위험 신호**")
        if combined_risks:
            for risk in combined_risks:
                st.write(f"- {risk}")
        else:
            st.write("- 별도 조합 리스크는 감지되지 않았습니다.")
        
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


def render_pretty_kpi_card(metric: str, latest, change):
    priority = priority_from_change(metric, change) or "정상"
    status = severity_style(priority)
    dot_class = dot_class_from_priority(priority)

    status_text = (
        priority
        .replace("🔴 ", "")
        .replace("🟡 ", "")
        .replace("🟢 ", "")
        .replace("⚪ ", "")
    )

    st.markdown(
        f"""
<div class="pretty-card pretty-card-{status}">
    <div class="pretty-card-title">
        <span class="pretty-dot {dot_class}"></span>
        <span>{metric}</span>
    </div>
    <div class="pretty-card-main">{display_num(metric, latest)}</div>
    <div class="pretty-card-trend">최근 평균 변화율: {trend_text(change)} ({status_text})</div>
</div>
""",
        unsafe_allow_html=True
    )


def render_kpi_cards(summary: dict):
    available_metrics = [metric for metric in CARD_METRICS if metric in summary]

    first_row = available_metrics[:3]
    second_row = available_metrics[3:6]

    if first_row:
        cols1 = st.columns(3)
        for idx, metric in enumerate(first_row):
            with cols1[idx]:
                render_pretty_kpi_card(
                    metric,
                    summary[metric].get("latest"),
                    summary[metric].get("recent_avg_vs_prev_avg_pct")
                )

    if second_row:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        cols2 = st.columns(3)
        for idx, metric in enumerate(second_row):
            with cols2[idx]:
                render_pretty_kpi_card(
                    metric,
                    summary[metric].get("latest"),
                    summary[metric].get("recent_avg_vs_prev_avg_pct")
                )


def render_kpi_card_section(summary):
    section_space()
    render_section_header(
        "핵심 KPI 카드",
        "핵심 지표의 최신값과 최근 평균 변화율을 카드 형태로 빠르게 확인합니다."
    )
    render_kpi_cards(summary)


def render_highlight_section(highlight_lines):
    section_space()
    render_section_header(
        "핵심 지표 한눈에 보기",
        "중요 지표를 짧은 요약 문장으로 정리했습니다."
    )

    with st.container(border=True):
        for line in highlight_lines:
            st.markdown(f"- {line}")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


def render_summary_table_section(summary_df):
    section_space()
    render_section_header(
        "지표 요약 표",
        "최신값, 최근 평균, 이전 평균, 변화 방향과 심각도를 한 번에 비교합니다."
    )

    with st.container(border=True):
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_chart_card(metric: str, image_bytes: bytes):
    st.markdown(
        f"""
<div class='pretty-chart-title'>{metric}</div>
""",
        unsafe_allow_html=True
    )
    st.image(image_bytes, use_container_width=True)


def render_chart_section(df):
    section_space()
    render_section_header(
        "주요 지표 그래프",
        "지표별 추이와 최근 흐름을 시각적으로 비교할 수 있습니다."
    )

    with st.container(border=True):
        chart_payloads = build_chart_payloads(df, tuple(CHART_METRICS))
        graph_cols = st.columns(2)

        shown = 0
        for metric in CHART_METRICS:
            payload = chart_payloads.get(metric)
            if not payload:
                continue

            image_bytes = render_chart_image_from_payload(
                json.dumps(payload, ensure_ascii=False)
            )

            with graph_cols[shown % 2]:
                render_chart_card(metric, image_bytes)
            shown += 1

        if shown == 0:
            st.info("표시할 그래프가 없습니다. 일자 컬럼과 주요 지표 값이 포함되어 있는지 확인해 주세요.")


@st.cache_data(show_spinner=False)
def format_ai_section_text_cached(text: str) -> str:
    return format_ai_section_text(text)


def render_ai_result_block(title: str, text: str):
    with st.expander(title, expanded=True):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown(format_ai_section_text_cached(text))
        st.markdown("")


def render_ai_results_section():
    if not st.session_state.insight_result:
        return

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    render_ai_result_block("1. 각 지표 분석 결과", st.session_state.analyst_result)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    render_ai_result_block("2. 위험 신호 정리", st.session_state.risk_result)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    render_ai_result_block("3. 개선 방안", st.session_state.improvement_result)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    render_ai_result_block("4. 종합 인사이트", st.session_state.insight_result)


def render_ai_section(summary_text, metric_risks, combined_risks):
    section_space()
    render_section_header(
        "AI 종합 분석",
        "지표 해석, 위험 신호 정리, 개선 방안, 종합 인사이트를 순차적으로 생성합니다."
    )

    with st.container(border=True):
        top_risks = [r for r in metric_risks if r.get("우선순위") == "🔴 즉시 대응"]

        if top_risks:
            risk_metrics = ", ".join([r["지표"] for r in top_risks])
            st.warning(f"📌 핵심 결론: {risk_metrics} 감소 → 즉각적인 대응이 필요합니다.")

        button_col1, button_col2, spacer_col = st.columns([1, 1, 3])

        with button_col1:
            run_analysis = st.button(
                "AI 상태 진단 시작",
                use_container_width=True,
                type="primary"
            )

        if run_analysis:
            message, level = run_ai_analysis_pipeline(summary_text, metric_risks, combined_risks)
            if message and level == "info":
                st.info("동일한 파일이 감지되어 이전 분석 결과를 재사용했습니다.")
            st.rerun()

        with button_col2:
            if st.session_state.insight_result:
                if st.session_state.ai_pdf_bytes is None:
                    make_pdf = st.button(
                        "진단 결과 PDF 준비",
                        use_container_width=True
                    )

                    if make_pdf:
                        try:
                            st.session_state.ai_pdf_bytes = build_ai_pdf_bytes(
                                st.session_state.analyst_result,
                                st.session_state.risk_result,
                                st.session_state.improvement_result,
                                st.session_state.insight_result,
                            )
                            st.rerun()
                        except Exception as e:
                            st.warning(f"PDF 생성 준비 중 문제가 발생했습니다: {e}")
                else:
                    st.download_button(
                        "진단 결과 PDF 다운로드",
                        data=st.session_state.ai_pdf_bytes,
                        file_name="game_diagnosis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.button(
                    "진단 결과 PDF 다운로드",
                    disabled=True,
                    use_container_width=True
                )

        with spacer_col:
            st.write("")

        render_ai_results_section()


def render_data_preview_section(df):
    pass


def render_dashboard(df):
    cached = build_dashboard_cache(df)

    render_overview_section(df, cached["action_items"], cached["metric_risks"], cached["combined_risks"], cached["summary"])
    render_risk_section(cached["metric_risks"], cached["combined_risks"])
    render_kpi_card_section(cached["summary"])
    render_summary_table_section(cached["summary_df"])
    render_chart_section(df)
    render_ai_section(cached["summary_text"], cached["metric_risks"], cached["combined_risks"])
    # render_data_preview_section(df)


def main():
    inject_custom_css()
    init_session_state()
    render_page_header()

    uploaded_file = st.file_uploader(
        "분석할 CSV 파일을 업로드해 주세요.",
        type=["csv"]
    )

    if not uploaded_file:
        return

    try:
        df, _ = load_uploaded_dataframe(uploaded_file)
    except Exception:
        st.error("CSV 형식을 확인해 주세요. 파일 인코딩이나 구분자, 헤더 구성이 예상과 다를 수 있습니다.")
        return

    if df.empty:
        st.warning("업로드된 CSV에 분석할 데이터가 없습니다.")
        return

    render_dashboard(df)


if __name__ == "__main__":
    main()