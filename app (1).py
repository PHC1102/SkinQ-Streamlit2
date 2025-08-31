import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# Cấu hình trang
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)

# Lấy API key từ secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# URLs API
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@st.cache_resource
def load_skin_disease_model():
    """Load dinov2-skindisease-finetuned model locally"""
    try:
        processor = AutoImageProcessor.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
        model = AutoModelForImageClassification.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
        return processor, model
    except Exception as e:
        st.error(f"Không thể load model: {str(e)}")
        return None, None

from openai import OpenAI

# Tạo client OpenAI nhưng trỏ sang OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def call_gpt_oss(messages):
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",   # nhớ đúng tên model
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            extra_headers={
                "HTTP-Referer": "https://skinq-app2-f7etnqs3jnubmrwnsjqdrq.streamlit.app",
                "X-Title": "SkinQ Chatbot",
            },
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi gọi API: {str(e)}"
        
def analyze_skin_image(image, processor, model):
    """Phân tích ảnh da liễu bằng model local"""
    try:
        # Preprocess ảnh
        inputs = processor(images=image, return_tensors="pt")
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Lấy top 3 predictions
        top3_indices = torch.topk(predictions, 3).indices[0]
        top3_scores = torch.topk(predictions, 3).values[0]
        
        # Format kết quả
        results = []
        for idx, score in zip(top3_indices, top3_scores):
            label = model.config.id2label[idx.item()]
            results.append({
                'label': label,
                'score': score.item()
            })
        
        return results
    except Exception as e:
        st.error(f"Lỗi khi phân tích ảnh: {str(e)}")
        return None

def format_diagnosis_prompt(predictions):
    """Tạo prompt cho GPT-OSS từ kết quả chẩn đoán"""
    prompt = "Đây là ảnh da liễu.\n\n"
    prompt += "Model thị giác dự đoán:\n"
    
    for i, pred in enumerate(predictions, 1):
        disease = pred['label']
        confidence = pred['score'] * 100
        prompt += f"{i}. {disease}, độ tin cậy {confidence:.1f}%\n"
    
    prompt += "\nGiải thích vì sao model có thể đưa ra dự đoán này dựa trên đặc điểm hình ảnh. Lưu ý: Đây chỉ là dự đoán của AI, không thay thế chẩn đoán y tế chuyên nghiệp."
    
    return prompt

# Load model một lần
processor, model = load_skin_disease_model()

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_diagnosis" not in st.session_state:
    st.session_state.show_diagnosis = False

# Header
st.title("🩺 Medical Chatbot")
st.markdown("---")

# Nút chẩn đoán da liễu
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔬 Chẩn đoán bệnh da liễu", type="primary", use_container_width=True):
        st.session_state.show_diagnosis = True

st.markdown("---")

# Chat interface
chat_container = st.container()
with chat_container:
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image" in message:
                st.image(message["image"], width=300)
            st.markdown(message["content"])

# Hiển thị upload ảnh nếu được kích hoạt
if st.session_state.show_diagnosis:
    if processor is None or model is None:
        st.error("Model chưa được load thành công. Vui lòng kiểm tra lại.")
        st.session_state.show_diagnosis = False
    else:
        st.markdown("### 📷 Upload ảnh da liễu để chẩn đoán")
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh (JPG, JPEG, PNG):",
            type=['jpg', 'jpeg', 'png'],
            key="diagnosis_uploader"
        )
        
        # Tự động phân tích khi có ảnh upload
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Hiển thị ảnh đã upload trong chat
            st.session_state.messages.append({
                "role": "user", 
                "content": "Tôi đã gửi ảnh da liễu để chẩn đoán",
                "image": image
            })
            
            with st.chat_message("user"):
                st.image(image, width=300)
                st.markdown("Tôi đã gửi ảnh da liễu để chẩn đoán")
            
            # Tự động phân tích ngay
            with st.chat_message("assistant"):
                with st.spinner("Đang phân tích ảnh..."):
                    # Phân tích ảnh với model local
                    predictions = analyze_skin_image(image, processor, model)
                    
                    if predictions:
                        # Tạo prompt cho GPT-OSS
                        diagnosis_prompt = format_diagnosis_prompt(predictions)
                        
                        with st.spinner("Đang tạo báo cáo chẩn đoán..."):
                            messages_for_api = [{"role": "user", "content": diagnosis_prompt}]
                            response = call_gpt_oss(messages_for_api)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "Không thể phân tích ảnh này. Vui lòng thử ảnh khác."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Quay lại mode chat thường và reset
            st.session_state.show_diagnosis = False
            st.rerun()

# Chat input - luôn hiển thị (mặc định là chat thường)
if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
    # Thêm tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gọi GPT-OSS
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            messages_for_api = [{"role": msg["role"], "content": msg["content"]} 
                              for msg in st.session_state.messages if "image" not in msg]
            response = call_gpt_oss(messages_for_api)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar với thông tin
with st.sidebar:
    st.markdown("### ℹ️ Thông tin")
    st.markdown("""
    **💬 Chat thường:**
    - Luôn sẵn sàng trò chuyện
    - Sử dụng GPT-OSS
    
    **🔬 Chẩn đoán da liễu:**
    - Nhấn nút "Chẩn đoán bệnh da liễu"
    - Upload ảnh → Tự động phân tích
    - Quay lại chat thường sau khi xong
    
    ⚠️ **Lưu ý quan trọng:**
    Kết quả chỉ mang tính tham khảo, 
    không thay thế chẩn đoán y tế chuyên nghiệp.
    """)
    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Model đang sử dụng:**")
    st.markdown("- Chat: GPT-OSS")
    st.markdown("- Vision: DinoV2 SkinDisease (Local)")
    
    # Hiển thị trạng thái model
    if processor is not None and model is not None:
        st.success("✅ Model đã load thành công")
    else:
        st.error("❌ Model chưa load được")

# CSS tùy chỉnh
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stFileUploader {
        border: 2px dashed #4facfe;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
