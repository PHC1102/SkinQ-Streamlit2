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
        processor = AutoImageProcessor.from_pretrained("dinov2-skindisease-finetuned")
        model = AutoModelForImageClassification.from_pretrained("dinov2-skindisease-finetuned")
        return processor, model
    except Exception as e:
        st.error(f"Không thể load model: {str(e)}")
        return None, None

def call_gpt_oss(messages):
    """Gọi GPT-OSS qua OpenRouter"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-oss",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
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
    
    prompt += "\nGiải thích vì sao model có thể đưa ra dự đoán này dựa trên đặc điểm hình ảnh. Cũng như giải thích lí do độ tin cậy ở mức kia. Lưu ý: Đây chỉ là dự đoán của AI, không thay thế chẩn đoán y tế chuyên nghiệp."
    
    return prompt

# Load model một lần
processor, model = load_skin_disease_model()

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# Header
st.title("🩺 Medical Chatbot")
st.markdown("---")

# Mode selection
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Chế độ hoạt động:")
with col2:
    mode = st.selectbox(
        "Chọn chế độ:",
        ["Chat thường", "Chẩn đoán da liễu"],
        index=0 if st.session_state.mode == "chat" else 1,
        key="mode_selector"
    )
    
    if mode == "Chat thường":
        st.session_state.mode = "chat"
    else:
        st.session_state.mode = "diagnosis"

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

# Input area
if st.session_state.mode == "chat":
    # Chat thường
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

else:
    # Chế độ chẩn đoán da liễu
    if processor is None or model is None:
        st.error("Model chưa được load thành công. Vui lòng kiểm tra lại.")
    else:
        st.markdown("### 📷 Tải ảnh da liễu để chẩn đoán")
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh (JPG, JPEG, PNG):",
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Hiển thị ảnh
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", width=400)
            
            if st.button("🔍 Phân tích ảnh", type="primary"):
                with st.spinner("Đang phân tích ảnh..."):
                    # Phân tích ảnh với model local
                    predictions = analyze_skin_image(image, processor, model)
                    
                    if predictions:
                        # Hiển thị ảnh trong chat
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": "Tôi đã gửi ảnh da liễu để chẩn đoán",
                            "image": image
                        })
                        
                        with st.chat_message("user"):
                            st.image(image, width=300)
                            st.markdown("Tôi đã gửi ảnh da liễu để chẩn đoán")
                        
                        # Tạo prompt cho GPT-OSS
                        diagnosis_prompt = format_diagnosis_prompt(predictions)
                        
                        # Gọi GPT-OSS với prompt chẩn đoán
                        with st.chat_message("assistant"):
                            with st.spinner("Đang phân tích và giải thích..."):
                                messages_for_api = [{"role": "user", "content": diagnosis_prompt}]
                                response = call_gpt_oss(messages_for_api)
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar với thông tin
with st.sidebar:
    st.markdown("### ℹ️ Thông tin")
    st.markdown("""
    **Chế độ Chat thường:**
    - Chat với AI như bình thường
    - Sử dụng GPT-OSS
    
    **Chế độ Chẩn đoán:**
    - Tải ảnh da liễu lên
    - AI sẽ phân tích và đưa ra dự đoán
    - Giải thích chi tiết từ chuyên gia AI
    
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