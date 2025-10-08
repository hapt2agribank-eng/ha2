import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
from google.genai import types
import json
from scipy import optimize
import math

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Hiệu quả Dự án Kinh doanh (NPV, IRR, PP)",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Hiệu quả Dự án Kinh doanh 📈")
st.markdown("Sử dụng AI để trích xuất dữ liệu từ văn bản và tính toán các chỉ số NPV, IRR, PP, DPP.")

# Lấy API Key từ Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.warning("Cảnh báo: Không tìm thấy Khóa 'GEMINI_API_KEY'. Vui lòng cấu hình trong Streamlit Secrets để sử dụng chức năng AI.")

# Khởi tạo Session State
if "data_extracted" not in st.session_state:
    st.session_state.data_extracted = None
if "metrics_calculated" not in st.session_state:
    st.session_state.metrics_calculated = None
if "cashflow_df" not in st.session_state:
    st.session_state.cashflow_df = None

# --- Khung JSON Schema cho việc Lọc Dữ liệu (YÊU CẦU 1) ---
EXTRACTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "investment_capital_vnd": types.Schema(
            type=types.Type.NUMBER,
            description="Tổng vốn đầu tư ban đầu của dự án, phải là giá trị số nguyên lớn (Ví dụ: 30000000000, KHÔNG DÙNG đơn vị tỷ)."
        ),
        "project_life_years": types.Schema(
            type=types.Type.INTEGER,
            description="Vòng đời dự án theo năm, phải là số nguyên."
        ),
        "annual_revenue_vnd": types.Schema(
            type=types.Type.NUMBER,
            description="Doanh thu hàng năm, phải là giá trị số nguyên lớn."
        ),
        "annual_cost_vnd": types.Schema(
            type=types.Type.NUMBER,
            description="Tổng chi phí hoạt động hàng năm, phải là giá trị số nguyên lớn."
        ),
        "wacc_rate": types.Schema(
            type=types.Type.NUMBER,
            description="Chi phí vốn bình quân (WACC) dưới dạng tỷ lệ thập phân (Ví dụ: 0.13 cho 13%)."
        ),
        "tax_rate": types.Schema(
            type=types.Type.NUMBER,
            description="Thuế suất thu nhập doanh nghiệp dưới dạng tỷ lệ thập phân (Ví dụ: 0.20 cho 20%)."
        ),
    },
    required=[
        "investment_capital_vnd", "project_life_years", "annual_revenue_vnd", 
        "annual_cost_vnd", "wacc_rate", "tax_rate"
    ]
)

# --- Hàm gọi API Gemini để Lọc Dữ liệu (YÊU CẦU 1) ---
def extract_project_info(project_text, api_key):
    """Sử dụng Gemini để trích xuất các thông số tài chính từ văn bản."""
    if not api_key:
        st.error("Lỗi: Không có Khóa API.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Bạn là một trợ lý phân tích tài chính. Nhiệm vụ của bạn là đọc bản tóm tắt dự án kinh doanh sau và trích xuất sáu thông số tài chính quan trọng. 
        Vốn đầu tư, Doanh thu, và Chi phí phải được trích xuất thành **giá trị số nguyên lớn (VND)**, không dùng đơn vị 'tỷ' hay 'triệu'.
        WACC và Thuế phải được trích xuất thành **tỷ lệ thập phân** (ví dụ: 0.13, 0.20).

        Nội dung dự án:
        ---
        {project_text}
        ---
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EXTRACTION_SCHEMA,
            )
        )
        
        # Parse chuỗi JSON thành dictionary
        data = json.loads(response.text)
        return data

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi: AI trả về định dạng JSON không hợp lệ. Vui lòng thử lại với nội dung rõ ràng hơn.")
        st.code(response.text)
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None

# --- Hàm tính toán Chỉ số Tài chính (YÊU CẦU 3) ---
def calculate_financial_metrics(params):
    """Tính toán NCF, NPV, IRR, PP, và DPP."""
    V0 = params['investment_capital_vnd']
    T = params['project_life_years']
    DT = params['annual_revenue_vnd']
    CP = params['annual_cost_vnd']
    WACC = params['wacc_rate']
    TAX = params['tax_rate']

    # 1. Tính Dòng tiền Thuần Hàng năm (NCF)
    EBT = DT - CP
    Tax_Payment = EBT * TAX if EBT > 0 else 0
    NCF_Annual = EBT - Tax_Payment
    
    # Giả định: Dòng tiền chỉ bắt đầu từ cuối năm 1
    cash_flows = [-V0] + [NCF_Annual] * T

    # 2. Xây dựng Bảng Dòng tiền (YÊU CẦU 2)
    cashflow_data = {
        'Năm': list(range(T + 1)),
        'Vốn Đầu tư (CF0)': [V0] + [0] * T,
        'Dòng tiền Thuần (NCF)': cash_flows,
        'Dòng tiền Chiết khấu': [0] * (T + 1),
        'Dư nợ Chiết khấu': [0] * (T + 1)
    }
    df = pd.DataFrame(cashflow_data)

    # 3. Tính NPV (Net Present Value)
    # np.npv(rate, values) - Lưu ý: V0 phải là phần tử đầu tiên của values
    NPV = np.npv(WACC, cash_flows)
    
    # 4. Tính IRR (Internal Rate of Return)
    try:
        IRR = np.irr(cash_flows)
    except ValueError:
        IRR = np.nan # Không thể tính nếu dòng tiền không đổi dấu

    # 5. Tính PP (Payback Period - Thời gian Hoàn vốn)
    # Giả định dòng tiền đều hàng năm
    PP = V0 / NCF_Annual if NCF_Annual > 0 else float('inf')

    # 6. Tính DPP (Discounted Payback Period - Thời gian Hoàn vốn có Chiết khấu)
    
    # Tính dòng tiền chiết khấu
    for t in range(T + 1):
        if t == 0:
            df.loc[t, 'Dòng tiền Chiết khấu'] = -V0
        else:
            discounted_cf = NCF_Annual / ((1 + WACC) ** t)
            df.loc[t, 'Dòng tiền Chiết khấu'] = discounted_cf
    
    # Tính dư nợ chiết khấu (Cumulative Discounted Cash Flow)
    cumulative_dcf = 0
    for t in range(T + 1):
        cumulative_dcf += df.loc[t, 'Dòng tiền Chiết khấu']
        df.loc[t, 'Dư nợ Chiết khấu'] = cumulative_dcf

    # Tính DPP
    DPP = float('inf')
    if df['Dư nợ Chiết khấu'].max() > 0:
        # Tìm năm đầu tiên dư nợ > 0
        payback_year = df[df['Dư nợ Chiết khấu'] >= 0].index.min()
        
        if payback_year is not np.nan and payback_year > 0:
            # Giá trị âm cuối cùng trước khi hoàn vốn
            prev_cumulative = df.loc[payback_year - 1, 'Dư nợ Chiết khấu']
            # Dòng tiền chiết khấu năm hoàn vốn
            dcf_payback_year = df.loc[payback_year, 'Dòng tiền Chiết khấu']
            
            # Tính phần lẻ: |Giá trị âm cuối cùng| / Dòng tiền chiết khấu năm đó
            fractional_year = -prev_cumulative / dcf_payback_year
            DPP = (payback_year - 1) + fractional_year
    
    # Chuẩn bị kết quả
    metrics = {
        "V0": V0, "T": T, "DT": DT, "CP": CP, "WACC": WACC, "TAX": TAX,
        "NCF_Annual": NCF_Annual,
        "NPV": NPV,
        "IRR": IRR,
        "PP": PP,
        "DPP": DPP
    }

    return metrics, df

# --- Hàm gọi AI để Phân tích Chỉ số (YÊU CẦU 4) ---
def get_ai_analysis_report(metrics, api_key):
    """Gửi các chỉ số đã tính toán đến Gemini để nhận phân tích chuyên sâu."""
    if not api_key:
        return "Lỗi: Không có Khóa API."
        
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Định dạng dữ liệu để AI dễ dàng xử lý
        metrics_text = (
            f"- Vốn Đầu tư (V0): {metrics['V0']:,.0f} VND\n"
            f"- Vòng đời Dự án (T): {metrics['T']} năm\n"
            f"- Chi phí Vốn (WACC): {metrics['WACC']:.2%}\n"
            f"- Dòng tiền Thuần Hàng năm (NCF): {metrics['NCF_Annual']:,.0f} VND\n"
            f"--- KẾT QUẢ ĐÁNH GIÁ HIỆU QUẢ ---\n"
            f"1. **Giá trị Hiện tại Thuần (NPV):** {metrics['NPV']:,.0f} VND\n"
            f"2. **Tỷ suất Hoàn vốn Nội bộ (IRR):** {metrics['IRR']:.2%} (Nếu IRR hợp lệ)\n"
            f"3. **Thời gian Hoàn vốn (PP):** {metrics['PP']:.2f} năm\n"
            f"4. **Thời gian Hoàn vốn có Chiết khấu (DPP):** {metrics['DPP']:.2f} năm\n"
        )
        
        prompt = f"""
        Bạn là một chuyên gia lập dự án kinh doanh. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra một **báo cáo phân tích ngắn gọn, chuyên nghiệp** (khoảng 3-4 đoạn). 
        
        Trong báo cáo, bạn cần:
        1. **Đánh giá NPV:** NPV là dương hay âm? Ý nghĩa.
        2. **Đánh giá IRR:** So sánh IRR với WACC ({metrics['WACC']:.2%}). Dự án có chấp nhận được không?
        3. **Đánh giá Thời gian Hoàn vốn (PP/DPP):** Thời gian hoàn vốn có nằm trong vòng đời dự án ({metrics['T']} năm) không? Rủi ro thanh khoản.

        Đây là các chỉ số chi tiết:
        {metrics_text}
        """

        system_instruction_analysis = (
            "Bạn là chuyên gia lập dự án kinh doanh, phân tích các chỉ số tài chính (NPV, IRR, PP, DPP) một cách khách quan và chuyên nghiệp. "
            "Sử dụng ngôn ngữ rõ ràng, đưa ra kết luận về tính khả thi của dự án."
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction_analysis
            )
        )
        return response.text
        
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình phân tích: {e}"


# --- Giao diện người dùng ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Nhập liệu Phương án Kinh doanh")
    
    # Input cho nội dung Word file
    project_text_input = st.text_area(
        "Dán nội dung Phương án Kinh doanh (từ file Word) vào đây:",
        height=350,
        value="""
        Dự án Đầu tư Dây chuyền Sản xuất Bánh mì.
        Vốn đầu tư ban đầu là 30 tỷ.
        Dự án có vòng đời 10 năm.
        Mỗi năm tạo ra 3,5 tỷ đồng doanh thu, chi phí mỗi năm là 2 tỷ đồng.
        Thuế suất thu nhập doanh nghiệp là 20%.
        WACC của doanh nghiệp là 13%.
        """
    )

    # Nút bấm để thực hiện thao tác lọc dữ liệu
    if st.button("✨ Lọc Dữ liệu và Bắt đầu Tính toán"):
        if api_key and project_text_input:
            with st.spinner('Đang gửi văn bản và chờ AI trích xuất dữ liệu...'):
                extracted_data = extract_project_info(project_text_input, api_key)
                
            if extracted_data:
                st.session_state.data_extracted = extracted_data
                
                # Thực hiện tính toán ngay sau khi trích xuất thành công
                try:
                    metrics, df_cashflow = calculate_financial_metrics(extracted_data)
                    st.session_state.metrics_calculated = metrics
                    st.session_state.cashflow_df = df_cashflow
                    st.success("Trích xuất và Tính toán thành công!")
                except Exception as e:
                    st.error(f"Lỗi tính toán: {e}")
                    st.session_state.metrics_calculated = None
                    st.session_state.cashflow_df = None
            else:
                st.error("Không thể trích xuất dữ liệu. Vui lòng kiểm tra nội dung đầu vào.")
                st.session_state.data_extracted = None
                st.session_state.metrics_calculated = None
                st.session_state.cashflow_df = None
        elif not api_key:
            st.error("Vui lòng cấu hình Khóa API Gemini.")
        else:
            st.warning("Vui lòng nhập nội dung dự án.")

with col2:
    if st.session_state.data_extracted:
        st.subheader("2. Dữ liệu Dự án Đã Lọc (AI)")
        
        # Hiển thị dữ liệu đã lọc
        data = st.session_state.data_extracted
        
        st.markdown(f"""
        | Chỉ tiêu | Giá trị |
        | :--- | :--- |
        | **Vốn Đầu tư (V0)** | {data['investment_capital_vnd']:,.0f} VND |
        | **Vòng đời Dự án** | {data['project_life_years']} năm |
        | **Doanh thu Hàng năm** | {data['annual_revenue_vnd']:,.0f} VND |
        | **Chi phí Hàng năm** | {data['annual_cost_vnd']:,.0f} VND |
        | **WACC (Chiết khấu)** | {data['wacc_rate']:.2%} |
        | **Thuế suất TNDN** | {data['tax_rate']:.0%} |
        """)
        
        # Tính toán NCF hàng năm
        st.info(f"**Dòng tiền Thuần Hàng năm (NCF)** = (Doanh thu - Chi phí) * (1 - Thuế) = **{st.session_state.metrics_calculated['NCF_Annual']:,.0f} VND**")

# --- Bảng Dòng tiền (YÊU CẦU 2) ---
if st.session_state.cashflow_df is not None:
    st.markdown("---")
    st.subheader("3. Bảng Dòng tiền và Hoàn vốn Chiết khấu (Yêu cầu 2)")
    
    st.dataframe(
        st.session_state.cashflow_df.style.format({
            'Vốn Đầu tư (CF0)': '{:,.0f}',
            'Dòng tiền Thuần (NCF)': '{:,.0f}',
            'Dòng tiền Chiết khấu': '{:,.0f}',
            'Dư nợ Chiết khấu': '{:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

# --- Chỉ số Đánh giá Hiệu quả (YÊU CẦU 3) ---
if st.session_state.metrics_calculated:
    st.markdown("---")
    st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án (Yêu cầu 3)")
    metrics = st.session_state.metrics_calculated
    
    col_npv, col_irr, col_pp, col_dpp = st.columns(4)
    
    with col_npv:
        st.metric(
            label="Giá trị Hiện tại Thuần (NPV)",
            value=f"{metrics['NPV']:,.0f} VND",
            delta="Chấp nhận" if metrics['NPV'] > 0 else "Từ chối"
        )
    with col_irr:
        # Kiểm tra tính hợp lệ của IRR
        irr_value = f"{metrics['IRR']:.2%}" if not np.isnan(metrics['IRR']) else "N/A"
        irr_delta = None
        if not np.isnan(metrics['IRR']):
            irr_delta = "Chấp nhận" if metrics['IRR'] > metrics['WACC'] else "Từ chối"
            
        st.metric(
            label="Tỷ suất Hoàn vốn Nội bộ (IRR)",
            value=irr_value,
            delta=irr_delta
        )
    with col_pp:
        st.metric(
            label="Thời gian Hoàn vốn (PP)",
            value=f"{metrics['PP']:.2f} năm"
        )
    with col_dpp:
        st.metric(
            label="Thời gian Hoàn vốn có Chiết khấu (DPP)",
            value=f"{metrics['DPP']:.2f} năm"
        )

    # --- Phân tích AI (YÊU CẦU 4) ---
    st.markdown("---")
    st.subheader("5. Phân tích Chuyên sâu từ AI (Yêu cầu 4)")
    
    if st.button("Yêu cầu AI Phân tích Hiệu quả"):
        if api_key:
            with st.spinner("Đang gửi chỉ số và chờ Gemini phân tích..."):
                analysis_report = get_ai_analysis_report(metrics, api_key)
            
            st.markdown(analysis_report)
        else:
            st.error("Vui lòng cấu hình Khóa API Gemini để sử dụng chức năng này.")

# Footer
st.markdown("---")
st.caption("Ứng dụng được phát triển bởi Gemini dựa trên Streamlit và Gemini API để phân tích dự án kinh doanh.")
