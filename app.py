import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fdm_engine import interpolate_fluid_temp, solve_sphere_fdm, solve_cube_fdm, calculate_lethality

st.set_page_config(page_title="乳品颗粒连续杀菌评估器", layout="wide")

# Custom CSS for FMCG Website Aesthetic
st.markdown("""
<style>
    /* Global Font and Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Input Container Styling */
    div[data-testid="stVerticalBlock"] div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Headers */
    h1 {
        font-weight: 600 !important;
        color: #1a1a24 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 0;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def render_header():
    st.markdown("<h1 style='text-align: center; margin-top: 2rem;'>乳品颗粒连续杀菌评估器</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #7f8c8d; font-weight: 400; font-size: 1rem; margin-bottom: 3rem;'>产品研发中心 工艺研发部 | 柳春洋</h4>", unsafe_allow_html=True)

if not st.session_state['authenticated']:
    render_header()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("🔒 该系统受内部访问控制保护")
        pwd = st.text_input("密码", type="password", placeholder="请输入访问密码", label_visibility="collapsed")
        if st.button("解锁系统", use_container_width=True, type="primary"):
            if pwd == "12345678":
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("密码错误，请重新输入！")
    st.stop()

render_header()

st.header("参数配置区")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    st.subheader("1. 产品基料与流体参数")
    flow_calc_mode = st.radio(
        "停留时间安全系数设定方式：",
        ["📝 经经验/查表选择 (手动)", "🧮 雷诺数精确推导 (自动计算)"]
    )
    
    if "手动" in flow_calc_mode:
        viscosity = st.radio(
            "选择流体体系特性：",
            ["选项A：低粘度/湍流体系 (最快颗粒时间系数 0.85)", "选项B：高粘度/层流体系 (最快颗粒时间系数 0.50)"]
        )
        residence_multiplier = 0.85 if "0.85" in viscosity else 0.50
        st.info(f"💡 当前设定的停留时间安全系数: **{residence_multiplier:.2f}**")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            flow_rate = st.number_input("产能流量 Q (L/h)", value=5000.0, step=100.0)
            pipe_diameter = st.number_input("保持管内径 D (mm)", value=51.0, step=1.0)
        with col_f2:
            density = st.number_input("基料密度 ρ (kg/m³)", value=1030.0, step=10.0)
            fluid_viscosity = st.number_input("动力粘度 μ (mPa·s)", value=50.0, step=5.0)
            
        Q_m3_s = flow_rate / 3600.0 / 1000.0
        D_m = pipe_diameter / 1000.0
        A_m2 = np.pi * (D_m / 2.0)**2
        v_m_s = Q_m3_s / A_m2 if A_m2 > 0 else 0
        Re = (density * v_m_s * D_m) / (fluid_viscosity / 1000.0) if fluid_viscosity > 0 else 0
        
        if Re < 2300:
            flow_status = "层流 (Laminar Flow)"
            residence_multiplier = 0.50
            status_color = "#e67e22" # Orange
        elif Re > 4000:
            flow_status = "湍流 (Turbulent Flow)"
            residence_multiplier = 0.83
            status_color = "#27ae60" # Green
        else:
            flow_status = "过渡流 (Transitional)"
            residence_multiplier = 0.50 # 保守取值
            status_color = "#f1c40f" # Yellow
            
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid {status_color};'>
            <strong>根据流体力学方程计算结果：</strong><br>
            管道内平均流速 $v$: {v_m_s:.2f} m/s <br>
            雷诺数 $Re$: <strong style='color:{status_color};'>{Re:.0f}</strong> ({flow_status})<br>
            中心颗粒极速安全系数推断: <strong>{residence_multiplier:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3. 配方与杀菌目标")
    ph = st.number_input("pH 值 (4.6 为阈值)", value=6.5, step=0.1)
    
    if ph < 3.8:
        target_mode = st.selectbox(
            "🎯 杀菌目标模式 (极酸体系 pH < 3.8)",
            [
                "果汁/饮料轻度巴氏杀菌 (PU_70)",
                "标准冷链巴氏杀菌 (PU)"
            ]
        )
        if "70" in target_mode:
            st.info("💡 极酸环境下芽孢无法萌发，主要杀灭目标为酵母、霉菌、乳酸菌。")
            default_t_ref = 70.0
            default_z_val = 10.0
            target_threshold = 10.0 # 常见果汁 PU70 推荐 10-20
            metric_name = "PU_70"
        else:
            st.info("💡 通用防腐目标，通常冷链或热灌装保护。")
            default_t_ref = 85.0
            default_z_val = 10.0
            target_threshold = 15.0
            metric_name = "PU"
            
    elif ph < 4.6:
        target_mode = st.selectbox(
            "🎯 杀菌目标模式 (高酸体系 3.8~4.6)",
            [
                "常规冷链巴氏杀菌 (PU)",
                "常温果汁/酸奶无菌 (P0)"
            ]
        )
        if "PU" in target_mode:
            st.info("💡 目标：杀灭营养体、酵母/霉菌等。推荐指标 PU ≥ 15")
            default_t_ref = 85.0
            default_z_val = 10.0
            target_threshold = 15.0
            metric_name = "PU"
        else:
            st.info("💡 目标：杀灭高耐热芽孢（如凝结芽孢杆菌、脂环酸芽孢杆菌）。推荐指标 P0 ≥ 15 ~ 30")
            default_t_ref = 90.0
            default_z_val = 10.0
            target_threshold = 15.0 # 常温高酸体系通常需要 15~30 分钟当量
            metric_name = "P0"
            
    else:
        target_mode = st.selectbox(
            "🎯 杀菌目标模式 (中/低酸性 pH ≥ 4.6)",
            [
                "基础商业无菌 (F0)",
                "高抗性热杀灭率 (B*)",
                "化学营养劣变率 (C*)"
            ]
        )
        if "F0" in target_mode:
            st.info("💡 目标：杀灭肉毒梭状芽孢杆菌。推荐指标 F0 ≥ 3.0")
            default_t_ref = 121.1
            default_z_val = 10.0
            target_threshold = 3.0
            metric_name = "F0"
        elif "B*" in target_mode:
            st.info("💡 目标：杀灭嗜热脂肪芽孢杆菌 (UHT)。推荐指标 B* ≥ 1.0")
            default_t_ref = 135.0
            default_z_val = 10.5
            target_threshold = 1.0
            metric_name = "B*"
        else:
            st.info("💡 注意：C* 是限制项，评估风味/色泽破坏。要求 C* ≤ 1.0")
            default_t_ref = 135.0
            default_z_val = 31.4
            target_threshold = 1.0
            metric_name = "C*"
            
    # Save targets for dashboard
    st.session_state['target_threshold'] = target_threshold
    st.session_state['metric_name'] = metric_name
        
    t_ref_col, z_val_col = st.columns(2)
    with t_ref_col:
        t_ref = st.number_input("参考温度 T_ref (℃)", value=default_t_ref, step=0.1)
    with z_val_col:
        z_val = st.number_input("温度系数 Z值 (℃)", value=default_z_val, step=0.1)
        
    with st.expander("📚 常见杀菌目标菌参数参考表"):
        st.markdown("""
        | 体系类型 | 典型目标菌 | 杀菌类型 | T_ref (℃) | Z值 (℃) | 目标评价参考 |
        | :--- | :--- | :--- | :---: | :---: | :---: |
        | **极酸** (pH < 3.8) | 营养体细胞、酵母、普通霉菌 | 轻度巴氏杀菌 | **70.0** | **10.0** | PU_70 ≥ 10 |
        | **高酸** (3.8≤pH<4.6) | 酵母、霉菌、部分耐酸微球菌 | 常规巴氏杀菌 | **85.0** | **10.0** | PU ≥ 15 |
        | **高酸** (常温要求) | 凝结芽孢杆菌、脂环酸芽孢杆菌 | 常温/高热无菌 | **90.0** | **7~10** | **P0 ≥ 15 ~ 30** |
        | **中/低酸** (pH ≥ 4.6) | 肉毒梭状芽孢杆菌 (C. botulinum) | 基础商业无菌 | **121.1** | **10.0** | F0 ≥ 3.0 ~ 6.0 |
        | **中/低酸** (常温要求) | 嗜热脂肪芽孢杆菌 (G. stearo.) | 强化商业无菌 | **121.1/135** | **10.0** | F0 ≥ 10.0~20.0 |
        """)

with input_col2:
    st.subheader("2. 颗粒形态")
    shape = st.radio("颗粒形状：", ["球体 (Sphere)", "正方体 (Cube)"])
    if "Sphere" in shape:
        size_mm = st.number_input("半径 (mm)", min_value=0.1, value=5.0, step=0.5)
    else:
        size_mm = st.number_input("半边长 (mm)", min_value=0.1, value=5.0, step=0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    h = st.number_input("对流换热系数 h (W/(m²·K))", value=300.0, disabled=True)

with input_col3:
    st.subheader(" 物性参数 (自动换算)")
    preset_category = st.selectbox(
        "热扩散率参考值",
        [
            "自定义 (下方填入常数)",
            "常见水果 (苹果/桃等) ≈ 1.35",
            "常见蔬菜 (土豆/萝卜等) ≈ 1.40",
            "水煮谷物 (玉米/燕麦) ≈ 1.15",
            "瘦肉丁 ≈ 1.30",
            "纯水溶液 ≈ 1.43"
        ]
    )
    if "水果" in preset_category: default_alpha = 1.35
    elif "蔬菜" in preset_category: default_alpha = 1.40
    elif "谷物" in preset_category: default_alpha = 1.15
    elif "肉" in preset_category: default_alpha = 1.30
    elif "水" in preset_category: default_alpha = 1.43
    else: default_alpha = 1.40
    alpha_base = st.number_input("热扩散率 α (×10⁻⁷ m²/s)", value=default_alpha, step=0.01)
    alpha = alpha_base * 1e-7

    k_preset_category = st.selectbox(
        "导热系数参考值",
        [
            "自定义 (下方填入常数)",
            "常见水果 (苹果/桃等) ≈ 0.55",
            "常见蔬菜 (土豆/萝卜等) ≈ 0.60",
            "水分较高谷物 ≈ 0.50",
            "瘦肉丁 ≈ 0.50",
            "纯水溶液 ≈ 0.60"
        ]
    )
    if "水果" in k_preset_category: default_k = 0.55
    elif "蔬菜" in k_preset_category: default_k = 0.60
    elif "谷物" in k_preset_category: default_k = 0.50
    elif "肉" in k_preset_category: default_k = 0.50
    elif "水" in k_preset_category: default_k = 0.60
    else: default_k = 0.60
    k = st.number_input("导热系数 k (W/(m·K))", value=default_k, step=0.01)

st.subheader("4. 时间-温度节点录入")
default_data = pd.DataFrame({
    "时间(秒)": [0, 60, 100, 130],
    "温度(℃)": [20, 135, 135, 20]
})
edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

evaluate_btn = st.button("开始杀菌评估", type="primary", use_container_width=True)

st.divider()

st.header("杀菌评估结果看板 (Dashboard)")
if evaluate_btn:
    try:
        clean_df = edited_df.dropna()
        times = pd.to_numeric(clean_df["时间(秒)"]).values
        temps = pd.to_numeric(clean_df["温度(℃)"]).values
        
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        temps = temps[sort_idx]
        
        if len(times) < 2:
            st.error("请至少输入两个时间-温度节点！")
        else:
            interp_times, interp_temps, dt = interpolate_fluid_temp(times, temps, residence_multiplier)
            
            with st.spinner("求解非稳态传热偏微分方程..."):
                if "Sphere" in shape:
                    center_temps, grid_history = solve_sphere_fdm(size_mm, alpha, k, h, interp_temps, dt)
                else:
                    center_temps, grid_history = solve_cube_fdm(size_mm, alpha, k, h, interp_temps, dt)
                    
                PU, F0 = calculate_lethality(center_temps, dt, ph, t_ref, z_val)
                fluid_PU, fluid_F0 = calculate_lethality(interp_temps, dt, ph, t_ref, z_val)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=interp_times, y=interp_temps, mode='lines', 
                                     line=dict(color='blue', width=2, dash='dash'), 
                                     name='流体插值温度 (液相环境)'))
            fig.add_trace(go.Scatter(x=interp_times, y=center_temps, mode='lines',
                                     line=dict(color='red', width=3),
                                     name='颗粒中心温度 (最冷点)'))
            fig.update_layout(title="温度对比曲线 (传热滞后监测)", xaxis_title="颗粒实际经历的最短时间 (s)", yaxis_title="温度 (℃)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("最终致死率诊断")
            metric_name = st.session_state.get('metric_name', 'F0')
            threshold = st.session_state.get('target_threshold', 3.0)
            
            if ph < 4.6:
                value = PU
                fluid_value = fluid_PU
            else:
                value = F0
                fluid_value = fluid_F0
                
            # Special logic for C* (which needs to be <= threshold)
            is_c_star = (metric_name == "C*")
                
            col_res1, col_res2 = st.columns(2)
            
            target_operator = '≤' if is_c_star else '≥'
            
            with col_res1:
                st.markdown(f"**颗粒中心 {metric_name}**  *(目标: {target_operator} {threshold:.1f})*")
                if (not is_c_star and value >= threshold) or (is_c_star and value <= threshold):
                    st.markdown(f"<div style='background-color:#d4edda; padding:15px; border-radius:10px;'><h3 style='color: #155724; text-align: center; margin:0;'>✅ {value:.2f} / {threshold:.1f}</h3></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#f8d7da; padding:15px; border-radius:10px;'><h3 style='color: #721c24; text-align: center; margin:0;'>⚠️ {value:.2f} / {threshold:.1f}</h3></div>", unsafe_allow_html=True)
            with col_res2:
                st.markdown(f"**外部液相环境 {metric_name}** *(目标: {target_operator} {threshold:.1f})*")
                if (not is_c_star and fluid_value >= threshold) or (is_c_star and fluid_value <= threshold):
                    st.markdown(f"<div style='background-color:#d4edda; padding:15px; border-radius:10px;'><h3 style='color: #155724; text-align: center; margin:0;'>✅ {fluid_value:.2f} / {threshold:.1f}</h3></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#f8d7da; padding:15px; border-radius:10px;'><h3 style='color: #721c24; text-align: center; margin:0;'>⚠️ {fluid_value:.2f} / {threshold:.1f}</h3></div>", unsafe_allow_html=True)
                
    except Exception as e:
        import traceback
        st.error(f"计算过程发生阻断错误: {str(e)}")
        st.code(traceback.format_exc(), language='python')
else:
    st.info("👆 请在上方输入或核对物理及环境参数，确认无误后点击「开始杀菌评估」提交流体动力学计算。")
