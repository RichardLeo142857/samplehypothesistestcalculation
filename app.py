import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ================================
# 主函数
# ================================
def main():
    st.title("📊 t 分布统计推断与预测工具")

    st.markdown("""
    本工具可以完成以下功能：
    1. 输入样本数据，计算 **均值 / 方差**  
    2. 绘制 **t 分布图**  
    3. 计算 **总体均值置信区间**  
    4. 输入预测值，判断是否合理（基于预测区间）  
    """)

    # ---------------------------
    # 用户输入数据
    st.subheader("数据输入")
    data_input = st.text_area("请输入数据 (用逗号或空格分隔):", "82, 85, 90, 87, 88, 91, 84")
    try:
        data = [float(x) for x in data_input.replace(",", " ").split()]
    except:
        st.error("❌ 数据输入格式错误，请输入数字，用空格或逗号分隔")
        return

    if len(data) < 2:
        st.error("❌ 至少需要两个数据点")
        return

    # ---------------------------
    # 用户选择置信水平
    alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    conf_choice = st.radio("选择置信水平:", list(alpha_map.keys()), index=1)
    alpha = alpha_map[conf_choice]

    # ---------------------------
    # 用户预测值
    user_prediction = st.number_input("请输入预测值:", value=95.0)

    # ---------------------------
    # 计算结果
    n = len(data)
    mean = np.mean(data)
    S2 = np.var(data, ddof=1)
    S = np.sqrt(S2)
    df = n - 1

    # t 临界值
    t_crit = stats.t.ppf(1 - alpha/2, df)

    # 置信区间
    ci_low = mean - t_crit * S / np.sqrt(n)
    ci_high = mean + t_crit * S / np.sqrt(n)

    # 预测区间
    pred_low = mean - t_crit * S * np.sqrt(1 + 1/n)
    pred_high = mean + t_crit * S * np.sqrt(1 + 1/n)

    # ---------------------------
    # 输出结果
    st.subheader("📌 计算结果")
    st.write(f"样本量 n = {n}")
    st.write(f"样本均值 = **{mean:.4f}**")
    st.write(f"样本方差 = **{S2:.4f}**")
    st.write(f"样本标准差 = **{S:.4f}**")
    st.write(f"自由度 df = {df}")
    st.write(f"{conf_choice} 置信区间 (总体均值) = **({ci_low:.4f}, {ci_high:.4f})**")
    st.write(f"{conf_choice} 预测区间 (新观测值) = **({pred_low:.4f}, {pred_high:.4f})**")

    if pred_low <= user_prediction <= pred_high:
        st.success(f"✅ 用户预测值 {user_prediction} 在预测区间内，合理。")
    else:
        st.error(f"❌ 用户预测值 {user_prediction} 超出预测区间，不合理。")

    # ---------------------------
    # 绘制 t 分布
    st.subheader("📈 t 分布与可视化")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：t 分布
    x = np.linspace(-4, 4, 500)
    t_pdf = stats.t.pdf(x, df)
    ax[0].plot(x, t_pdf, label=f"t-distribution (df={df})")
    ax[0].axvline(-t_crit, color="blue", linestyle="--", label="临界值")
    ax[0].axvline(t_crit, color="blue", linestyle="--")
    t_val = (user_prediction - mean) / (S / np.sqrt(n))
    ax[0].axvline(t_val, color="red", linestyle="-", label=f"预测值 {user_prediction}")
    ax[0].set_title("t 分布概率密度函数")
    ax[0].set_xlabel("t 值")
    ax[0].set_ylabel("概率密度")
    ax[0].grid(True)
    ax[0].legend()

    # 右图：原始数据尺度
    ax[1].axvline(mean, color="black", label=f"均值 {mean:.2f}")
    ax[1].axvspan(ci_low, ci_high, color="blue", alpha=0.3, label="均值置信区间")
    ax[1].axvspan(pred_low, pred_high, color="orange", alpha=0.3, label="预测区间")
    ax[1].axvline(user_prediction, color="red", linestyle="--", label=f"预测值 {user_prediction}")
    ax[1].set_title("原始数据尺度")
    ax[1].set_yticks([])
    min_val = min(data + [pred_low, ci_low, user_prediction])
    max_val = max(data + [pred_high, ci_high, user_prediction])
    ax[1].set_xlim(min_val - 5, max_val + 5)
    ax[1].legend()

    st.pyplot(fig)

# ================================
# 程序入口
# ================================
if __name__ == "__main__":
    main()
