import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ================================
# Main function
# ================================
def main():
    st.title("🎂Happy 18th Birthday Mr.Lee")

    st.markdown("""
亲爱的 lzh叔叔【笑】，这个工具可以通过你给的样本预测一些数据【我这里用的是 t 分布因此即便你样本数比较小也可以用】。

比方说你可以试着输入你几次数学成绩，然后它会通过这个样本均值和样本方差来呈现你分数分布的 PDF，你可以尝试预估你能考多少分，然后它就会根据你已有的数据来进行假设检验判断你的预估合不合理（不过对你来说 150 可能也是合理的叭 LOL）。

然后你可以自己选择合适的显著性水平 significance level，假设你在卖一个产品你着急说明你的产品有效，你大可把你的 significance level 设得低点 bushi。
""")

    # ---------------------------
    # User input for data
    st.subheader("Data Input")
    data_input = st.text_area("Enter your data (comma or space separated):", "82, 85, 90, 87, 88, 91, 84")
    try:
        data = [float(x) for x in data_input.replace(",", " ").split()]
    except:
        st.error("❌ Invalid data format. Please enter numbers separated by commas or spaces.")
        return

    if len(data) < 2:
        st.error("❌ You need at least two data points.")
        return

    # ---------------------------
    # Confidence level
    alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    conf_choice = st.radio("Select confidence level:", list(alpha_map.keys()), index=1)
    alpha = alpha_map[conf_choice]

    # ---------------------------
    # User prediction
    user_prediction = st.number_input("Enter your predicted value:", value=95.0)

    # ---------------------------
    # Calculations
    n = len(data)
    mean = np.mean(data)
    S2 = np.var(data, ddof=1)
    S = np.sqrt(S2)
    df = n - 1

    t_crit = stats.t.ppf(1 - alpha/2, df)

    # Confidence interval for mean
    ci_low = mean - t_crit * S / np.sqrt(n)
    ci_high = mean + t_crit * S / np.sqrt(n)

    # Prediction interval for new observation
    pred_low = mean - t_crit * S * np.sqrt(1 + 1/n)
    pred_high = mean + t_crit * S * np.sqrt(1 + 1/n)

    # ---------------------------
    # Display results
    st.subheader("📌 Results")
    st.write(f"Sample size n = {n}")
    st.write(f"Sample mean = **{mean:.4f}**")
    st.write(f"Sample variance = **{S2:.4f}**")
    st.write(f"Sample standard deviation = **{S:.4f}**")
    st.write(f"Degrees of freedom df = {df}")
    st.write(f"{conf_choice} Confidence interval (population mean) = **({ci_low:.4f}, {ci_high:.4f})**")
    st.write(f"{conf_choice} Prediction interval (new observation) = **({pred_low:.4f}, {pred_high:.4f})**")

    if pred_low <= user_prediction <= pred_high:
        st.success(f"✅ Your predicted value {user_prediction} is within the prediction interval, reasonable.")
    else:
        st.error(f"❌ Your predicted value {user_prediction} is outside the prediction interval, not reasonable.")

    # ---------------------------
    # Visualization: t-distribution only
    st.subheader("📈 t-Distribution Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))  # Single plot only
    x = np.linspace(-4, 4, 500)
    t_pdf = stats.t.pdf(x, df)
    ax.plot(x, t_pdf, label=f"t-distribution (df={df})")

    # Critical values
    ax.axvline(-t_crit, color="blue", linestyle="--", label="Critical value")
    ax.axvline(t_crit, color="blue", linestyle="--")

    # User prediction
    t_val = (user_prediction - mean) / (S / np.sqrt(n))
    ax.axvline(t_val, color="red", linestyle="-", label=f"Prediction {user_prediction}")

    ax.set_title("t-Distribution PDF")
    ax.set_xlabel("t value")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


# ================================
# Entry point
# ================================
if __name__ == "__main__":
    main()

