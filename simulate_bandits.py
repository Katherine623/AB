import numpy as np
import matplotlib.pyplot as plt

# 解決 matplotlib 中文表示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ----- 問題設定 -----
bandits = {'Bandit A': 0.8, 'Bandit B': 0.7, 'Bandit C': 0.5}
keys = list(bandits.keys())
means = list(bandits.values())
optimal_mean = max(means)
total_budget = 10000

# =========================================================================
# 任務 1：各個機率分配的獨立測試 (對應您照片中的 A/B Test Simulation 基礎機率分佈)
# =========================================================================
def plot_independent_running_avg(budget=3500):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#d6751c', '#2ca02c']
    
    for idx, (name, mean) in enumerate(bandits.items()):
        rewards = np.random.binomial(1, mean, budget)
        running_avg = np.cumsum(rewards) / np.arange(1, budget + 1)
        plt.plot(running_avg, label=name, color=colors[idx])
        
    plt.title("Independent Arm Pulling: Average Return vs Dollars Spent")
    plt.xlabel("Dollars Spent per Bandit")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig("1_independent_running_avg.png")
    plt.close()

# =========================================================================
# 任務 2：這 6 種多臂演算法的具體實作
# =========================================================================

# 1. A/B 測試 (Explore-then-Exploit)
def simulate_ab_test(budget):
    cumulative_regret = np.zeros(budget)
    rewards_history = np.zeros(budget)
    
    # 依題意：前 2000 平均分給 A 與 B (各 1000 次)
    explore_budget = 2000
    pulls_A = explore_budget // 2
    pulls_B = explore_budget // 2
    
    emp_A, emp_B, emp_C = 0, 0, 0  # 紀錄估計的平均值
    
    for t in range(budget):
        if t < pulls_A:
            action = 0 # 拉動 A
        elif t < explore_budget:
            action = 1 # 拉動 B
        else:
            if t == explore_budget: # 計算探索結束後的真實經驗值
                emp_A = np.sum(rewards_history[:pulls_A]) / pulls_A
                emp_B = np.sum(rewards_history[pulls_A:explore_budget]) / pulls_B
                emp_C = 0 # 題目指定 A/B 階段忽略 C
                
            # 開發階段：選擇經驗中最好的人
            action = 0 if emp_A > emp_B else 1
            
        reward = np.random.binomial(1, means[action])
        rewards_history[t] = reward
        cumulative_regret[t] = optimal_mean - means[action] + (cumulative_regret[t-1] if t > 0 else 0)
        
    return cumulative_regret, rewards_history, [emp_A, emp_B, emp_C]

# 2. 樂觀初始值
def simulate_optimistic(initial_value, budget):
    q_values = np.full(3, initial_value, dtype=float)
    counts = np.zeros(3)
    cumulative_regret = np.zeros(budget)
    
    for t in range(budget):
        action = np.argmax(q_values)
        reward = np.random.binomial(1, means[action])
        cumulative_regret[t] = optimal_mean - means[action] + (cumulative_regret[t-1] if t > 0 else 0)
        
        counts[action] += 1
        q_values[action] += (reward - q_values[action]) / counts[action]
        
    return cumulative_regret

# 3. eps-Greedy
def simulate_eps_greedy(epsilon, budget):
    q_values = np.zeros(3)
    counts = np.zeros(3)
    cumulative_regret = np.zeros(budget)
    
    for t in range(budget):
        if np.random.rand() < epsilon:
            action = np.random.randint(3)
        else:
            action = np.argmax(q_values)
        
        reward = np.random.binomial(1, means[action])
        cumulative_regret[t] = optimal_mean - means[action] + (cumulative_regret[t-1] if t > 0 else 0)
        
        counts[action] += 1
        q_values[action] += (reward - q_values[action]) / counts[action]
        
    return cumulative_regret

# 4. Softmax
def simulate_softmax(temperature, budget):
    q_values = np.zeros(3)
    counts = np.zeros(3)
    cumulative_regret = np.zeros(budget)
    
    for t in range(budget):
        exp_values = np.exp((q_values - np.max(q_values)) / temperature)
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(3, p=probs)
        
        reward = np.random.binomial(1, means[action])
        cumulative_regret[t] = optimal_mean - means[action] + (cumulative_regret[t-1] if t > 0 else 0)
        
        counts[action] += 1
        q_values[action] += (reward - q_values[action]) / counts[action]
        
    return cumulative_regret

# 5. UCB
def simulate_ucb(budget):
    q_values = np.zeros(3)
    counts = np.zeros(3)
    cumulative_regret = np.zeros(budget)
    
    for t in range(budget):
        if t < 3:
            action = t  
        else:
            ucb_values = q_values + np.sqrt(2 * np.log(t) / counts)
            action = np.argmax(ucb_values)
            
        reward = np.random.binomial(1, means[action])
        cumulative_regret[t] = optimal_mean - means[action] + (cumulative_regret[t-1] if t > 0 else 0)
        
        counts[action] += 1
        q_values[action] += (reward - q_values[action]) / counts[action]
        
    return cumulative_regret

# 6. Thompson Sampling
def simulate_thompson(budget):
    successes = np.zeros(3)
    failures = np.zeros(3)
    cumulative_regret = np.zeros(budget)
    
    for t in range(budget):
        samples = [np.random.beta(successes[i] + 1, failures[i] + 1) for i in range(3)]
        action = np.argmax(samples)
        
        reward = np.random.binomial(1, means[action])
        if reward == 1:
            successes[action] += 1
        else:
            failures[action] += 1
            
        cumulative_regret[t] = optimal_mean - means[action] + (cumulative_regret[t-1] if t > 0 else 0)
        
    return cumulative_regret

# =========================================================================
# 執行模擬與繪製圖表
# =========================================================================
n_runs = 10
explore_budget = 2000

regret_ab = np.zeros(total_budget)
regret_opt = np.zeros(total_budget)
regret_eps = np.zeros(total_budget)
regret_softmax = np.zeros(total_budget)
regret_ucb = np.zeros(total_budget)
regret_ts = np.zeros(total_budget)

# 用來存放 A/B Test 的歷史記錄，以繪製如同您照片中的專屬圖表
ab_runs_history = np.zeros((n_runs, total_budget))
ab_last_est_means = None

for i in range(n_runs):
    # A/B Test 特別需要接收 Reward History 與估算平均以畫照片上的圖
    r_ab, rews_ab, est_m = simulate_ab_test(total_budget)
    regret_ab += r_ab
    ab_runs_history[i] = np.cumsum(rews_ab) / np.arange(1, total_budget + 1)
    if i == 0:
        ab_last_est_means = est_m
        
    regret_opt += simulate_optimistic(initial_value=5.0, budget=total_budget)
    regret_eps += simulate_eps_greedy(epsilon=0.1, budget=total_budget)
    regret_softmax += simulate_softmax(temperature=0.1, budget=total_budget)
    regret_ucb += simulate_ucb(total_budget)
    regret_ts += simulate_thompson(total_budget)

# 變為平均悔恨值
regret_ab /= n_runs
regret_opt /= n_runs
regret_eps /= n_runs
regret_softmax /= n_runs
regret_ucb /= n_runs
regret_ts /= n_runs

# ====== 畫圖 1：產生各老虎機分開獨立的流動平均圖 (對應照片 A/B Test Simulation) ======
plot_independent_running_avg(3500)

# ====== 畫圖 2：針對您的 A/B Test (Explore-Then-Exploit) 生成詳細追蹤圖 (兩張合併) ======
mean_running_avg = np.mean(ab_runs_history, axis=0)
std_running_avg = np.std(ab_runs_history, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Explore-then-Exploit Bandit Simulation\nExplore: $2,000 | Exploit: $8,000 | Total Budget: $10,000", fontweight='bold')

# 左圖: 平均回報率與標準差
ax1.plot(mean_running_avg, label=f"Avg Return ({n_runs} runs)", color='#377eb8', linewidth=1.5)
ax1.fill_between(range(total_budget), 
                 mean_running_avg - std_running_avg, 
                 mean_running_avg + std_running_avg, 
                 color='#377eb8', alpha=0.3, label="±1 Std Dev")

ax1.axvspan(0, explore_budget, color='orange', alpha=0.08, label="Explore Phase ($2,000)")
ax1.axvspan(explore_budget, total_budget, color='green', alpha=0.05, label="Exploit Phase ($8,000)")
ax1.axhline(0.8, color='red', linestyle=':', linewidth=1, label="Best Bandit True Mean (0.8)")
ax1.axvline(explore_budget, color='orange', linestyle='--', linewidth=1.5)

ax1.set_title("Cumulative Average Return vs. Dollars Spent")
ax1.set_xlabel("Dollars Spent (Total Budget)")
ax1.set_ylabel("Average Return per Dollar")
ax1.set_xlim(0, total_budget)
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize='small', loc='upper right')

# 右圖: 真實平均與探索後估算平均
x = np.arange(len(keys))
width = 0.35

ax2.bar(x - width/2, means, width, label='True Mean', color='#5c8cbc')
ax2.bar(x + width/2, ab_last_est_means, width, label='Estimated Mean\n(after $2,000 explore)', color='#fca311')

for i in range(len(keys)):
    ax2.text(x[i] - width/2, means[i] + 0.02, f"{means[i]:.2f}", ha='center', va='bottom', fontsize=9)
    # 避免顯示沒有被探索機器的醜陋數值，將忽略的C設定為 0 不標示，或據實標示
    if ab_last_est_means[i] > 0:
        ax2.text(x[i] + width/2, ab_last_est_means[i] + 0.02, f"{ab_last_est_means[i]:.2f}", ha='center', va='bottom', fontsize=9)

ax2.set_title("True vs. Estimated Bandit Means\n(After Exploration Phase)")
ax2.set_ylabel("Mean Return")
ax2.set_xticks(x)
ax2.set_xticklabels(["Bandit A", "Bandit B", "Bandit C"])
ax2.set_ylim(0, 1.1)
ax2.grid(True, axis='y', alpha=0.4)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("2_explore_exploit_diagram.png")
plt.close()

# ====== 畫圖 3：6 種策略的累計悔恨值比較圖 (原本的圖) ======
plt.figure(figsize=(12, 7))
plt.plot(regret_ab, label="A/B 測試 (A/B Testing)")
plt.plot(regret_opt, label="樂觀初始值 (Optimistic Initial Values, Q=5.0)")
plt.plot(regret_eps, label="$\epsilon$-貪婪 ($\epsilon$-Greedy, 0.1)")
plt.plot(regret_softmax, label="Softmax (Boltzmann, $\\tau=0.1$)")
plt.plot(regret_ucb, label="信賴區間上界 (UCB)")
plt.plot(regret_ts, label="湯普森抽樣 (Thompson Sampling)")
plt.xlabel("步數 (總預算)")
plt.ylabel("累計悔恨值 (Cumulative Regret)")
plt.title("6 種多臂老虎機策略累計悔恨值比較")
plt.legend()
plt.grid(True)
plt.savefig("3_bandit_strategies_regret.png")
plt.close()

print("模擬完成！已根據程式碼演算結果產出 3 張對應的統計圖表。")
