import os
import json
import uuid
import math
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

app = Flask(__name__)
app.secret_key = 'business_sim_secret_2024'
app.jinja_env.globals['enumerate'] = enumerate

SCENARIOS_DIR = 'scenarios'
DATA_FILE = 'data/user_progress.json'
IMAGES_DIR = 'static/images'

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)


# ─── Data helpers ────────────────────────────────────────────────────────────

def load_scenarios():
    scenarios = []
    for fname in os.listdir(SCENARIOS_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(SCENARIOS_DIR, fname), 'r', encoding='utf-8') as f:
                scenarios.append(json.load(f))
    return scenarios


def load_scenario(scenario_id):
    path = os.path.join(SCENARIOS_DIR, f'{scenario_id}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_progress():
    if not os.path.exists(DATA_FILE):
        return {"simulations": {}}
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_progress(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_simulation(sim_id):
    progress = load_progress()
    return progress['simulations'].get(sim_id)


def save_simulation(sim_id, sim_data):
    progress = load_progress()
    progress['simulations'][sim_id] = sim_data
    save_progress(progress)


# ─── Simulation logic ─────────────────────────────────────────────────────────

def init_simulation(scenario):
    sim_id = str(uuid.uuid4())[:8]
    ic = scenario['initial_conditions']
    
    sim = {
        'id': sim_id,
        'scenario_id': scenario['id'],
        'scenario_title': scenario['title'],
        'started_at': datetime.now().isoformat(),
        'current_month': 0,
        'budget': ic['budget'],
        'market_share': ic['market_share'],
        'reputation': ic['reputation'],
        'decisions': [],
        'budget_history': [ic['budget']],
        'market_share_history': [ic['market_share']],
        'reputation_history': [ic['reputation']],
        'completed': False,
        'decision_index': 0,
    }
    save_simulation(sim_id, sim)
    return sim_id


def advance_simulation(sim, months=1):
    """Advance simulation by given months, applying passive growth/decay."""
    for _ in range(months):
        sim['current_month'] += 1
        # Passive dynamics each month
        sim['budget'] += int(sim['market_share'] * 8000)  # revenue from market share
        sim['market_share'] = max(0, sim['market_share'] + 0.2)  # slow organic growth
        sim['reputation'] = max(0, min(100, sim['reputation'] - 0.3))  # slight decay

        sim['budget_history'].append(sim['budget'])
        sim['market_share_history'].append(round(sim['market_share'], 1))
        sim['reputation_history'].append(round(sim['reputation'], 1))

    return sim


def apply_decision(sim, scenario, option_index):
    """Apply chosen decision effect and advance to next decision point."""
    decision_points = scenario['decision_points']
    di = sim['decision_index']

    if di >= len(decision_points):
        return sim

    dp = decision_points[di]
    chosen = dp['options'][option_index]
    effect = chosen['effect']

    # Record decision
    sim['decisions'].append({
        'month': dp['month'],
        'question': dp['question'],
        'chosen_text': chosen['text'],
        'rating': chosen['rating'],
        'effect': effect,
    })

    # Advance to decision month
    months_to_advance = dp['month'] - sim['current_month']
    if months_to_advance > 0:
        sim = advance_simulation(sim, months_to_advance)

    # Apply effects
    sim['budget'] = max(0, sim['budget'] + effect['budget'])
    sim['market_share'] = max(0, min(100, sim['market_share'] + effect['market_share']))
    sim['reputation'] = max(0, min(100, sim['reputation'] + effect['reputation']))

    # Update last history entry with post-decision values
    if sim['budget_history']:
        sim['budget_history'][-1] = sim['budget']
        sim['market_share_history'][-1] = sim['market_share']
        sim['reputation_history'][-1] = sim['reputation']

    sim['decision_index'] += 1

    # Check completion
    if sim['decision_index'] >= len(decision_points):
        # Advance to end of simulation
        months_left = scenario['duration'] - sim['current_month']
        if months_left > 0:
            sim = advance_simulation(sim, months_left)
        sim['completed'] = True

    return sim


def calculate_score(sim, scenario):
    """Calculate final score 0-100."""
    ic = scenario['initial_conditions']
    
    # Normalize values relative to initial
    budget_norm = min(100, max(0, (sim['budget'] / max(ic['budget'], 1)) * 50))
    market_norm = min(100, sim['market_share'] * 2)
    rep_norm = min(100, sim['reputation'])

    raw_score = (budget_norm * 0.3) + (market_norm * 0.5) + (rep_norm * 0.2)
    return round(raw_score, 1)


def get_grade(score):
    if score >= 80:
        return ('Отлично', '#22c55e', '🏆')
    elif score >= 60:
        return ('Хорошо', '#3b82f6', '👍')
    elif score >= 40:
        return ('Удовлетворительно', '#f59e0b', '📊')
    else:
        return ('Неудовлетворительно', '#ef4444', '⚠️')


def generate_feedback(sim, scenario):
    decisions = sim['decisions']
    optimal = sum(1 for d in decisions if d['rating'] == 'оптимальное')
    допустимое = sum(1 for d in decisions if d['rating'] == 'допустимое')
    ошибочное = sum(1 for d in decisions if d['rating'] == 'ошибочное')

    recs = []
    if ошибочное > 0:
        recs.append("Избегайте пассивных решений — промедление в бизнесе почти всегда стоит дороже, чем действие.")
    if optimal < len(decisions) // 2:
        recs.append("Анализируйте долгосрочные последствия: краткосрочная экономия часто приводит к большим потерям.")
    if sim['reputation'] < 50:
        recs.append("Репутация — ключевой актив. Инвестируйте в PR и клиентский опыт с самого начала.")
    if sim['market_share'] < 10:
        recs.append("Доля рынка критична: без агрессивного маркетинга на старте сложно занять позицию.")
    if not recs:
        recs.append("Отличная стратегия! Продолжайте балансировать между ростом и устойчивостью.")

    return {
        'optimal': optimal,
        'acceptable': допустимое,
        'mistakes': ошибочное,
        'recommendations': recs,
    }


# ─── Chart generation ─────────────────────────────────────────────────────────

def generate_chart(sim_id, sim_data):
    months = list(range(len(sim_data['budget_history'])))

    budget_norm = [b / max(sim_data['budget_history']) * 100 if max(sim_data['budget_history']) > 0 else 0
                   for b in sim_data['budget_history']]
    market = sim_data['market_share_history']
    reputation = sim_data['reputation_history']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.patch.set_facecolor('#0f172a')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e293b')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.yaxis.label.set_color('#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.grid(True, color='#1e293b', linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    # Plot 1: Budget (normalized) and Reputation
    ax1.plot(months, budget_norm, color='#38bdf8', linewidth=2.5, label='Бюджет (% от макс.)', zorder=3)
    ax1.fill_between(months, budget_norm, alpha=0.15, color='#38bdf8')
    ax1.plot(months, reputation, color='#f472b6', linewidth=2.5, label='Репутация', zorder=3)
    ax1.fill_between(months, reputation, alpha=0.15, color='#f472b6')

    # Mark decision points
    dp_months = [d['month'] for d in sim_data['decisions']]
    for dm in dp_months:
        if dm < len(budget_norm):
            ax1.axvline(x=dm, color='#fbbf24', linewidth=1, linestyle='--', alpha=0.6)

    ax1.set_title('Бюджет и Репутация', color='#e2e8f0', fontsize=12, pad=10, fontweight='bold')
    ax1.set_ylabel('Значение (%)', color='#94a3b8')
    ax1.set_ylim(0, 110)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)

    # Plot 2: Market share
    ax2.plot(months, market, color='#4ade80', linewidth=2.5, label='Доля рынка (%)', zorder=3)
    ax2.fill_between(months, market, alpha=0.2, color='#4ade80')
    for dm in dp_months:
        if dm < len(market):
            ax2.axvline(x=dm, color='#fbbf24', linewidth=1, linestyle='--', alpha=0.6)

    ax2.set_title('Доля рынка', color='#e2e8f0', fontsize=12, pad=10, fontweight='bold')
    ax2.set_xlabel('Месяц симуляции', color='#94a3b8')
    ax2.set_ylabel('Доля рынка (%)', color='#94a3b8')
    ax2.set_ylim(0, max(max(market) * 1.3, 10))
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)

    # Annotation for decision points
    legend_patch = mpatches.Patch(color='#fbbf24', alpha=0.8, label='Точки решений')
    ax2.legend(handles=[ax2.lines[0], legend_patch],
               facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)

    fig.suptitle('Динамика показателей симуляции', color='#f1f5f9', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    chart_path = os.path.join(IMAGES_DIR, f'chart_{sim_id}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    return chart_path


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    scenarios = load_scenarios()
    progress = load_progress()
    completed = [s for s in progress['simulations'].values() if s.get('completed')]
    return render_template('index.html', scenarios=scenarios, completed_count=len(completed))


@app.route('/scenarios')
def scenarios():
    scenarios_list = load_scenarios()
    return render_template('scenarios.html', scenarios=scenarios_list)


@app.route('/simulation/<scenario_id>', methods=['GET', 'POST'])
def simulation(scenario_id):
    scenario = load_scenario(scenario_id)
    if not scenario:
        return redirect(url_for('scenarios'))

    sim_id = request.args.get('sim_id')

    if request.method == 'POST':
        sim_id = request.form.get('sim_id')
        option_index = int(request.form.get('option_index', 0))
        sim = get_simulation(sim_id)

        if sim:
            sim = apply_decision(sim, scenario, option_index)
            save_simulation(sim_id, sim)

            if sim['completed']:
                return redirect(url_for('results', simulation_id=sim_id))

    if not sim_id:
        sim_id = init_simulation(scenario)
        sim = get_simulation(sim_id)
    else:
        sim = get_simulation(sim_id)
        if not sim:
            sim_id = init_simulation(scenario)
            sim = get_simulation(sim_id)

    if sim['completed']:
        return redirect(url_for('results', simulation_id=sim_id))

    # Get current decision point
    di = sim['decision_index']
    decision_points = scenario['decision_points']

    if di >= len(decision_points):
        # Finalize
        sim['completed'] = True
        save_simulation(sim_id, sim)
        return redirect(url_for('results', simulation_id=sim_id))

    current_dp = decision_points[di]
    total_dps = len(decision_points)
    progress_pct = int((di / total_dps) * 100)

    return render_template(
        'simulation.html',
        scenario=scenario,
        sim=sim,
        sim_id=sim_id,
        current_dp=current_dp,
        decision_index=di,
        total_decisions=total_dps,
        progress_pct=progress_pct,
    )


@app.route('/results/<simulation_id>')
def results(simulation_id):
    sim = get_simulation(simulation_id)
    if not sim:
        return redirect(url_for('scenarios'))

    scenario = load_scenario(sim['scenario_id'])
    if not scenario:
        return redirect(url_for('scenarios'))

    # Generate chart
    chart_path = generate_chart(simulation_id, sim)
    chart_url = '/' + chart_path.replace('\\', '/')

    score = calculate_score(sim, scenario)
    grade, grade_color, grade_icon = get_grade(score)
    feedback = generate_feedback(sim, scenario)

    return render_template(
        'results.html',
        sim=sim,
        scenario=scenario,
        chart_url=chart_url,
        score=score,
        grade=grade,
        grade_color=grade_color,
        grade_icon=grade_icon,
        feedback=feedback,
    )


@app.route('/api/simulation_state/<sim_id>')
def simulation_state(sim_id):
    sim = get_simulation(sim_id)
    if not sim:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(sim)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
