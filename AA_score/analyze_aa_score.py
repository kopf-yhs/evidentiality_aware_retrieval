import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

MODEL_NAME = 'picl'

#with open(f'../rocketqa/counterfactual_comparison.json','r') as f:
with open(f'../DPR/outputs/compare_cf/{MODEL_NAME}_answer_counterfactual_comparison_percent/counterfactual_comparison.json','r') as f:
    dpr_scores = json.load(f)

with open(f'../DPR/outputs/compare_cf/{MODEL_NAME}_answer_counterfactual_comparison_percent/counterfactual_comparison.json','r') as f:
    picl_scores = json.load(f)

assert 'answer_percentage' in scores[0].keys()

mismatch_count = 0
aur_between = 0
plot_data = list()
for line in tqdm(scores[:-1]):
    if eval(line['answer_percentage']) > 0 and eval(line['answer_percentage']) < 100:
        plot_data.append([eval(line['answer_percentage']), line['answer_sentence_score'], line['counterfactual_score'], line['passage_score'], line['counterfactual_win']])
        if eval(line['answer_percentage']) >= 25.0 and eval(line['answer_percentage']) <= 75.0:
            aur_between += 1
            if line['counterfactual_win']:
                mismatch_count += 1
    else:
        print(line['passage'])
        print(line['answer_sentence'])
plot_data.sort(key=lambda x : x[0], reverse=True)
x1 = np.array([line[0] for line in plot_data if line[4]])
y1 = np.array([line[2] - line[1] for line in plot_data if line[4]])
x2 = np.array([line[0] for line in plot_data if not line[4]])
y2 = np.array([line[2] - line[1] for line in plot_data if not line[4]])

#ax = fig.add_axes([0,0,100,100])

x3 = np.array([line[0] for line in plot_data if line[4]])
x4 = np.array([line[0] for line in plot_data if not line[4]])

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.set_ylim([-25.0, 25.0])
#ax.set_ylim([-25.0, 25.0])
ax1.bar(x2, y2, color=(0.0,0.0,0.0,0.4))
ax1.bar(x1, y1, color='r')
ax2.hist(x4, bins=50, color=(0.0,0.0,0.0,0.4))
ax2.hist(x3, bins=50, color='r')
#plt.plot(x, y2, 'bo--')

print(f'AUR_Q : {mismatch_count / aur_between * 100}')
plt.savefig(f'{MODEL_NAME}_aur_analysis.jpg')

