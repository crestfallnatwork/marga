# Generated using GEMINI AI 
# SPDX-License-Identifier: Unlicense

"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# BFS Data
bfs_data = {
    'Graph': ['soc-orkut', 'soc-twitter', 'GAP-twitter', 'soc-LiveJournal1'],
    'Pictor': [13, 71, 163, 10],
    'graphrtx(hybrid)': [45, 360, np.nan, 33],
    'graphrtx(optix)': [75, 661, np.nan, 57]
}
df_bfs = pd.DataFrame(bfs_data)

# PageRank Data
pr_data = {
    'Graph': ['soc-orkut', 'soc-twitter', 'GAP-twitter', 'soc-LiveJournal1'],
    'Pictor': [58, 113, 908, 17],
    'graphrtx(hybrid)': [75, 238, np.nan, 36],
    'graphrtx(optix)': [94, 357, np.nan, 40]
}
df_pr = pd.DataFrame(pr_data)

def plot_data(df, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df['Graph']))
    width = 0.25
    
    # Plotting the bars
    rects1 = ax.bar(x - width, df['Pictor'], width, label='Pictor', color='#1f77b4')
    rects2 = ax.bar(x, df['graphrtx(hybrid)'], width, label='graphrtx(hybrid)', color='#ff7f0e')
    rects3 = ax.bar(x + width, df['graphrtx(optix)'], width, label='graphrtx(optix)', color='#2ca02c')
    
    # Titles and Formatting
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Graph'])
    ax.legend()
    
    # Add OOM annotations
    max_y = df['Pictor'].max()
    for i, val in enumerate(df['graphrtx(hybrid)']):
        if np.isnan(val):
            ax.text(i, max_y * 0.05, 'OOM', ha='center', va='bottom', rotation=90, color='red', fontweight='bold')
            
    for i, val in enumerate(df['graphrtx(optix)']):
        if np.isnan(val):
            ax.text(i + width, max_y * 0.05, 'OOM', ha='center', va='bottom', rotation=90, color='red', fontweight='bold')

    # Add numeric data labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate the plots
plot_data(df_bfs, 'BFS Performance on Device 2070 Super', 'bfs_performance.png')
plot_data(df_pr, 'PageRank (1 iteration avg) Performance on Device 2070 Super', 'pagerank_performance.png')
