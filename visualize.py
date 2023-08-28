import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def vis_performance_per_type(file, task_type):
    df = pd.read_csv(file)
    df = df.sort_values(['task', 'seed'], ascending=[True, True])

    filtered_df = df[df['type'] == task_type]
    num_tasks = len(filtered_df['task'].unique())
    num_cols = 3  
    num_rows = (num_tasks - 1) // num_cols + 1  

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 5 * num_rows))
    fig.suptitle(f'Performance per {task_type} task for 5 seeds', fontsize=14)
    tasks = filtered_df['task'].unique()

    row, col = 0, 0

    for i, task in enumerate(tasks):
        ax = axes[row, col]

        task_df = filtered_df[filtered_df['task'] == task]
        orig_acc = task_df['orig']
        masked_acc = task_df['masked']
        retrained_acc = task_df['retrained']
        seeds = task_df['seed']

        bar_width = 0.2
        x = range(len(seeds))
        x_orig = [i - bar_width for i in x]
        x_masked = x
        x_retrained = [i + bar_width for i in x]

        bars_orig = ax.bar(x_orig, orig_acc, width=bar_width, label='Original', color='k', alpha=0.7)
        bars_masked = ax.bar(x_masked, masked_acc, width=bar_width, label='Masked', color='m', alpha=0.7)
        bars_retrained = ax.bar(x_retrained, retrained_acc, width=bar_width, label='Retrained', color='r', alpha=0.7)

        for bar in (bars_orig + bars_masked + bars_retrained):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height-0.2),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        rotation = 90,
                        c = 'w',
                        ha='center', va='bottom', fontsize=8)

        #ax.set_xlabel('Seed')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task}')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)

        col += 1
        if col == num_cols:
            col = 0
            row += 1
    for i in range(len(tasks), num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) 
    plt.subplots_adjust(bottom=0.04) 
    plt.subplots_adjust(right=0.89) 
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def vis_performance(file):
    df = pd.read_csv(file)
    df = df.sort_values(['task','seed'], ascending=[True,True])

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Performance per task', fontsize=20)
    tasks = df['task'].unique()

    for i, task in enumerate(tasks):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        task_df = df[df['task'] == task]
        orig_acc = task_df['orig']
        masked_acc = task_df['masked']
        retrained_acc = task_df['retrained']
        seeds = task_df['seed']

        bar_width = 0.2
        x = range(len(seeds))
        x_orig = [i - bar_width for i in x]
        x_masked = x
        x_retrained = [i + bar_width for i in x]

        bars_orig = ax.bar(x_orig, orig_acc, width=bar_width, label='Original', color='b', alpha=0.7)
        bars_masked = ax.bar(x_masked, masked_acc, width=bar_width, label='Masked', color='g', alpha=0.7)
        bars_retrained = ax.bar(x_retrained, retrained_acc, width=bar_width, label='Retrained', color='r', alpha=0.7)

        for bar in (bars_orig + bars_masked + bars_retrained):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Seed')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task} - {task_df["type"].values[0]}')
        ax.set_yticklabels([])
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend()

    plt.tight_layout()
    plt.show()


def vis_avg_performance(file):
    df = pd.read_csv(file)

    average_accuracy = df.groupby('task')[['orig', 'masked', 'retrained']].mean()
    average_accuracy = average_accuracy.sort_values('orig', ascending=False)

    plt.figure(figsize=(10, 8))
    tasks = average_accuracy.index
    y_pos = np.arange(len(tasks))
    orig_acc = average_accuracy['orig']
    masked_acc = average_accuracy['masked']
    retrained_acc = average_accuracy['retrained']
    bar_width = 0.26

    plt.barh(y_pos - bar_width, orig_acc, bar_width, label='Original', color='b', alpha=0.7)
    plt.barh(y_pos, masked_acc, bar_width, label='Masked', color='g', alpha=0.7)
    plt.barh(y_pos + bar_width, retrained_acc, bar_width, label='Retrained', color='r', alpha=0.7)

    for i, task in enumerate(tasks):
        plt.text(orig_acc[i] + 0.01, y_pos[i] - 1.2*bar_width, f'{orig_acc[i]:.2f}', va='center', fontsize=6, color='b')
        plt.text(masked_acc[i] + 0.01, y_pos[i], f'{masked_acc[i]:.2f}', va='center', fontsize=6, color='g')
        plt.text(retrained_acc[i] + 0.01, y_pos[i] + bar_width, f'{retrained_acc[i]:.2f}', va='center', fontsize=6, color='r')

    plt.yticks(y_pos, tasks)
    plt.xlabel('Accuracy')
    plt.title('Average Performance per Task')
    plt.legend()
    plt.tight_layout()
    plt.show()


def vis_mask_sparsity(file):
    df = pd.read_csv(file)
    average_sparsity = df.groupby('task')['mask_sparsity'].mean()
    average_sparsity = average_sparsity.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))  
    tasks = average_sparsity.index
    y_pos = np.arange(len(tasks))
    mask_sparsity = average_sparsity.values
    bar_width = 0.4  
    plt.barh(y_pos, mask_sparsity, bar_width, color='b', alpha=0.7)

    for i, task in enumerate(tasks):
        plt.text(mask_sparsity[i] + 0.001, y_pos[i], f'{mask_sparsity[i]:.4f}', va='center', fontsize=10, color='b')

    plt.yticks(y_pos, [f'{task}\n ' for task in tasks])
    plt.xlabel('Average Mask Sparsity')
    plt.title('Average Mask Sparsity per task for 5 seeds')
    plt.tight_layout()
    plt.show()


def vis_sparsity_per_type(file):

    df = pd.read_csv(file)
    vision_df = df[df['type'] == 'vis']
    nlp_df = df[df['type'] == 'nlp']

    average_mask_sparsity_vision = vision_df.groupby('task')['mask_sparsity'].mean()
    average_mask_sparsity_nlp = nlp_df.groupby('task')['mask_sparsity'].mean()

    average_mask_sparsity_vision = average_mask_sparsity_vision.sort_values(ascending=False)
    average_mask_sparsity_nlp = average_mask_sparsity_nlp.sort_values(ascending=False)

    _, axs = plt.subplots(1, 2, figsize=(8, 10))  

    def create_bar_plot(ax, average_mask_sparsity, title):
        tasks = average_mask_sparsity.index
        x_pos = np.arange(len(tasks))
        mask_sparsity = average_mask_sparsity.values
        bar_width = 0.3
        ax.bar(x_pos, mask_sparsity, width = bar_width, color='b', alpha=0.7)
        ax.set_title(title)

        for i, sparsity in enumerate(mask_sparsity):
            ax.text(x_pos[i], sparsity + 0.005, f'{sparsity:.4f}', ha='center', fontsize=10, color='b')

        ax.set_xticks(x_pos)
        ax.set_ylabel('Mask sparsity')
        ax.set_yticks(np.arange(0,1.2,0.2))
        ax.set_xticklabels(tasks)

    create_bar_plot(axs[0], average_mask_sparsity_vision, 'Average Mask Sparsity (Vision Tasks) for 5 seeds')
    create_bar_plot(axs[1], average_mask_sparsity_nlp, 'Average Mask Sparsity (NLP Tasks) for 5 seeds')
    plt.tight_layout()
    plt.show()


def vis_sim_per_layer(file):

    df = pd.read_csv(file)
    tasks = df['task1'].unique()
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.5)

    for i, task in enumerate(tasks):
        row = i // 4
        col = i % 4
        
        filtered_df = df[(df['task1'] == task) & (df['task2'] == task)]
        mean_df = filtered_df.groupby("layer")[["jaccard", "cka", "fs"]].mean().reset_index()
        
        ax = axes[row, col]
        ax.plot(range(1, len(mean_df["layer"]) + 1), mean_df["jaccard"], label="Jaccard ")
        ax.plot(range(1, len(mean_df["layer"]) + 1), mean_df["cka"], label="CKA")
        ax.plot(range(1, len(mean_df["layer"]) + 1), mean_df["fs"], label="Functional ")
        ax.set_xlabel("Layer",fontsize=7)
        ax.set_ylabel("Similarity",fontsize=7)
        ax.set_xticks(np.arange(1, 13, 1))
        ax.set_title(f"{task}",fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05) 
    plt.subplots_adjust(right=0.89) 
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def vis_avg_sim_per_type(file):

    df = pd.read_csv(file)
    vis_tasks = ['aircraft','cifar10','cifar100','dtd','flowers','food','pets']
    nlp_tasks = ['mnli','mrpc','qnli','qqp','rte','sst-2','wnli']

    vis_df =  df[df['task1'].isin(vis_tasks) & df['task2'].isin(vis_tasks)]
    nlp_df = df[df['task1'].isin(nlp_tasks) & df['task2'].isin(nlp_tasks)]
    mean_vis_df = vis_df.groupby("layer")[["jaccard", "cka", "fs"]].mean().reset_index()
    mean_nlp_df = nlp_df.groupby("layer")[["jaccard", "cka", "fs"]].mean().reset_index()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].plot(range(1, len(mean_vis_df["layer"]) + 1), mean_vis_df["jaccard"], label="Jaccard", marker='o')
    axes[0].plot(range(1, len(mean_vis_df["layer"]) + 1), mean_vis_df["cka"], label="CKA", marker='o')
    axes[0].plot(range(1, len(mean_vis_df["layer"]) + 1), mean_vis_df["fs"], label="Functional", marker='o')
    axes[0].set_xlabel("Layer")
    axes[0].set_xticks(np.arange(1, 13, 1))
    axes[0].set_ylabel("Similarity")
    axes[0].set_title("Vision Tasks")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(range(1, len(mean_nlp_df["layer"]) + 1), mean_nlp_df["jaccard"], label="Jaccard", marker='o')
    axes[1].plot(range(1, len(mean_nlp_df["layer"]) + 1), mean_nlp_df["cka"], label="CKA", marker='o')
    axes[1].plot(range(1, len(mean_nlp_df["layer"]) + 1), mean_nlp_df["fs"], label="Functional", marker='o')
    axes[1].set_xlabel("Layer")
    axes[1].set_xticks(np.arange(1, 13, 1))
    axes[1].set_ylabel("Similarity")
    axes[1].set_title("NLP Tasks")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def layer_sim_table(file):
    vis_tasks = ['aircraft','cifar10','cifar100','dtd','flowers','food','pets']
    nlp_tasks = ['mnli','mrpc','qnli','qqp','rte','sst-2','wnli']
    df = pd.read_csv(file)
    vis_df =  df[df['task1'].isin(vis_tasks) & df['task2'].isin(vis_tasks)]
    vis_df = vis_df.groupby('layer')[['jaccard', 'cka', 'fs']].agg(['mean', 'std']).reset_index()
    vis_df.columns = ['layer', 'jaccard_mean', 'jaccard_std', 'cka_mean', 'cka_std', 'fs_mean', 'fs_std']
    vis_df['jaccard'] = vis_df.apply(lambda row: f"{row['jaccard_mean']:.2f} ± {row['jaccard_std']:.2f}", axis=1)
    vis_df['cka'] = vis_df.apply(lambda row: f"{row['cka_mean']:.2f} ± {row['cka_std']:.2f}", axis=1)
    vis_df['fs'] = vis_df.apply(lambda row: f"{row['fs_mean']:.2f} ± {row['fs_std']:.2f}", axis=1)
    vis_df.drop(['jaccard_mean', 'jaccard_std'], axis=1, inplace=True)
    vis_df.drop(['cka_mean', 'cka_std'], axis=1, inplace=True)
    vis_df.drop(['fs_mean', 'fs_std'], axis=1, inplace=True)
    vis_df['layer'] = vis_df['layer'] + 1

    nlp_df = df[df['task1'].isin(nlp_tasks) & df['task2'].isin(nlp_tasks)]
    nlp_df = nlp_df.groupby('layer')[['jaccard', 'cka', 'fs']].agg(['mean', 'std']).reset_index()
    nlp_df.columns = ['layer', 'jaccard_mean', 'jaccard_std', 'cka_mean', 'cka_std', 'fs_mean', 'fs_std']
    nlp_df['jaccard'] = nlp_df.apply(lambda row: f"{row['jaccard_mean']:.2f} ± {row['jaccard_std']:.2f}", axis=1)
    nlp_df['cka'] = nlp_df.apply(lambda row: f"{row['cka_mean']:.2f} ± {row['cka_std']:.2f}", axis=1)
    nlp_df['fs'] = nlp_df.apply(lambda row: f"{row['fs_mean']:.2f} ± {row['fs_std']:.2f}", axis=1)
    nlp_df.drop(['jaccard_mean', 'jaccard_std'], axis=1, inplace=True)
    nlp_df.drop(['cka_mean', 'cka_std'], axis=1, inplace=True)
    nlp_df.drop(['fs_mean', 'fs_std'], axis=1, inplace=True)
    nlp_df['layer'] = nlp_df['layer'] + 1

    return vis_df, nlp_df


def performance_table(file,task_type):
    
    df = pd.read_csv(file)
    df = df.sort_values(['task', 'seed'], ascending=[True, True])
    filtered_df = df[df['type'] == task_type]
    filtered_df = filtered_df.groupby('task').agg({
        'orig': ['mean', 'std'],
        'masked': ['mean', 'std'],
        'retrained': ['mean', 'std']
    }).reset_index()
    filtered_df.columns = ['task', 'orig_mean', 'orig_std', 'masked_mean', 'masked_std', 'retrained_mean', 'retrained_std']
    filtered_df['orig'] = filtered_df.apply(lambda row: f"{row['orig_mean']:.2f} ± {row['orig_std']:.2f}", axis=1)
    filtered_df.drop(['orig_mean', 'orig_std'], axis=1, inplace=True)
    filtered_df['masked'] = filtered_df.apply(lambda row: f"{row['masked_mean']:.2f} ± {row['masked_std']:.2f}", axis=1)
    filtered_df.drop(['masked_mean', 'masked_std'], axis=1, inplace=True)
    filtered_df['retrained'] = filtered_df.apply(lambda row: f"{row['retrained_mean']:.2f} ± {row['retrained_std']:.2f}", axis=1)
    filtered_df.drop(['retrained_mean', 'retrained_std'], axis=1, inplace=True)

    return filtered_df


def vis_heatmap(file, task_type):

    df = pd.read_csv(file)
    if task_type == 'nlp':
        tasks = ['mnli','mrpc','qnli','qqp','rte','sst-2','wnli']
    else:
        tasks = ['aircraft','cifar10','cifar100','dtd','flowers','food','pets']

    filtered_df = df[df['task1'].isin(tasks) & df['task2'].isin(tasks)]
    agg_df = filtered_df.groupby(['task1', 'task2']).mean().reset_index()
    jaccard_heatmap = agg_df.pivot('task1', 'task2', 'jaccard')
    cka_heatmap = agg_df.pivot('task1', 'task2', 'cka')
    fs_heatmap = agg_df.pivot('task1', 'task2', 'fs')

    _, axes = plt.subplots(1, 3, figsize=(10, 4))

    sns.heatmap(jaccard_heatmap,  annot=True, fmt=".2f", ax=axes[0], cbar=False)
    sns.heatmap(cka_heatmap,  annot=True, fmt=".2f", ax=axes[1], cbar=False)
    sns.heatmap(fs_heatmap, annot=True, fmt=".2f", ax=axes[2], cbar=False)
    axes[0].set_title('Jaccard Similarity')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Task')
    axes[0].set_ylabel('Task')
    axes[1].set_title('CKA ')
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('Task')
    axes[1].set_ylabel('Task')
    axes[2].set_title('Functional Similarity')
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('Task')
    axes[2].set_ylabel('Task')

    for ax in axes:
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelrotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # file = "./evaluation.csv"
    # vision_table, nlp_table = performance_table(file,'vis'), performance_table(file,'nlp')
    # vis_performance_per_type(file, "vis")
    # vis_performance_per_type(file, "nlp")
    # vis_performance(file)
    # vis_avg_performance(file)
    # vis_mask_sparsity(file)
    # vis_sparsity_per_type(file)
    file = './comparison.csv'
    # vis_sim_per_layer(file)
    # vis_avg_sim_per_type(file)
    # vis_df, nlp_df = layer_sim_table(file)
    vis_heatmap(file,'nlp')