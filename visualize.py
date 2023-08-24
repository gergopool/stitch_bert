import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == "__main__":

    file = "./evaluation.csv"
    vis_performance_per_type(file, "vis")
    vis_performance_per_type(file, "nlp")
    # vis_performance(file)
    vis_avg_performance(file)
    vis_mask_sparsity(file)
    vis_sparsity_per_type(file)

