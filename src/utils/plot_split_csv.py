import os
import pandas as pd
import matplotlib.pyplot as plt

def merge_and_sort_csv_files(csv_dir, output_dir, output_file):
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    # Read and concatenate all CSV files
    all_data = pd.concat([pd.read_csv(os.path.join(csv_dir, file)) for file in csv_files])

    # Group by 'Step' and keep the last entry of each group
    filtered_data = all_data.groupby('Step').last().reset_index()

    # # Sort by 'step' column
    # sorted_data = all_data.sort_values(by='Step')

    # Write the sorted data to a new CSV file
    filtered_data.to_csv(f"{output_dir}/{output_file}", index=False)

def plot_csv(input_file, x_value, y_value, x_label, y_label, output_dir, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Calculate a moving average (change 'window_size' as needed)
    window_size = 3
    df['Smoothed'] = df[y_value].rolling(window=window_size, min_periods=1).mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_value], df[y_value], label='Original')
    plt.plot(df[x_value], df['Smoothed'], color='red', label='Smoothed')

    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.title(f'{x_label} vs {y_label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{output_file}")


csv_dir = "csvs/My_UNETR_no_batch_sync"  
output_file = "merged_filtered_data.csv"


for entry in os.listdir(csv_dir):
    full_path = os.path.join(csv_dir, entry)
    # Use os.path.isdir() to check if it's a directory
    if os.path.isdir(full_path):
        print(entry)
        merge_and_sort_csv_files(full_path, csv_dir, f"{entry}.csv")
        plot_csv(f"{csv_dir}/{entry}.csv", 'Step', 'Value', 'Step', entry, csv_dir, f'{entry}.png')


# merge_and_sort_csv_files(csv_dir, output_dir, output_file)
# plot_csv('merged_filtered_data.csv', 'Step', 'Value', 'Step', 'Average Validation Dice', output_dir, 'avg_val_dice.png')
# plot_csv('merged_filtered_data.csv', 'Step', 'Value', 'Step', 'Average Validation Loss', output_dir, 'avg_val_loss.png')