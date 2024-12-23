import pandas as pd
import matplotlib.pyplot as plt

def process_and_generate_table(csv_path, output_path, rows=30):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Convert "Date" column to Brazilian date format (dd/mm/yyyy)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')

    # Select the specified number of rows (last `rows` rows)
    df = df.tail(rows)

    # Identify columns with "Diferenca" and format as percentage
    percentage_cols = [col for col in df.columns if "Diff" in col]
    for col in percentage_cols:
        df[col] = df[col].apply(lambda x: f"{x:.2f}%")

    # Print the mean of each "Diff" column (only for the last `rows` rows)
    for col in percentage_cols:
        mean_value = df[col].str.rstrip('%').astype(float).mean()
        print(f"Mean of {col} (last {rows} rows): {mean_value:.2f}%")

    # Identify numeric columns and round to 2 decimal places
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
csv_file_path = "comparacao_TAEE4.csv"  # Replace with your CSV file path
output_image_path = "tabela_final.png"  # Replace with your desired output image path
process_and_generate_table(csv_file_path, output_image_path)
