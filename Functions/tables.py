import pandas as pd
import os

def dataframe(trainer, path):
    # Calculate specific epochs
    num_epochs = len(trainer.histories['loss_history'])
    specific_epochs = [num_epochs // 4, num_epochs // 2, 3 * num_epochs // 4, num_epochs]
    columns = [f"Epoch {epoch}" for epoch in specific_epochs] + [f'Best Epoch ({trainer.best_epoch})']
    full_path_excel = os.path.join(path, 'Losses.xlsx')

    # Prepare data
    data = []
    # First row: Loss values
    row = [f'{trainer.histories['loss_history'][epoch - 1]:.5f}' for epoch in specific_epochs]
    row.append(f"{trainer.best_score:.5f}")
    data.append(row)
    # Create DataFrame with a custom index label
    df = pd.DataFrame(data, columns=columns)
    df.index = ["Error: "]  # Set the index label
    # Save DataFrame to Excel
    df.to_excel(full_path_excel, sheet_name='Loss History', index=True)
    # Print DataFrame to check
    print(df)