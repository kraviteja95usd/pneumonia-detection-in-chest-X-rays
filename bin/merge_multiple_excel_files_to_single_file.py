import os
import pandas as pd


def merge_excel_files():
    # Define the path to the Excel files
    excel_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "<give_your_dir_name_of_existing_excel_files_here>")
    print("excel_file_path - ", excel_file_path)

    # Define the exact file names you want to merge
    file_names = [
        "f1.xlsx",
        "f12xlsx",
        "f3.xlsx",
        "f4.xlsx",
        "f5.xlsx"
    ]

    # Define the sheet names (assuming they are the same across all files)
    sheet_names = [
        "sheet1",
        "sheet2",
        "sheet3",
        "sheet4"
    ]

    # Create an empty dictionary to store the merged data
    merged_data = {
        sheet: pd.DataFrame()
        for sheet in sheet_names
    }

    # Iterate over each file and merge the data
    for file_name in file_names:
        file = os.path.join(excel_file_path, str(file_name))
        print("file - ", file)
        for sheet in sheet_names:
            df = pd.read_excel(file, sheet_name=sheet)
            # Check if the dataframe is not empty before merging
            if not df.empty:
                if merged_data[sheet].empty:
                    merged_data[sheet] = df
                else:
                    merged_data[sheet] = merged_data[sheet].merge(df, how='outer')

    # Create a new Excel file with the merged data
    merged_file_path = os.path.join(excel_file_path, 'merged.xlsx')
    with pd.ExcelWriter(merged_file_path) as writer:
        for sheet, df in merged_data.items():
            df.drop_duplicates(subset='Image Filename', keep='first', inplace=True)
            df.to_excel(writer, sheet_name=sheet, index=False)
