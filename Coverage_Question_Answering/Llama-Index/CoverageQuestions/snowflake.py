import pandas as pd
from openpyxl import Workbook

def concatenate_body(file_path, output_file='output.xlsx'):
    # Define the order of ticket IDs
    ticket_ids_order = [802591, 802219, 801270, 799775, 795285, 793101, 793028, 791927, 788259, 782471, 779717, 777898, 777763, 777050, 776739, 774236, 774095, 773353, 772190, 772127, 771725, 770885, 769000, 767032, 766181, 765967, 764968, 764623, 763562, 763011, 761336, 760287, 760204, 754970, 754967, 753962, 753374, 753182, 753082, 750903, 750700, 749471, 749232, 746395, 744198, 743502, 743057, 741804, 740964, 739810, 739507, 738588, 738227, 737410, 737210, 736683, 735650, 734419, 733845, 730349, 727009, 723349, 721445, 720017, 719910, 719697, 719480, 718942, 714558, 712117, 709752, 709037, 708509, 707630, 707194, 706201, 703617, 701733, 698082, 697025, 695196, 694541, 694318, 693331, 692724, 692700, 690218, 690109, 689471, 689468, 687702, 687426, 686033, 685851, 685161, 684806, 684210, 682391, 682315, 681489, 681084, 680799, 680617, 679718, 677985, 676697, 672577, 630274]
    print(len(ticket_ids_order))
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Track first occurrences
    first_occurrences = set()

    # Dictionary to concatenate BODY texts for each ticket ID
    concatenated_body_dict = {str(ticket): [] for ticket in ticket_ids_order}

    for index, row in df.iterrows():
        ticket = str(row['TICKET_ID'])
        body = row['BODY']
        
        if ticket in first_occurrences:
            concatenated_body_dict[ticket].append(body)
        else:
            first_occurrences.add(ticket)

    # Concatenate BODY texts with "==============\n"
    concatenated_bodies = {ticket: '\n==============\n'.join(bodies) for ticket, bodies in concatenated_body_dict.items() if bodies}

    # Filter out empty entries and prepare data for output
    result_data = [
        {'TICKET_ID': ticket, 'Concatenated_BODY': body} 
        for ticket, body in concatenated_bodies.items()
    ]

    # Convert to DataFrame
    result_df = pd.DataFrame(result_data)

    # Write to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Concatenated_Bodies', index=False)

    print(f"Concatenated BODY texts have been saved to {output_file}")

if __name__ == "__main__":
    file_path = input("Enter the path to the CSV file: ")
    output_file = input("Enter the name of the output Excel file (default: output.xlsx): ")
    if not output_file.strip():
        output_file = 'output.xlsx'
    
    concatenate_body(file_path, output_file)