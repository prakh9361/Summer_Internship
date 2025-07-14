import re
import csv

def metadata_parse(filename):
    """
    Parse LAMMPS input file to extract voltage and temperature data
    """
    data = []
    frame_count = 0
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split content by 'run' commands to identify different simulation phases
    sections = re.split(r'\nrun\s+\d+', content)
    
    current_volt = 0.0
    current_temp = 300.0  # Default starting temperature
    
    for i, section in enumerate(sections):
        if i == 0:  # Skip the initial setup section
            continue
            
        # Extract voltage from fix 2 line (echemdid)
        volt_match = re.search(r'fix\s+2\s+all\s+echemdid.*?volt\s+([\d.-]+)', section)
        if volt_match:
            target_volt = float(volt_match.group(1))
        else:
            target_volt = current_volt
        
        # Extract temperature from fix 4 line (temp/berendsen)
        temp_match = re.search(r'fix\s+4\s+all\s+temp/berendsen\s+([\d.-]+)\s+([\d.-]+)', section)
        if temp_match:
            target_temp = float(temp_match.group(1))  # First temperature value
        else:
            target_temp = current_temp
        
        # Generate 20 frames for each simulation phase
        for frame_in_phase in range(20):
            # Create filename
            frame_filename = f"frame_{frame_count:06d}.xyz"
            
            # Calculate time (frame 0 = 0, successive frames +0.5)
            time = frame_count * 0.5
            
            # Add data point with only required columns
            data.append({
                'filename': frame_filename,
                'volt_curr': current_volt,
                'temp_curr': current_temp,
                'time': time
            })
            
            frame_count += 1
        
        # Update current values for next phase
        current_volt = target_volt
        current_temp = target_temp
    
    return data

def write_csv(data, output_filename='output.csv'):
    """
    Write parsed data to CSV file with simplified columns
    """
    fieldnames = ['filename', 'volt_curr', 'temp_curr', 'time']
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"CSV file '{output_filename}' created successfully!")
    print(f"Number of frames processed: {len(data)}")

# Example usage
if __name__ == "__main__":
    # Replace 'input.lammps' with your actual LAMMPS file name
    input_file = 'in.temp_1100_pos'
    
    try:
        # Parse the LAMMPS file
        parsed_data = metadata_parse(input_file)
        
        # Write to CSV
        write_csv(parsed_data, 'metadata.csv')
        
        # Display first few rows as preview
        print("\nPreview of parsed data:")
        print("filename,volt_curr,temp_curr,time")
        for i, row in enumerate(parsed_data[:10]):  # Show first 10 rows
            print(f"{row['filename']},{row['volt_curr']},{row['temp_curr']},{row['time']}")
        if len(parsed_data) > 10:
            print("...")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please make sure the LAMMPS input file exists in the current directory.")
    except Exception as e:
        print(f"Error processing file: {e}")