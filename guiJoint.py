import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import csv
import pandas as pd
import sys
from scipy import signal
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
from PIL import Image, ImageTk

def create_gui():
    def filter_type_callback(*args):
        filter_type = filter_type_var.get()
        if filter_type in ['cheby1', 'ellip']:
            e8.configure(state='normal')
        else:
            e8.delete(0, 'end')
            e8.configure(state='disabled')
        if filter_type in ['cheby2', 'ellip']:
            e9.configure(state='normal')
        else:
            e9.delete(0, 'end')
            e9.configure(state='disabled')

    def b_type_callback(*args):
        b_type = b_type_var.get()
        if b_type == 'lowpass' or b_type == 'highpass':
            e5.configure(state='normal')
            e6.configure(state='disabled')
            e7.configure(state='disabled')
        elif b_type == 'bandpass' or b_type == 'bandstop':
            e5.configure(state='disabled')
            e6.configure(state='normal')
            e7.configure(state='normal')

    def generate_csv(filter_type, b_type, Fs, fc, rp, rs, order, filename):

        cwd = os.getcwd()
        Td = 1/Fs
        wd = np.multiply(fc,2*np.pi)
        wc = np.multiply(2/Td,np.tan(np.multiply(wd,Td/2)))
        z, p, k=signal.iirfilter(order, wc, rp, rs, btype=b_type, analog=True, ftype=filter_type, output='zpk')
        # Generate prewarped analog second order stages
        sos2_a = signal.zpk2sos(z, p, k, pairing='nearest')
        b,a=signal.zpk2tf(z,p,k)
        # Get response for plotting
        

        w, h = signal.freqs(b, a,worN=np.linspace(np.mean(wc)/100, np.mean(wc)*100, 2000))

        gain = 20*np.log10((np.abs(h)))
        phase = np.angle(h)
        frequency = w/(2*np.pi)
        # Detect analog s domain section Q values
        Q_s2 = np.sqrt(sos2_a[:, 5])/sos2_a[:, 4]

        # Do bilinear transform to z domain
        Zd, Pd, Kd = signal.bilinear_zpk(z, p, k, Fs)

        # Second order sections of z domain part
        sos2_2 = signal.zpk2sos(Zd, Pd, Kd, pairing='nearest')


        # find scale part and replace with 1 2 1 in low pass case of z domain parts

        sosScale=np.power(sos2_2[0, 0],1/np.shape(sos2_2)[0])

        sosReplace_2 = (sos2_2[1, 0:3])[np.newaxis]
        sos2_2[:, 0:3] = sosReplace_2*sosScale

        # Second order stage form must be in this format
        #
        # (a_2*z^{-2}+a_1*z^{-1}+a_0 )/ (b_2*z^{-2}+b_1*z^{-1}+1)
        #
        # So that last term of the sos array scaled to 1, then scale sos2_2[:,3:5]
        # accordingly
 
        sos2_2_scaled = sos2_2

        # Check whether Qs2 is less than 3, if so, do low Q case, if not, do high Q case for for loop
        # Scale the sos2_2[:,3:5] accordingly


        for x in range(len(sos2_2)):
            if Q_s2[x]<3:
                sos2_2_scaled[x, 0:6] = sos2_2[x, 0:6]/sos2_2[x, 5]
            else:
                sos2_2_scaled[x, 0:6] = sos2_2[x, 0:6]/sos2_2[x, 3]


        # Check and calculate low and high Q cases, 
        # Forms are in the book Roubik Gregorian, Gabor C Temes - 
        # Analog MOS Integrated Circuits for Signal Processing (1986)
        # Eq 5.40 (low Q) and Eq 5.46 (high Q)
        # Initialize two DataFrames with the cap_list as the index
        cap_list = ['cVal1', 'c_1p', 'cVal2', 'cVal3', 'cVal4', 'cVal5', 'cVal6', 'cVal7']
        df_a = pd.DataFrame(index=cap_list)
        df_b = pd.DataFrame(index=cap_list)


        for x in range(len(sos2_2_scaled)):
            # Generate multiple csv's
            # dataHeaders = [['Corner', 'bilinearTR_o2_'+str(x+1)], ['Enable', 't']]
            res_list=[]
            val_list=[]
            if Q_s2[x]<3:
                # Low Q case, Eq 5.40 add calculations

                # 
                # Give a nominal capacitance value

                c_Op1=1
                c_Op2=1
                # C_1''=a_0
                c_1pp=sos2_2_scaled[x,2]
                # C_1'=a_2-C_1''
                c_1p=sos2_2_scaled[x,0]-c_1pp
                # C_4=b_2-1
                c_4=sos2_2_scaled[x,3]-1
                # C_2*C_3=b_1+b_2+1, make C_2=C_3 so that you would get minimum spread
                c_2=np.sqrt(sos2_2_scaled[x,3]+sos2_2_scaled[x,4]+1)
                c_3=c_2
                # C_1=(a_0+a_1+a_2)/C_3
                c_1=(sos2_2_scaled[x,0]+sos2_2_scaled[x,1]+sos2_2_scaled[x,2])/c_3
                

            else:
                # High Q 

                c_Op1=1
                c_Op2=1
                # C_1''=a_2/b_2
                c_1pp=sos2_2_scaled[x,0]/sos2_2_scaled[x,3]
                # C_2*C_3=(b_1+b_2+1)/b_2, make C_2=C_3 so that you would get minimum spread
                c_2=np.sqrt((sos2_2_scaled[x,4]+sos2_2_scaled[x,3]+1)/sos2_2_scaled[x,3])
                c_3=c_2
                # C_4=(1-1/b_2)/sqrt((b_1+b_2+1)/b_2)
                c_4=(1-1/sos2_2_scaled[x,3])/(c_2)

                # C_1'=a_2-C_1''
                c_1p=(sos2_2_scaled[x,0]-sos2_2_scaled[x,2])/(c_3*sos2_2_scaled[x,3])
                # C_1=(a_0+a_1+a_2)/C_3
                c_1=(sos2_2_scaled[x,0]+sos2_2_scaled[x,1]+sos2_2_scaled[x,2])/(c_3*sos2_2_scaled[x,3])
                



            # cap_list=['cVal1','c_1p','cVal2','cVal3','cVal4','cVal5','cVal6','cVal7']
            val_list.append([c_1,c_1p,c_2,c_Op1,c_4,c_3,c_1pp,c_Op2])
            # Build a Series with the data and append it as a column to df    # Build a Series with the data
            data_series = pd.Series(val_list[-1], index=cap_list)
                # Append the data to the correct DataFrame based on the value in Qs_2
            if Q_s2[x] <3:
                df_a['name'+str(x+1)] = data_series
            else:
                df_b['name'+str(x+1)] = data_series

            for i in range(len(cap_list)):
                    
                # Old one
                # res_list.append(cap_list[i]+'_'+str(x+1))
                
                # Generates c_1 suffixes only
                res_list.append(cap_list[i])
                


            


        # Transpose the final DataFrame and write to a CSV file
        # Prepare the headers
        # Prepare the headers
        headers = [['Corner'] + [filename+'_lowQ_'+str(i+1) for i in range(len(df_a.columns))],
                ['Enable'] + ['t' for _ in range(len(df_a.columns))]]


        # Write the headers and DataFrame df_a to the CSV file
        with open(os.path.join(cwd, filename+'_combined_data_a.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(headers)
        df_a.to_csv(os.path.join(cwd, filename+'_combined_data_a.csv'), mode='a', header=False)

        # Update the headers for df_b
        headers[0] = ['Corner'] + [filename+'_highQ_'+str(i+1) for i in range(len(df_b.columns))]
        headers[1] = ['Enable'] + ['t' for _ in range(len(df_b.columns))]

        # Write the headers and DataFrame df_b to another CSV file
        with open(os.path.join(cwd, filename+'_combined_data_b.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(headers)
        df_b.to_csv(os.path.join(cwd, filename+'_combined_data_b.csv'), mode='a', header=False)



        return frequency, gain, phase

    def run_program():
        def calc_cost(x, arr):

            zeroremoved_arr = arr[arr != 0]
            rounded_arr = np.round(x * zeroremoved_arr)
            errors = np.square(rounded_arr - x * zeroremoved_arr) / \
                (x * zeroremoved_arr)
            error_cost = np.sum(errors)
            orig_arr = np.round(x * arr)

            return error_cost, orig_arr


        def scale_inputs(inputs):
            # Scale the inputs to the minimum value of each column ignoring zeros
            inputs_wo_zero = np.where(inputs != 0.0, inputs, np.nan)
            input_min_vals = np.nanmin(inputs_wo_zero, axis=0)
            scaled_columns = inputs / input_min_vals
            return scaled_columns
            file_path = file_path_entry.get()
            grid_range = tuple(float(x) for x in grid_range_entry.get().split())
            column_header_prefix = column_header_prefix_entry.get()
            # Then you can proceed with your code, using these variables instead of inputs
        
        def optimize_capacitor_values(scaled_coef, grid_range):
            # Then for capacitance minimization, we need to find the minimum value
            # of the coefficients in the opamp inputs
            # OP1 inputs c_1_1*c_a, c_2_1*c_a, c_a
            # first group to scale within:
            # cVal1
            # cVal2
            # cVal3
            op1_inputs = scale_inputs(scaled_coef[:3, :])
            # OP2 inputs c_4_1*c_a, c_3_1*c_a, c_b, c_1pp_1*c_a
            # second group to scale within:
            # cVal4
            # cVal5
            # cVal6
            # cVal7
            op2_inputs = scale_inputs(scaled_coef[-4:, :])
            # Horizontally join OP1_scaled_columns and OP2_scaled_columns
            joined_columns = np.vstack((op1_inputs, op2_inputs))
            # Joined_columns are the scaled capacitor values for dynamic range
            # and spread minimization. They will be used for the integer
            # optimization

            # Optimized
            # Get column number of joined_columns, it should be same as the number
            # of corners simulated in ADEXL output
            colNum = joined_columns.shape[1]
            # Ask user for the range of values to search for the minimum cost
            # function.
            # Make a grid of values to search for the minimum cost function
            grid = np.arange(grid_range[0], grid_range[1], 0.01)

            # Initialize the output array for the cost function
            output = np.empty(len(grid))

            # Initialize the dataframe for the error and integer scaled capacitor
            # values
            df_error = pd.DataFrame(columns=['error', 'int_cap'])

            # Define the array and range of scalar values to iterate over the
            # joined_columns array by using colNum in a for loop
            for i in range(colNum):
                # Define the array to iterate over the joined_columns array
                arr = joined_columns[:, i]

                # Find the minimum cost function value and corresponding integer
                # scaled capacitor values in the grid
                # TODO: do this with a minimization function built in in scipy package later

                # Cost function returns least squared fit errors in percentage for
                # the specific scales that comes from the grid and the sum of the absolute values of the scaled
                # capacitor values
                errors, int_arrs = zip(*[calc_cost(g, arr) for g in grid])

                # Output array is the cost function values for the specific grid
                # values and the sum of the absolute values of the scaled capacitor
                # values for the specific grid values multiplied
                output = np.array(errors) * \
                    np.array([np.sum(int_arr) for int_arr in int_arrs])

                # Find the index of the minimum value of the output array
                min_idx = np.argmin(output)

                # Find the grid value that gives the minimum cost function value
                min_grid = grid[min_idx]

                # Find the minimum cost function error value and corresponding
                # integer scaled capacitor values
                min_error, min_int_arr = calc_cost(min_grid, arr)

                # Register min_error and min_int_arr values in an output dataframe
                # to be used for the final output file for the capacitor values
                df_error.loc[i] = [min_error, min_int_arr]

            scaled_int_columns = np.vstack(df_error['int_cap']).T
            return scaled_int_columns
        def ideal_capacitor_values(scaled_coef, grid_range):
            # Then for capacitance minimization, we need to find the minimum value
            # of the coefficients in the opamp inputs
            # OP1 inputs c_1_1*c_a, c_2_1*c_a, c_a
            # first group to scale within:
            # cVal1
            # cVal2
            # cVal3
            op1_inputs = scale_inputs(scaled_coef[:3, :])
            # OP2 inputs c_4_1*c_a, c_3_1*c_a, c_b, c_1pp_1*c_a
            # second group to scale within:
            # cVal4
            # cVal5
            # cVal6
            # cVal7
            op2_inputs = scale_inputs(scaled_coef[-4:, :])
            # Horizontally join OP1_scaled_columns and OP2_scaled_columns
            joined_columns = np.vstack((op1_inputs, op2_inputs))
            # Joined_columns are the scaled capacitor values for dynamic range
            # and spread minimization. They will be used for the integer
            # optimization
            return joined_columns
        
        file_path = file_path_entry.get()
        grid_range = tuple(float(x) for x in grid_range_entry.get().split())
        column_header_prefix = column_header_prefix_entry.get()
        df = pd.read_csv(file_path)

            # Find output parts and split it
        # TODO: Regex search for output
        outputIdx = df[df['Parameter'] == 'Output'].index[0]
            # Split the dataframe into two parts for input and output
            # Input represents the coefficients of the capacitor values that are
            # used in the simulation
        df_input = df.iloc[:outputIdx, :]

            # Output represents the voltages of the opamp outputs
            # that are used in the simulation
            # ADEXL voltage output form is:
        df_out = df.iloc[outputIdx+1:, :]

            # Get column names
        colNames = list(df.columns)
            # Get size of the column names and use it to get the number of corners
            # that are simulated
        numCorners = len(colNames)-2

            # Get the capacitor values from the input dataframe
        cap_coef = df_input[colNames[-numCorners:]].to_numpy().astype(float)
            # Get the opamp output values from the output dataframe
        coef_out = df_out[colNames[-numCorners:]].to_numpy().astype(float)
        c_a, c_b = 1.0, 1.0


        # c_1p is zero for low pass filter case with low Q factor
        # TODO: Ask user for c_1p value for general cases
        c_1p = 0.0

        # Check values again!!!

        # These are the values for the dynamic range equalization of the filters
        # for low Q Q<3 cases, add high Q cases
        # cVal1=c_1_1*c_a
        # cVal2=c_2_1*c_a, multiply by OP1 output
        # cVal3=c_a, multiply by OP2 output
        # cVal4=c_4_1*c_a, multiply by OP1 output
        # cVal5=c_3_1*c_a, multiply by OP2 output
        # cVal6=c_1pp_1*c_a
        # cVal7=c_b, multiply by OP1 output

        scaled_coef = np.array([
            cap_coef[0],
            cap_coef[1] * coef_out[0],
            cap_coef[2] * coef_out[1],
            cap_coef[3] * coef_out[0],
            cap_coef[4] * coef_out[1],
            cap_coef[5],
            cap_coef[6] * coef_out[0]
        ])
        scaled_int_columns = optimize_capacitor_values(scaled_coef, grid_range)
        
        # Define column headers
        column_headers = ['Corner']
        column_headers_enable = ['Enable']


        # Iterate over the columns of the scaled_int_columns array
        for i in range(np.shape(scaled_int_columns)[1]):
            # Append the column header prefix and column number to the list of column headers
            column_headers.append(column_header_prefix + str(i+1))
            # Append 't' (representing 'true') to the list of column headers for the 'Enable' column
            column_headers_enable.append('t')

        # Stack the column headers and 'Enable' values into a 2D array
        dataHeaders = np.vstack((column_headers, column_headers_enable))

        # Define the capacitor values
        # TODO: Ask user for the capacitor values in the order of the column headers in future for other circuits
        cap_list = ['cVal1', 'cVal2', 'cVal3',
                    'cVal4', 'cVal5', 'cVal6', 'cVal7', 'c_1p']

        # Define the enable values for the capacitors in the order of the column headers
        c_1p_stack = np.repeat(c_1p, scaled_int_columns.shape[1])
        val_list3 = np.vstack((scaled_int_columns, c_1p_stack))

        # Concatenate the capacitor values and enable values into a 2D array
        dataCSV_d = np.hstack(
            (np.array(cap_list).reshape(val_list3.shape[0], 1), val_list3))

        # Join the current working directory, column header prefix and file extension to create the full file name with the path.
        fullFileName = os.path.join(os.getcwd(), column_header_prefix + "." + 'csv')
            # Open the file with the full file name in write mode, with UTF8 encoding and without adding any extra line breaks.
        with open(fullFileName, 'w', encoding='UTF8', newline='') as f:

            # Create a csv writer object to write to the file with the appropriate line terminator
            writer = csv.writer(f, lineterminator=os.linesep)

            # Write the header row of the CSV file to the file
            writer.writerows(dataHeaders)

            # Write the data rows of the CSV file to the file
            writer.writerows(dataCSV_d)

        status_label.config(text="Optimization complete!")
    def plot_bode(frequency, gain, phase):

        # Create a new matplotlib Figure and an Axes which fills it.
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.semilogx(frequency, gain)  # Bode magnitude plot
        # ax1.semilogx(frequency, phase)  # Bode phase plot
        tick_spacing=10
        ax1.grid()
        # Set the legend to bottom left corner of the plot and set the font size
        # to 10 points
        ax1.set_title('Ideal Filter Response')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Output (dB)', )
        # Set size of the tick labels
        ax1.tick_params(axis='both', which='major')
        
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        # show x-axis values not in engineering notation but as numbers in
        # ticker 
        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

        ax1.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=12))
        ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10,subs=np.arange(2, 10)*.1,numticks=12))
        ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
        # Create the canvas.
        canvas = FigureCanvasTkAgg(fig, master=root)
        plot_widget = canvas.get_tk_widget()

        # Add the plot to the tkinter widget.
        plot_widget.grid(row=13, column=1)

    def on_button():
        try:
            filter_type = filter_type_var.get()
            b_type = b_type_var.get()
            Fs = float(e3.get())
            if b_type == 'lowpass' or b_type == 'highpass':
                fc = float(e5.get())
            elif b_type == 'bandpass' or b_type == 'bandstop':
                fc = [float(e6.get()), float(e7.get())]
            if filter_type in ['cheby1', 'ellip']:
                rp = float(e8.get())
            else:
                rp = None
            if filter_type in ['cheby2', 'ellip']:
                rs = float(e9.get())
            else:
                rs = None
            order = int(e10.get())
            filename = e11.get()

            frequency, gain, phase = generate_csv(filter_type, b_type, Fs, fc, rp, rs, order, filename)
            plot_bode(frequency, gain, phase)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
            create_gui()

    root = Tk()
    root.title("pyswcap - Capacitor Sizing Tool GUI")
    # root.geometry('500x500')

    Label(root, text="IIR Filter Design Type:").grid(row=0)
    Label(root, text="Filter Type:").grid(row=1)
    Label(root, text="Sampling Frequency:").grid(row=2)
    

    filter_type_var = StringVar(root)
    filter_type_var.trace("w", filter_type_callback)

    filter_types = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    filter_type_var.set(filter_types[0])  # default value
    OptionMenu(root, filter_type_var, *filter_types).grid(row=0, column=1, sticky='ew')

    b_type_var = StringVar(root)
    b_type_var.trace("w", b_type_callback)

    b_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    b_type_var.set(b_types[0])  # default value
    OptionMenu(root, b_type_var, *b_types).grid(row=1, column=1, sticky='ew')
    
    e3 = Entry(root)
    e3.grid(row=2, column=1, sticky='ew')

    Label(root, text="Cutoff Value in Hz:").grid(row=4)
    e5 = Entry(root)
    e5.grid(row=4, column=1, sticky='ew')
    Label(root, text="Lower Cutoff Value in Hz:").grid(row=5)
    e6 = Entry(root)
    e6.grid(row=5, column=1, sticky='ew')
    Label(root, text="Upper Cutoff Value in Hz:").grid(row=6)
    e7 = Entry(root)
    e7.grid(row=6, column=1, sticky='ew')
    Label(root, text="Passband Ripple (rp):").grid(row=7)
    e8 = Entry(root)
    e8.configure(state='disabled')
    e8.grid(row=7, column=1, sticky='ew')
    Label(root, text="Stopband Ripple (rs):").grid(row=8)
    e9 = Entry(root)
    e9.configure(state='disabled')
    e9.grid(row=8, column=1, sticky='ew')
    Label(root, text="Filter Order:").grid(row=9)
    e10 = Entry(root)
    e10.grid(row=9, column=1, sticky='ew')
    Label(root, text="CSV Filename:").grid(row=10)
    e11 = Entry(root)
    e11.grid(row=10, column=1, sticky='ew')

    Button(root, text='Generate filters and export as CSV', command=on_button).grid(row=11, column=0, sticky=W, pady=4)


    # CSV File Path input
    Label(root, text="CSV File Path:").grid(row=3, column=4)
    file_path_entry = tk.Entry(root)
    file_path_entry.grid(row=3, column=5, sticky='ew')

    def set_file_path():
        file_path = filedialog.askopenfilename()
        file_path_entry.delete(0, tk.END)  # Remove current text in entry
        file_path_entry.insert(0, file_path)  # Insert the 'path'

    browse_button = tk.Button(root, text="Browse", command=set_file_path)
    browse_button.grid(row=3, column=6, sticky='ew')

    # Grid range input
    Label(root, text="Limits of cost function search (e.g. 0.1 6):").grid(row=4, column=4)
    grid_range_entry = tk.Entry(root)
    grid_range_entry.grid(row=4, column=5, sticky='ew')

    # Column header prefix input
    Label(root, text="Prefix for output column headers:").grid(row=5,column=4)
    column_header_prefix_entry = tk.Entry(root)
    column_header_prefix_entry.grid(row=5, column=5, sticky='ew')

    # Run button
    run_button = tk.Button(root, text="Run Capacitor Sizing", command=run_program)
    run_button.grid(row=6, column=5, sticky=W, pady=4)

     # Load the image file
    img = Image.open('logo.png')  # Adjust 'logo.png' to your image file path
    img = img.resize((100, 100), Image.ANTIALIAS)  # Resize image (width, height)
    img = ImageTk.PhotoImage(img)

    # Create a label and assign the image
    logo_label = tk.Label(root, image=img)
    logo_label.image = img  # Keep a reference to the image to prevent it from being garbage collected
    logo_label.grid(row=0, column=6)  # Adjust the row and column as per your GUI layout
    
    status_label = tk.Label(root, text="")
    status_label.grid(row=7, column=5)

    # This creates a vertical line
    hline = Frame(root, bg="black", width=2)
    hline.grid(row=12, column=0, sticky="ew",columnspan=1000)  # Adjust the row and column as per your GUI layout
    # This creates a vertical line
    vline = Frame(root, bg="black", width=2)
    vline.grid(row=0, column=3, sticky="ns",rowspan=12)  # Adjust the row and column as per your GUI layout


    filter_type_callback()  # Call the function to set initial state
    b_type_callback()  # Call the function to set initial state
    mainloop()

create_gui()
