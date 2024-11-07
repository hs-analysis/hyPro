import gradio as gr
import pandas as pd
import requests
import os
import plotly.graph_objects as go
import numpy as np
from io import StringIO
import csv
from datetime import datetime
import json
import mysql.connector
import string
from dotenv import load_dotenv

load_dotenv()
api_base_url = os.getenv("API_BASE_URL")

# Fetch the list of available models from the API
try:
    response = requests.get(f"{api_base_url}/v1/models")
    if response.status_code == 200:
        models_data = response.json()
        available_models = models_data.get("models", [])
    else:
        available_models = []
except Exception:
    available_models = []

# Prepare model options for the dropdown menu
model_display_options = []
model_details_map = {}
for model in available_models:
    display_name = f"{model['name']} v{model['version']}"
    model_display_options.append(display_name)
    model_details_map[display_name] = {
        'id': model['id'],
        'version': model['version'],
        'input_cols': model['input_cols'],
        'description': model['description'],
        'output_cols': model['output_cols']
    }

def display_selected_model_description(selected_model_name):
    """
    Updates the model description based on the selected model.
    """
    model_info = model_details_map.get(selected_model_name)
    if model_info:
        description = f"**Model Description:** {model_info['description']}"
        output_columns = model_info.get('output_cols', [])
        description += f"\n\n**Model Output Columns:** {', '.join(output_columns)}"
        return description
    else:
        return "No model selected."

def display_model_input_requirements(selected_model_name, input_method):
    """
    Displays the required inputs for the selected model based on the input method.
    """
    model_info = model_details_map.get(selected_model_name)
    predefined_manual_inputs = [
        "ang_open_percent",
        "T_furnace_C",
        "T_load_C",
        "T_start_C",
        "t_heat_s",
        "t_heat_total_s",
        "t_process_s",
        "t_vacuum_s",
        "t_z_cool_s",
        "v_cool_mm_s",
        "v_x_back_mm_s",
        "v_x_towards_mm_s",
        "v_z_up_mm_s",
        "z_heat_mm",
        "z_home_mm"
    ]
    if model_info:
        if input_method == "CSV File":
            input_columns = model_info.get('input_cols', {})
            input_columns_list = [f"| {col} | {min_val} | {max_val} |" for col, (min_val, max_val) in input_columns.items()]
            inputs_markdown = f"**API Endpoint:** `/v1/pred/from_timeseries/{{model_id}}/{{model_version}}/`<br />"
            inputs_markdown += "**Required CSV Columns:**\n"
            inputs_markdown += "| Column Name | Min Value | Max Value |\n"
            inputs_markdown += "|-------------|-----------|-----------|\n"
            inputs_markdown += "\n".join(input_columns_list)
        elif input_method == "Manual Input":
            inputs_markdown = f"**API Endpoint:** `/v1/pred/from_presets/{{model_id}}/{{model_version}}/`<br />"
            inputs_markdown += "**Required Manual Inputs:**<br />- " + "<br />- ".join(predefined_manual_inputs)
        elif input_method == "Database Input":
            inputs_markdown = f"**API Endpoint:** `/v1/pred/from_ps_and_ts/{{model_id}}/{{model_version}}/`<br />"
            inputs_markdown += "**Required Database Inputs:**<br />- table_name<br />"
            inputs_markdown += "**Required Machine Preset Inputs:**<br />- " + "<br />- ".join(predefined_manual_inputs)
        else:
            inputs_markdown = ""
    else:
        inputs_markdown = "No model selected."
    return inputs_markdown

def handle_user_inputs(
    machine_id_input, selected_model_name,
    input_method,
    uploaded_csv_file, csv_delimiter_input, csv_text_qualifier,
    ang_open_percent_input, furnace_temp_input, load_temp_input, start_temp_input,
    heating_time_input, total_heating_time_input, process_time_input, vacuum_time_input, cooling_time_input,
    cooling_speed_input, x_back_speed_input, x_towards_speed_input, z_up_speed_input,
    heat_z_position_input, home_z_position_input,
    database_tmpexp_table, database_ipt_table
):
    """
    Processes user inputs based on the selected input method and model.
    """
    model_info = model_details_map.get(selected_model_name)
    if not model_info:
        return {"error": "Invalid model selected."}, None, None, None, None, None, None

    model_id = model_info['id']
    model_version = model_info['version']

    if input_method == "Manual Input":
        payload = {
            "machine_id": machine_id_input,
            "ang_open_percent": ang_open_percent_input,
            "T_furnace_C": furnace_temp_input,
            "T_load_C": load_temp_input,
            "T_start_C": start_temp_input,
            "t_heat_s": heating_time_input,
            "t_heat_total_s": total_heating_time_input,
            "t_process_s": process_time_input,
            "t_vacuum_s": vacuum_time_input,
            "t_z_cool_s": cooling_time_input,
            "v_cool_mm_s": cooling_speed_input,
            "v_x_back_mm_s": x_back_speed_input,
            "v_x_towards_mm_s": x_towards_speed_input,
            "v_z_up_mm_s": z_up_speed_input,
            "z_heat_mm": heat_z_position_input,
            "z_home_mm": home_z_position_input
        }
        api_url = f"{api_base_url}/v1/pred/from_presets/{model_id}/{model_version}/"
        params = {'machine_id': machine_id_input}
        try:
            api_response = requests.post(api_url, json=payload, params=params)
        except Exception as e:
            return {"error": f"Error during API call: {str(e)}"}, None, None, None, None, None

        if api_response.status_code == 200:
            response_data = api_response.json()
        else:
            return {"error": f"API call failed with status code {api_response.status_code}: {api_response.text}"}, None, None, None, None, None

        segments = response_data.get("segments", [])
        if not segments:
            return {"error": "No segments found in API response."}, None, None, None, None, None

        df_segments = pd.DataFrame(segments)
        df_segments['time'] = 0
        df_segments['time_relative'] = 0

        heatmap_fig = generate_heatmap_plot(df_segments)

        if not df_segments.empty:
            default_x = df_segments['xy_location'].iloc[0][0]
            default_y = df_segments['xy_location'].iloc[0][1]
        else:
            default_x, default_y = None, None

        timeline_fig = regenerate_timeline_plot(default_x, default_y, df_segments)

        return {
            "message": "Success",
            "start_time": "N/A",
            "end_time": "N/A",
        }, heatmap_fig, df_segments, gr.update(value=default_x), gr.update(value=default_y), timeline_fig, gr.update(minimum=0, maximum=0, value=0)

    elif input_method == "CSV File":
        if uploaded_csv_file is None:
            return {"error": "Please upload a CSV file."}, None, None, None, None, None, None, None, None
        else:
            api_url = f"{api_base_url}/v1/pred/from_timeseries/{model_id}/{model_version}/"
            params = {'machine_id': machine_id_input}

            delimiter = csv_delimiter_input or ','
            quotechar = csv_text_qualifier or '"'

            try:
                with open(uploaded_csv_file.name, 'r', newline='', encoding='utf-8-sig') as f:
                    reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
                    lines = list(reader)
            except Exception as e:
                return {"error": f"Error reading CSV file: {str(e)}"}, None, None, None, None, None, None, None, None

            header = lines[0]
            data_lines = lines[1:]

            if 'Timestamp' not in header:
                return {"error": "CSV file must contain a 'Timestamp' column."}, None, None, None, None, None, None, None, None
            timestamp_idx = header.index('Timestamp')

            time_to_records = {}
            for line in data_lines:
                if len(line) != len(header):
                    continue
                try:
                    timestamp = datetime.strptime(line[timestamp_idx], "%Y-%m-%d-%H:%M:%S.%f").timestamp()
                except ValueError:
                    continue
                t = int(np.floor(timestamp))
                if t not in time_to_records:
                    time_to_records[t] = []
                line_str = delimiter.join([f'{quotechar}{field}{quotechar}' if delimiter in field or quotechar in field else field for field in line])
                time_to_records[t].append(line_str)

            if not time_to_records:
                return {"error": "No valid data found in CSV file."}, None, None, None, None, None, None, None, None

            api_responses = []
            for t in range(min(time_to_records.keys()), max(time_to_records.keys()) + 1):
                header_line = delimiter.join([f'{quotechar}{col}{quotechar}' if delimiter in col or quotechar in col else col for col in header])
                lines_up_to_t = [header_line]
                for tt in range(min(time_to_records.keys()), t + 1):
                    if tt in time_to_records:
                        lines_up_to_t.extend(time_to_records[tt])

                csv_buffer = StringIO('\n'.join(lines_up_to_t))
                csv_buffer.seek(0)
                files = {'file': ('data.csv', csv_buffer, 'text/csv')}
                try:
                    api_response = requests.post(api_url, files=files, params=params)
                except Exception as e:
                    return {"error": f"Error during API call at t={t}: {str(e)}"}, None, None, None, None, None, None, None, None
                if api_response.status_code == 200:
                    response_data = api_response.json()
                    api_responses.append((t, response_data))
                else:
                    return {"error": f"API call failed at t={t} with status code {api_response.status_code}: {api_response.text}"}, None, None, None, None, None, None, None, None

            all_segments = []
            for t, response_data in api_responses:
                segments = response_data.get("segments", [])
                for segment in segments:
                    segment['time'] = t
                    all_segments.append(segment)
            df_segments = pd.DataFrame(all_segments)
            df_segments['time_relative'] = df_segments['time'] - min(time_to_records.keys())
            initial_time = int(df_segments['time_relative'].max())
            df_initial_time = df_segments[df_segments['time_relative'] == initial_time]
            heatmap_fig = generate_heatmap_plot(df_initial_time)

            if not df_segments.empty:
                default_x = df_segments['xy_location'].iloc[0][0]
                default_y = df_segments['xy_location'].iloc[0][1]
            else:
                default_x, default_y = None, None

            timeline_fig = regenerate_timeline_plot(default_x, default_y, df_segments)

            duration = int(df_segments['time_relative'].max())
            initial_time = duration

            return {
                "message": "Success",
                "start_time": datetime.fromtimestamp(min(time_to_records.keys())).strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.fromtimestamp(max(time_to_records.keys())).strftime('%Y-%m-%d %H:%M:%S'),
            }, heatmap_fig, df_segments, gr.update(value=default_x), gr.update(value=default_y), timeline_fig, gr.update(minimum=0, maximum=duration, value=initial_time)

    elif input_method == "Database Input":
        db_username = os.getenv('DB_USERNAME')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')

        mydb = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=db_username,
            password=db_password,
            database=db_name,
        )
        cursor = mydb.cursor()
        # TODO: get from input
        table1 = database_ipt_table
        table2 = database_tmpexp_table

        # validate table names
        accaptable = set(string.ascii_letters + string.digits + '$' + '_') # set of valid sql table character if unquoted table name
        if not set(table1).issubset(accaptable):
            return {"error": f"Error: Invalid table name: {table1}"}, None, None, None, None, None, None
        if not set(table2).issubset(accaptable):
            return {"error": f"Error: Invalid table name: {table2}"}, None, None, None, None, None, None

        sql = f"""SELECT 
        DATE_FORMAT({table1}.`Timestamp`, '%Y-%m-%d-%H:%i:%s'),
        {table1}.ta1,
        {table1}.ta2,
        {table1}.ta3,
        {table1}.ta4,
        {table1}.ta5,
        {table1}.tb1,
        tb2,
        tb3,
        tb4,
        tb5,
        tc1,
        tc2,
        tc3,
        tc4,
        tc5,
        td1,
        pz,
        px,
        fllr1,
        flhr1,
        xv,
        zv,
        av,
        vs,
        ht,
        st,
        sp1,
        pc
        FROM   {table1}
        LEFT JOIN {table2}
        ON 
        ABS(TIMESTAMPDIFF(MICROSECOND , {table2}.`Timestamp`, {table1}.`Timestamp`)) < 500000
        ORDER BY {table1}.timestamp DESC
        LIMIT 900;
        """
        res = None
        try:
            cursor.execute(sql)
            res = cursor.fetchall()
        except Exception as e:
            return {"error": f"Error reading data from table {database_tmpexp_table}: {str(e)}"}, None, None, None, None, None, None

        csv_buffer = StringIO()
        df = pd.DataFrame(res)
        delimiter = ';'
        quotechar = '"'

        df.to_csv(csv_buffer, index=False, sep=";", header=["Timestamp", 
                                                            "TA1", 
                                                            "TA2", 
                                                            "TA3", 
                                                            "TA4", 
                                                            "TA5", 
                                                            "TB1", 
                                                            "TB2", 
                                                            "TB3", 
                                                            "TB4", 
                                                            "TB5", 
                                                            "TC1", 
                                                            "TC2", 
                                                            "TC3", 
                                                            "TC4", 
                                                            "TC5", 
                                                            "TD1", 
                                                            "PZ", 
                                                            "PX", 
                                                            "FLLR1", 
                                                            "FLHR1", 
                                                            "XV", 
                                                            "ZV", 
                                                            "AV", 
                                                            "VS", 
                                                            "HT", 
                                                            "ST", 
                                                            "SP1", 
                                                            "PC"
                                                            ])

        csv_buffer.seek(0)
        reader = csv.reader(csv_buffer, delimiter=delimiter, quotechar=quotechar)
        lines = list(reader)

        header = lines[0]
        data_lines = lines[1:]

        if 'Timestamp' not in header:
            return {"error": "CSV file must contain a 'Timestamp' column."}, None, None, None, None, None, None, None, None
        timestamp_idx = header.index('Timestamp')

        time_to_records = {}
        for line in data_lines:
            if len(line) != len(header):
                continue
            try:
                timestamp = datetime.strptime(line[timestamp_idx], "%Y-%m-%d-%H:%M:%S").timestamp()
            except ValueError:
                continue
            t = int(np.floor(timestamp))
            if t not in time_to_records:
                time_to_records[t] = []
            line_str = delimiter.join([f'{quotechar}{field}{quotechar}' if delimiter in field or quotechar in field else field for field in line])
            time_to_records[t].append(line_str)

        if not time_to_records:
            return {"error": "No valid data found in CSV file."}, None, None, None, None, None, None, None, None

        api_responses = []
        machine_presets = {
            "ang_open_percent": ang_open_percent_input,
            "T_furnace_C": furnace_temp_input,
            "T_load_C": load_temp_input,
            "T_start_C": start_temp_input,
            "t_heat_s": heating_time_input,
            "t_heat_total_s": total_heating_time_input,
            "t_process_s": process_time_input,
            "t_vacuum_s": vacuum_time_input,
            "t_z_cool_s": cooling_time_input,
            "v_cool_mm_s": cooling_speed_input,
            "v_x_back_mm_s": x_back_speed_input,
            "v_x_towards_mm_s": x_towards_speed_input,
            "v_z_up_mm_s": z_up_speed_input,
            "z_heat_mm": heat_z_position_input,
            "z_home_mm": home_z_position_input
        }
        params = {'machine_id': machine_id_input}
        api_url = f"{api_base_url}/v1/pred/from_ps_and_ts/{model_id}/{model_version}/"
        for t in range(min(time_to_records.keys()), max(time_to_records.keys()) + 1):
            header_line = delimiter.join([f'{quotechar}{col}{quotechar}' if delimiter in col or quotechar in col else col for col in header])
            lines_up_to_t = [header_line]
            for tt in range(min(time_to_records.keys()), t + 1):
                if tt in time_to_records:
                    lines_up_to_t.extend(time_to_records[tt])

            csv_buffer = StringIO('\n'.join(lines_up_to_t))
            csv_buffer.seek(0)
            csv_string = csv_buffer.getvalue()
            csv_string = csv_string.replace('.',',')
            files = {'file': ('data.csv', csv_string, 'text/csv'), 'presets': (None, json.dumps(machine_presets))}
            try:
                api_response = requests.post(api_url, files=files)
            except Exception as e:
                return {"error": f"Error during API call at t={t}: {str(e)}"}, None, None, None, None, None, None, None, None

            if api_response.status_code == 200:
                response_data = api_response.json()
                api_responses.append((t, response_data))
            else:
                return {"error": f"API call failed at t={t} with status code {api_response.status_code}: {api_response.text}"}, None, None, None, None, None, None, None, None

        all_segments = []
        for t, response_data in api_responses:
            segments = response_data.get("segments", [])
            for segment in segments:
                segment['time'] = t
                all_segments.append(segment)
        df_segments = pd.DataFrame(all_segments)
        df_segments['time_relative'] = df_segments['time'] - min(time_to_records.keys())
        initial_time = int(df_segments['time_relative'].max())
        df_initial_time = df_segments[df_segments['time_relative'] == initial_time]
        heatmap_fig = generate_heatmap_plot(df_initial_time)

        if not df_segments.empty:
            default_x = df_segments['xy_location'].iloc[0][0]
            default_y = df_segments['xy_location'].iloc[0][1]
        else:
            default_x, default_y = None, None

        timeline_fig = regenerate_timeline_plot(default_x, default_y, df_segments)

        duration = int(df_segments['time_relative'].max())
        initial_time = duration

        return {
            "message": "Success",
            "start_time": datetime.fromtimestamp(min(time_to_records.keys())).strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.fromtimestamp(max(time_to_records.keys())).strftime('%Y-%m-%d %H:%M:%S'),
        }, heatmap_fig, df_segments, gr.update(value=default_x), gr.update(value=default_y), timeline_fig, gr.update(minimum=0, maximum=duration, value=initial_time)

    else:
        return {"error": "Invalid input type selected."}, None, None, None, None, None, None

def generate_heatmap_plot(df_segments):
    """
    Generates a heatmap plot based on the segments data.
    """
    if df_segments.empty:
        return None
    df_segments['x_mm'] = df_segments['x_range_mm'].apply(lambda x: (x[0] + x[1]) / 2)
    df_segments['y_mm'] = df_segments['y_range_mm'].apply(lambda y: (y[0] + y[1]) / 2)
    heatmap_data = df_segments.pivot(index='y_mm', columns='x_mm', values='pred_p2v')
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
    fig = go.Figure(data=go.Heatmap(
        x=heatmap_data.columns,
        y=heatmap_data.index,
        z=heatmap_data.values,
        colorscale='Blues',
        colorbar=dict(title='Pred P2V')
    ))
    fig.update_layout(
        title='Predicted P2V Heatmap at Last Time Point',
        xaxis_title='X Position (mm)',
        yaxis_title='Y Position (mm)',
    )
    return fig

def update_heatmap_plot(selected_time, df_segments):
    """
    Updates the heatmap plot based on the selected time.
    """
    if df_segments is None or df_segments.empty:
        return None
    df_time_segment = df_segments[df_segments['time_relative'] == selected_time]
    if df_time_segment.empty:
        return None
    fig = generate_heatmap_plot(df_time_segment)
    return fig

def regenerate_timeline_plot(x_coord, y_coord, df_segments):
    """
    Regenerates the timeline plot based on selected X and Y coordinates.
    """
    if df_segments is None or df_segments.empty:
        return None

    df_segments['x_location'] = df_segments['xy_location'].apply(lambda x: x[0])
    df_segments['y_location'] = df_segments['xy_location'].apply(lambda y: y[1])

    matching_segments = df_segments[
        (df_segments['x_location'] == x_coord) & 
        (df_segments['y_location'] == y_coord)
    ]
    if matching_segments.empty:
        return None
    else:
        matching_segments = matching_segments.sort_values('time_relative')
        times = matching_segments['time_relative']
        pred_p2v_values = matching_segments['pred_p2v']
        pred_rms_values = matching_segments['pred_rms']

        total_duration = times.max() if not times.empty else 0

        tick_values = []
        tick_labels = []
        time_unit = ""
        
        if total_duration <= 60:
            tick_interval = 10
            time_unit = "seconds"
            tick_values = [t for t in times if t % tick_interval == 0]
            tick_labels = [f"{int(t)}s" for t in tick_values]
        elif total_duration <= 3600:
            tick_interval = 60
            time_unit = "minutes"
            tick_values = [t for t in times if t % tick_interval == 0]
            tick_labels = [str(datetime.utcfromtimestamp(t).strftime('%M:%S')) for t in tick_values]
        else:
            tick_interval = 300
            time_unit = "hours"
            tick_values = [t for t in times if t % tick_interval == 0]
            tick_labels = [str(datetime.utcfromtimestamp(t).strftime('%H:%M:%S')) for t in tick_values]

        if 0 not in tick_values:
            tick_values.insert(0, 0)
            if time_unit == "seconds":
                tick_labels.insert(0, "0s")
            elif time_unit == "minutes":
                tick_labels.insert(0, "00:00")
            else:
                tick_labels.insert(0, "00:00:00")

        if total_duration not in tick_values:
            tick_values.append(total_duration)
            if time_unit == "seconds":
                tick_labels.append(f"{int(total_duration)}s")
            elif time_unit == "minutes":
                tick_labels.append(str(datetime.utcfromtimestamp(total_duration).strftime('%M:%S')))
            else:
                tick_labels.append(str(datetime.utcfromtimestamp(total_duration).strftime('%H:%M:%S')))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, 
            y=pred_p2v_values, 
            mode='lines+markers', 
            name='Pred P2V'
        ))
        fig.add_trace(go.Scatter(
            x=times, 
            y=pred_rms_values, 
            mode='lines+markers', 
            name='Pred RMS'
        ))

        fig.update_layout(
            title='Timeline Plot',
            xaxis_title=f'Time ({time_unit})',
            yaxis_title='Value',
            xaxis=dict(
                range=[0, total_duration],
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_labels
            )
        )
    return fig

def navigate_to_output_tab():
    """
    Switches the interface to the Outputs tab.
    """
    return gr.Tabs(selected=1)

def update_manual_inputs_from_csv(uploaded_file):
    """
    Updates manual input fields based on the uploaded CSV file.
    """
    error_message = ""
    if uploaded_file is None:
        # No file uploaded; clear any previous error messages
        return [gr.update()] * 15 + [gr.update(value="")]
    
    try:
        df = pd.read_csv(uploaded_file.name, sep='\r\n', header=None)
    except Exception as e:
        error_message = f"Error reading CSV file: {str(e)}"
        return [gr.update()] * 15 + [gr.update(value=error_message)]
    
    id_row = df.iloc[0][0].split(';')[1:]
    value_row = df.iloc[2][0].split(';')[1:]
    
    if not value_row:
        error_message = "No 'Value' row found in CSV file"
        return [gr.update()] * 15 + [gr.update(value=error_message)]
        
    # Map variable names to values
    var_to_value = dict(zip(id_row, value_row))
    
    # Mapping of variable names to input fields
    variable_to_input_field = {
        'ang_open_percent': ang_open_percent_input,
        'T_furnace_C': furnace_temp_input,
        'T_load_C': load_temp_input,
        'T_start_C': start_temp_input,
        't_heat_s': heating_time_input,
        't_heat_total_s': total_heating_time_input,
        't_process_s': process_time_input,
        't_vacuum_s': vacuum_time_input,
        't_z_cool_s': cooling_time_input,
        'v_cool_mm_s': cooling_speed_input,
        'v_x_back_mm_s': x_back_speed_input,
        'v_x_towards_mm_s': x_towards_speed_input,
        'v_z_up_mm_s': z_up_speed_input,
        'z_heat_mm': heat_z_position_input,
        'z_home_mm': home_z_position_input
    }
    
    updated_values = []
    for var_name in variable_to_input_field.keys():
        if var_name in var_to_value:
            value = var_to_value[var_name]
            try:
                value = float(value)
            except ValueError:
                error_message = f"Invalid value for {var_name}: {value}"
                return [gr.update()] * 15 + [gr.update(value=error_message)]
            updated_values.append(gr.update(value=value))
        else:
            updated_values.append(gr.update())
    
    # Clear any previous error messages
    updated_values.append(gr.update(value=""))
    return updated_values

with gr.Blocks() as demo:
    gr.Markdown("# Gradio Frontend for Data Processing")

    with gr.Tabs() as tabs:
        # Inputs Tab
        with gr.TabItem("Inputs", id=0):
            with gr.Row():
                machine_id_input = gr.Textbox(
                    label="Machine ID",
                    value="Unknown"
                )

                selected_model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=model_display_options,
                    value=model_display_options[0] if model_display_options else None
                )

            model_description_markdown = gr.Markdown("")

            selected_model_dropdown.change(
                fn=display_selected_model_description,
                inputs=[selected_model_dropdown],
                outputs=[model_description_markdown]
            )

            with gr.Row():
                selected_input_type_radio = gr.Radio(
                    choices=["Manual Input", "CSV File", "Database Input"],
                    label="Select Input Type",
                    value="Manual Input"
                )
                constraints_display_checkbox = gr.Checkbox(label="Display constraints", value=False)

            with gr.Row():
                required_inputs_markdown = gr.Markdown("", visible=False)

                with gr.Group(visible=True) as manual_input_group:
                    gr.Markdown("## Manual Input Fields")
                    
                    uploaded_manual_input_csv = gr.File(label="Upload Manual Input CSV", file_types=['.csv'])
                    manual_input_error = gr.Markdown(value="", visible=True)

                    ang_open_percent_input = gr.Number(label="Angle Open Percent")
                    furnace_temp_input = gr.Number(label="Furnace Temperature (°C)")
                    load_temp_input = gr.Number(label="Load Temperature (°C)")
                    start_temp_input = gr.Number(label="Start Temperature (°C)")
                    heating_time_input = gr.Number(label="Heating Time (s)")
                    total_heating_time_input = gr.Number(label="Total Heating Time (s)")
                    process_time_input = gr.Number(label="Process Time (s)")
                    vacuum_time_input = gr.Number(label="Vacuum Time (s)")
                    cooling_time_input = gr.Number(label="Cooling Time (s)")
                    cooling_speed_input = gr.Number(label="Cooling Speed (mm/s)")
                    x_back_speed_input = gr.Number(label="X Back Speed (mm/s)")
                    x_towards_speed_input = gr.Number(label="X Towards Speed (mm/s)")
                    z_up_speed_input = gr.Number(label="Z Up Speed (mm/s)")
                    heat_z_position_input = gr.Number(label="Heat Z Position (mm)")
                    home_z_position_input = gr.Number(label="Home Z Position (mm)")
                
                with gr.Group(visible=False) as csv_input_group:
                    uploaded_csv_file = gr.File(label="Upload CSV File", file_types=['.csv'])
                    csv_delimiter_input = gr.Textbox(label="CSV Delimiter", value=";", placeholder="Enter the delimiter used in your CSV file")
                    csv_text_qualifier = gr.Textbox(label="Text Qualifier", value='"', placeholder="Enter the text qualifier used in your CSV file")

                with gr.Group(visible=False) as database_input_group:
                    gr.Markdown("## Database Input Fields")
                    database_tempexp_table = gr.Textbox(label="Tempexp Table Name", placeholder="Enter the tempexp table name")
                    database_ipt_table = gr.Textbox(label="IPT Table Name", placeholder="Enter the ipt table name")

                def update_input_components(selected_input_method, selected_model_name):
                    updated_required_inputs = display_model_input_requirements(selected_model_name, selected_input_method)
                    manual_input_visibility = selected_input_method in ["Manual Input", "Database Input"]
                    csv_input_visibility = selected_input_method == "CSV File"
                    database_input_visibility = selected_input_method == "Database Input"
                    return [
                        gr.update(visible=manual_input_visibility),
                        gr.update(visible=csv_input_visibility),
                        gr.update(visible=database_input_visibility),
                        updated_required_inputs
                    ]

                selected_input_type_radio.change(
                    fn=update_input_components,
                    inputs=[selected_input_type_radio, selected_model_dropdown],
                    outputs=[manual_input_group, csv_input_group, database_input_group, required_inputs_markdown]
                )

                selected_model_dropdown.change(
                    fn=update_input_components,
                    inputs=[selected_input_type_radio, selected_model_dropdown],
                    outputs=[manual_input_group, csv_input_group, database_input_group, required_inputs_markdown]
                )

            demo.load(
                fn=lambda: (display_selected_model_description(selected_model_dropdown.value), display_model_input_requirements(selected_model_dropdown.value, selected_input_type_radio.value)),
                inputs=[],
                outputs=[model_description_markdown, required_inputs_markdown]
            )

            submit_button = gr.Button("Submit")

        # Outputs Tab
        with gr.TabItem("Outputs", id=1):
            with gr.Row():
                display_api_response_checkbox = gr.Checkbox(label="Display response", value=False)

            with gr.Row():
                api_response_container = gr.Column(visible=False)
                with api_response_container:
                    api_response_json = gr.JSON(label="API Response")

            with gr.Row():
                with gr.Column():                
                    time_slider = gr.Slider(label="Time (Seconds)", minimum=0, maximum=100, step=1, value=0, interactive=True)
                    segment_heatmap_plot = gr.Plot(label="Segment Heatmap Plot")

                with gr.Column():
                    x_coordinate_input = gr.Number(label="X Coordinate")
                    y_coordinate_input = gr.Number(label="Y Coordinate")
                    segments_timeline_plot = gr.Plot(label="Segments Timeline Plot")

            def toggle_api_response_visibility(show_response):
                return gr.update(visible=show_response)

            display_api_response_checkbox.change(
                fn=toggle_api_response_visibility,
                inputs=[display_api_response_checkbox],
                outputs=[api_response_container]
            )

            def toggle_constraints_visibility(show_constraints, current_required_inputs):
                if show_constraints:
                    return gr.update(value=current_required_inputs, visible=True)
                else:
                    return gr.update(visible=False)

            constraints_display_checkbox.change(
                fn=toggle_constraints_visibility,
                inputs=[constraints_display_checkbox, required_inputs_markdown],
                outputs=[required_inputs_markdown]
            )

            segments_data_state = gr.State()

            submit_button.click(navigate_to_output_tab, None, tabs)
            submit_button.click(
                fn=handle_user_inputs,
                inputs=[
                    machine_id_input, selected_model_dropdown,
                    selected_input_type_radio,
                    uploaded_csv_file, csv_delimiter_input, csv_text_qualifier,
                    ang_open_percent_input, furnace_temp_input, load_temp_input, start_temp_input,
                    heating_time_input, total_heating_time_input, process_time_input, vacuum_time_input, cooling_time_input,
                    cooling_speed_input, x_back_speed_input, x_towards_speed_input, z_up_speed_input,
                    heat_z_position_input, home_z_position_input,
                    database_tempexp_table, database_ipt_table
                ],
                outputs=[
                    api_response_json,
                    segment_heatmap_plot, 
                    segments_data_state, 
                    x_coordinate_input, 
                    y_coordinate_input, 
                    segments_timeline_plot, 
                    time_slider
                ]
            )
            
            time_slider.change(
                fn=update_heatmap_plot,
                inputs=[time_slider, segments_data_state],
                outputs=[segment_heatmap_plot]
            )

            x_coordinate_input.change(
                fn=regenerate_timeline_plot,
                inputs=[x_coordinate_input, y_coordinate_input, segments_data_state],
                outputs=[segments_timeline_plot]
            )

            y_coordinate_input.change(
                fn=regenerate_timeline_plot,
                inputs=[x_coordinate_input, y_coordinate_input, segments_data_state],
                outputs=[segments_timeline_plot]
            )

    uploaded_manual_input_csv.change(
        fn=update_manual_inputs_from_csv,
        inputs=[uploaded_manual_input_csv],
        outputs=[
            ang_open_percent_input, furnace_temp_input, load_temp_input, start_temp_input,
            heating_time_input, total_heating_time_input, process_time_input, vacuum_time_input, cooling_time_input,
            cooling_speed_input, x_back_speed_input, x_towards_speed_input, z_up_speed_input,
            heat_z_position_input, home_z_position_input,
            manual_input_error  # Output for error messages
        ]
    )

    demo.launch(server_name="0.0.0.0")
