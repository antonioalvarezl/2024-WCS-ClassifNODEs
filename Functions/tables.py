import pandas as pd
import matplotlib.colors
import matplotlib.cm

def color_scale_num(val):
    """Apply color based on the value."""
    norm = matplotlib.colors.Normalize(vmin=val.min(), vmax=val.max(), clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='Wistia')
    # Apply the color mapping to each individual value in the Series
    return val.apply(lambda x: f"background-color: {matplotlib.colors.rgb2hex(mapper.to_rgba(x))};")

def color_scale_numtxt(val, min_val, max_val):
    try:
        num_val = float(val.split(" (")[0])
    except ValueError:
        return ''  # Returns empty style if error
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='Wistia')
    return f"background-color: {matplotlib.colors.rgb2hex(mapper.to_rgba(num_val))};"

def highlight_min(data):
    """Highlight the minimum value in the data."""
    min_val = data.min()
    return ['font-weight: bold; background-color: #98FB98;' if v == min_val else '' for v in data]

def prepare_data_frame(data, index, columns, color_scale_type, caption):
    df = pd.DataFrame(data, index=index, columns=columns)
    
    if color_scale_type == 'numtxt':
        # Compute min and max one time for the whole DataFrame
        min_val, max_val = df.stack().str.extract(r"(\d+\.\d+)")[0].astype(float).agg(['min', 'max'])
        color_func = lambda s: color_scale_numtxt(s, min_val, max_val)
        style_func = lambda df: (df.style.map(color_func))
    else:
        style_func = lambda df: (df.style.apply(color_scale_num, axis=0))

    styled_df = style_func(df)
    styled_df = (styled_df
                 .apply(highlight_min, subset=df.columns)
                 .set_caption(f"<b style='font-size:16px; padding-bottom:10px;'>{caption}</b>")
                 .set_properties(**{'text-align': 'center', 'font-size': '14px'})
                 .set_table_styles([
                     {'selector': 'th, td', 
                      'props': [('border-style', 'solid'), ('border-width', '2px'), ('border-color', 'black')]},
                     {'selector': 'th',
                      'props': [('background-color', '#d9d9d9'), 
                                ('font-weight', 'bold'), 
                                ('text-align', 'center')]},
                     {'selector': 'tr th',
                      'props': [('background-color', '#c0c0c0'), 
                                ('font-weight', 'bold'), 
                                ('text-align', 'center')]},
                     {'selector': 'table',
                      'props': [('border-collapse', 'collapse'), 
                                ('margin', '25px 0'), 
                                ('font-size', '14px')]},
                     {'selector': '.index_name', 
                      'props': [('display', 'none')]}
                 ])
                 .format(None, na_rep="-"))
    return df, styled_df