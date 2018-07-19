import os
from jinja2 import Environment, FileSystemLoader

def create_data_file(df, data_file_path, filename = "data.js"):
    content = "var data = {}\n".format(df.to_json(orient="records"))
    file_path = os.path.join(data_file_path, filename)
    with open(file_path, "w") as f:
        f.write(content)
    return filename

def gen_column_description(df, sortable_set=None, link_formatter_set=None, character_pixel_size = 7):
    """
    Generates the javascript description of the dataframes columns
    :param df: the dataframe
    :param column_template: the column description template
    :sortable: set/list of column that can be sorted
    :link_formatter: set/list of column which contain only links
    """
    column_template = """{{id:"{}", name:"{}", field:"{}", sortable:{}, formatter:{}, minWidth:{}, width:{}}}"""
    link_formatter = """linkFormatter = function ( row, cell, value, columnDef, dataContext ) {return '<a href="' + value.split("@")[1] + '">' + value.split("@")[0] + '</a>';}"""
    
    decription_list = []
    for column, dtype in zip(df.columns, df.dtypes):
        col_id = "column_id_{}".format(column)
        column_serie = df[column]
        if dtype == object:
            if "@" in column_serie[0]:
                column_max_string_len = column_serie.str.split('@').str.get(0).map(len).max()
            else:
                column_max_string_len = df[column].map(len).max()
        else:
            column_max_string_len = df[column].astype(str).str.len().max()

        # We compare with the length of column's name
        column_max_string_len = max(len(column), column_max_string_len)
        
        if sortable_set:
            isSortable = "true" if column in sortable_set else "false"
        else:
            isSortable = "false"
        if link_formatter_set:
            isLink = link_formatter if column in link_formatter_set else "null" 
        else:
            isLink = "null"
        decription_list.append(column_template.format(col_id, column, column, isSortable, isLink, 
                                                      column_max_string_len*(character_pixel_size-1),
                                                      column_max_string_len*character_pixel_size))
    
    return "var columns = [{}];".format((",\n"+17*" ").join(decription_list))

def create_html(wd, df, data_file_path, template_path, sortable_set=None, link_formatter_set=None,
                base_template="base.html", slickGrid_path = "../SlickGrid"):
    
    data_file_name = create_data_file(df, data_file_path)
    column_description_string = gen_column_description(df, 
                                                       sortable_set=sortable_set, 
                                                       link_formatter_set=link_formatter_set)
    file_loader = FileSystemLoader(template_path)
    env = Environment(loader=file_loader)
    template = env.get_template(base_template)
    template_variables = {"page_title": "Generated HTML",
                          "slickgrid_path": slickGrid_path,
                          "full_page_style": True,
                          "add_formatting_function":True,
                          "data_file": data_file_name,
                          "container_id": "#container",
                          "column_description": column_description_string,
                          "add_multi_sorting": sortable_set is not None}    
    html = template.render(template_variables)
    return html