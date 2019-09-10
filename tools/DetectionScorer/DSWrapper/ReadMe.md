# DSWrapper

DSWrapper is a Python module that helps creating pipeline based on the Detection Scorer and the DMRender.  

## Requirements

DSWrapper needs Python 3.4 or later (pathlib) and has the following dependencies:
- Jinja2

## Input description

The DSWrapper use two json files to generated the html summary

* Scoring json file:
    - Defines the metadata of each scoring, its name, location and scoring options

* Plot json file:
    - Defines the way to group multiple scoring plot into ones, with description and line options

#### Scoring json file format
```python
{
    "scoring_key": {
        "name": ..., # Name of the scoring (not used in the summary)
        "sub_output_folder": ..., # Name of the folder created for this scoring under the main output folder
        "description": ..., # Scoring description (not used in the summary)
        "options": ... # Any options for the detection scorer not already provided to the DSWrapper, 
                       # like --outMeta, --outSubMeta, --dump, --farStop, --ciLev, --ci, -e, -qm, -t, --plotTitle
    }, 
    {
        ...
    }
}
```

#### Plot json file format
```python
{
    "group_plot_key": {
        "group_plot_name": ..., # Paragraphe name for the plot
        "description": ..., # Text display under the group plot name
        "ss_list": [
            {
                "s_name": ..., # scoring_key defined in the scoring json file
                "s_line_options": {} # Matplotlib line options for this scoring curves
            },
            {
                "s_name": ...,
                "s_line_options": {}
            }
        ]
    }
}
```

## Usage

```bash
python dswrapper.py \
--scoring-dict ./scoring_dicts/my.scorings.json \
--plotgroup-dict ./plot_group_dicts/my.plots.json \
--datasetDir ./datasets/ \
--system  ./system/my_system.csv \
--output ./output/my_prefix_output \
--index indexes/index.csv \
--ref reference/manipulation-image/ref.csv \
--only-group-plots \
--mediscore-path ../../..
```