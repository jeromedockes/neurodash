# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Encoding with NeuroQuery
# ========================
#
# The model used here is the same as the one deployed on the [neuroquery
# website](https://neuroquery.saclay.inria.fr).

# ## Encode a query into a statistical map of the brain

from neuroquery import fetch_neuroquery_model, NeuroQueryModel
from nilearn.plotting import plot_img, view_img
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown

# %%capture
encoder = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

query = widgets.Text(value="brainstem")
button = widgets.Button(description="Run query")
display(widgets.HBox([query, button]))
main_output = widgets.Output()
display(main_output)

def title_as_link(df):
    return df.apply(lambda x: f"<a href=\"{x['pubmed_url']}\" target=\"_blank\">{x['title']}</a>", axis=1)

def run_and_display_query(_):
    result = encoder(query.value)
    similar_docs = result["similar_documents"].head().copy()
    similar_docs.loc[:, 'title'] = title_as_link(similar_docs)
    
    bmap_out = widgets.Output(layout=widgets.Layout(width='auto', min_width='200px', height='200px', grid_area='brainmap'))
    terms_out = widgets.Output(layout=widgets.Layout(width='auto', height='350px', grid_area='terms'))
    docs_out = widgets.Output(layout=widgets.Layout(width='auto', grid_area='docs'))
    
    bmap_out.append_display_data(HTML(view_img(result["z_map"], threshold=3.1).get_iframe(), 350, 120))
    terms_out.append_display_data(Markdown("## Similar Words"))
    sw = result["similar_words"].head(12)
    terms_out.append_display_data(sw.style.bar(subset=['weight_in_brain_map', 'weight_in_query'], color='lightgreen'))
    docs_out.append_display_data(Markdown("## Similar Documents"))   
    docs_out.append_display_data(similar_docs[['title', 'similarity']].style.hide_index().bar(color='lightgreen'))

    with main_output:
        main_output.clear_output()
        
        display(
            widgets.GridBox(
                children=[bmap_out, terms_out, docs_out],
                layout=widgets.Layout(
                    width='100%',
                    grid_template_rows='auto auto',
                    grid_template_columns='50% 50%',
                    grid_template_areas='''
                    "braimap terms"
                    "docs    docs"
                ''')
            )
        )

button.on_click(run_and_display_query)

run_and_display_query(None)


