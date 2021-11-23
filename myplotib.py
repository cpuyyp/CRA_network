import pandas as pd
import networkx as nx
import altair as alt
import matplotlib.pyplot as plt


def createZoomableBarChart(df, X, Y, selection, title = '',zoom_encodings = ['x'], top = 100, figsize = (500, 200)):
    '''
    Create Zoomable altair chart from a pandas DataFrame.

    Notice: In consideration of applying the same selection multiple times,
        especially when joint multiple charts together, the selection
        object is not applied to the chart. It should be applied manually
        by adding .add_selection(selection) to the final chart.

    :param df: A pandas DataFrame holding the data.
    :type df: DataFrame
    :param X: Name of the column in df for x-axis.
    :type df: str
    :param Y: Name of the column in df for y-axis.
    :type df: str
    :param selection: A collection of selection objects for filtering data.
    :type df: altair.selection_single
    :param title: Title of the plot. '' by default.
    :type df: str
    :param zoom_encodings: Specifying the zoomable domain. ['x'] by default.
    :type df: list(str)
    :param top: Only show the top n bars. 100 by default.
    :type df: int
    :param figsize: Size of the chart. (500, 200) by default.
    :type df: tuple(int,int)

    :return: A chart with base view in the bottom and zoomed view on the top.
    :rtype: altair.Chart
    '''
    brush = alt.selection(type="interval", encodings=zoom_encodings)

    base = alt.Chart(df, title="Select a range in the base view below").mark_bar().encode(x=alt.X( X+':O', sort='-y'),
        y=Y+':Q'
    ).add_selection(
        brush
    ).transform_filter(
        selection
    ).properties(
        width=figsize[0],
        height=figsize[1]*0.1
    )

    zoomed = alt.Chart(df, title=title).mark_bar().transform_filter(
        selection
    ).transform_filter(
        brush
    ).encode(x=alt.X(X+':O', sort='-y'),
        y=Y+':Q',
        tooltip=[X, Y]
    ).properties(
        width=figsize[0],
        height=figsize[1]*0.9
    )

    if top:
        base.transform_window(
            rank='rank('+Y+')',
            sort=[alt.SortField(Y, order='descending')]
        ).transform_filter(
            (alt.datum.rank < top)
        )
        zoomed.transform_window(
            rank='rank('+Y+')',
            sort=[alt.SortField(Y, order='descending')]
        ).transform_filter(
            (alt.datum.rank < top)
        )

    chart = zoomed & base

    return chart
