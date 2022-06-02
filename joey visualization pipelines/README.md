# README
General requirement: You need to manually put 'output_attr_stats.csv' inside this directory. 'output_attr_stats.csv' is too big and exceed github's file limit.
Specific requirements for each notebook are listed at the beginning of itself. Unless mentioned, if missing a package, you can simply 'pip install xxx' to install it.

Generic preparing steps (run in order):
1. identify cra, commissioner, and gov people.ipynb
2. extracting CRA and Commissioner emails in stages.ipynb
3. build adjacency matrix and calculate centrality.ipynb (heavy calculation)

After the steps above, you should be able to run other notebooks:

- centrality & density evaluation over time plot.ipynb: generate 'centrality/network density evalotion over months' plot.
- concentric network with 4 color notation.ipynb: generate '4 stages network plot'.
- interactive plots with altair.ipynb: generate interactive htmls to visualize centralities
- get email domains.ipynb: generates 'top 200 email extensions.csv' and 'Top 50 unique email domains' bar plot.

Each of these notebooks, as the name suggested, is responsible for generating one of the figures/results. Each notebook starts with several blocks to load existing data/calculation. Then several function definitions. Then it follows some scripts that call the defined functions and generate results. Functions are well documented with doc string and can use 'help(func_name)' to retrieve the description. 
