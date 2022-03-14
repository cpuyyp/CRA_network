# Execute multiple notebooks sequentially from a new notebook
## !jupyter nbconvert --execute --inplace execute_*.ipynb

!jupyter nbconvert --execute --inplace execute_new_preprocess.ipynb
!jupyter nbconvert --execute --inplace preprocessing_testing.ipynb
!jupyter nbconvert --execute --inplace sent_header.ipynb
!jupyter nbconvert --execute --inplace handle_to_header.ipynb
!jupyter nbconvert --execute --inplace standardize\ names.ipynb
!jupyter nbconvert --execute --inplace add_attributes_to_output.ipynb

# Test this
new_preprocess.ipynb  # not sure)
preprocessing_testing.ipynb  # (not sure)
handle_sent_header.ipynb
handle_to_header.ipynb
standadize\ names.ipynb

# also removes email duplicates
#   (smae From, Send, nb of chars, nb of words
# output: "output_with_attributes_no_duplicates"
# Also outputs: "output_with_stats_columns", which includes
# means and std of attribues (nb of chars/words) per Sender.
add_attributes_to_output.ipynb
