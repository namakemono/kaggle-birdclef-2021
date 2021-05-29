def add_secondlabel(prob_df):
  """
  prob_dfを入力とし，
  most_likely列：397次元のうち確率値最大の鳥を格納
  birds列：secondary_labelsを追加したもので更新
  上記の修正をしたprob_dfを出力とする．
  """
    LABEL = {i:bird for i,bird in zip([i for i in range(397)], list(prob_df.columns))}
    INV_LABEL = {bird:i for i, bird in zip([i for i in range(397)], list(prob_df.columns))}
    
    def extract_most_likely(row):
        row = row[0:397].values
        return LABEL[np.argmax(row)]
    
    def add_second_label(row):
        second_label_empty = (row['secondary_labels']=='[]')
        most_likely_isnot_primary = (row['primary_label']!=row['most_likely'])
        most_likely_prob = row[row['most_likely']]
        if second_label_empty & most_likely_isnot_primary & (most_likely_prob>0.9):
            return row['birds'] + ' ' + row['most_likely']
        else:
            return row['birds']
    
    prob_df['most_likely'] = prob_df.apply(extract_most_likely, axis=1)
    prob_df['birds'] = prob_df.apply(add_second_label, axis=1)
    return prob_df