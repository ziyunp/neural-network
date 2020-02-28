def filter_attributes(X_raw, threshold):
  rm_attr = []
  for feat in range(X_raw.shape[1]):
    count = 0
    for data in X_raw[:,feat]:
        if not data or data != data:
            count += 1
    if count > threshold * X_raw.shape[0]:
        rm_attr.append(feat)
  return rm_attr

def filter_data(X_raw, threshold, removed_att):
  rm_rows = []
  for row in range(len(X_raw)):
    count = 0
    for i in range(X_raw.shape[1]):
        if i not in removed_att:
            data = X_raw[row][i]
            if not data or data != data:
                count += 1
    if count > threshold * X_raw.shape[1]:
        rm_rows.append(row)
  return rm_rows