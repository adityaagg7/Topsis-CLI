import sys
import pandas as pd
import numpy as np


argument = sys.argv
data = argument[1]
weights = argument[2]
impacts = argument[3].split(",")
output = argument[4]

df = pd.read_csv(data, header=0)
n = df.shape[0]
c = df.shape[1]

matrix = np.array(df.drop(['Fund Name'], axis=1))
matrix_sq = np.square(matrix)
sum_col = np.sum(matrix_sq, axis=0)
sqrt_sum_col = np.sqrt(sum_col)
sq_sum_matrix = np.array([sqrt_sum_col]*n)
normalized_matrix = np.divide(matrix, sq_sum_matrix)


weight_matrix = np.array([[int(i) for i in weights.split(",")]]*n)
weighed_input = np.multiply(normalized_matrix, weight_matrix)


V_Positive = []
V_Negative = []
for i in range(c-1):
    if impacts[i] == '+':
        V_Positive.append(max(weighed_input[:, i]))
        V_Negative.append(min(weighed_input[:, i]))
    else:
        V_Positive.append(min(weighed_input[:, i]))
        V_Negative.append(max(weighed_input[:, i]))
V_Postive_matrix = np.array([V_Positive]*n)
V_Negative_matrix = np.array([V_Negative]*n)

difference_matrix_positive = np.subtract(weighed_input, V_Postive_matrix)
difference_matrix_negative = np.subtract(weighed_input, V_Negative_matrix)
difference_matrix_negative_sq = np.square(difference_matrix_negative)
difference_matrix_positive_sq = np.square(difference_matrix_positive)


S_Positive = np.sqrt(np.sum(difference_matrix_positive_sq, axis=1))
S_Negative = np.sqrt(np.sum(difference_matrix_negative_sq, axis=1))

P = S_Negative/(S_Positive+S_Negative)

P_df = pd.DataFrame(P.transpose())
df["Topsis Score"] = P
df["Rank"] = P_df.rank(ascending=False)
print("\n\nResult Found:\n")
print(df)

print(f"\n \nResult Stored at {output}")
df.to_csv(output, index=False)
