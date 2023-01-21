import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing


argument = sys.argv
try:
    data = argument[1]
    weights = argument[2]
    impacts = argument[3]
    output = argument[4]
except:
    sys.exit("\n!!!!ERROR!!!\nTHERE IS SOME PROBLEM WITH THE ARGUMENTS!!!!\nTHE COMMAND TO BE USED IS: \npython3 main.py input_FILENAME.csv \"1,2,3,1,1\" \"\"-\",\"+\",\"+\",\"+\",\"+\"\" output_FILENAME.csv\n\n")

try:
    df = pd.read_csv(data, header=0)
except:
    sys.exit(
        "\n!!!ERROR!!!\nFILE \'f{data}\' NOT FOUND\n This tool accpets \'CSV\' files only!!\n")

n = df.shape[0]
c = df.shape[1]

if (c <= 2):
    sys.exit("!!!!ERROR!!!!\n INPUT DATA SHOULD HAVE ATLEAST 3 COLUMNS!")
if (len(weights) < c-1 or len(impacts) < c-1):
    sys.exit(
        "\n!!!!ERROR!!!!\nTHERE IS SOME PROBLEM WITH THE INPUT ARGUMENTS' LENGTH \n")
try:
    weights = [int(i) for i in weights.split(',')]
except:
    sys.exit("\n!!!!ERROR!!!\n RECHECK WEIGHTS ASSIGNED!!!]n")

impacts = impacts.split(',')
for i in impacts:
    if not (i == '+' or i == '-'):
        sys.exit("\n!!!ERROR!!!\n RECHECK IMPACTS ASSIGNED!!!\n")


for z, i in enumerate(df.iloc[0, 1:]):
    if type(i) == str:
        label_encoder = preprocessing.LabelEncoder()
        df.iloc[:, z + 1] = label_encoder.fit_transform(df.iloc[:, z+1])


matrix = np.array(df.drop(['Fund Name'], axis=1))

matrix_sq = np.square(matrix)
sum_col = np.sum(matrix_sq, axis=0)
sqrt_sum_col = np.sqrt(sum_col)
sq_sum_matrix = np.array([sqrt_sum_col]*n)
normalized_matrix = np.divide(matrix, sq_sum_matrix)


weight_matrix = np.array([[int(i) for i in weights]]*n)
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
