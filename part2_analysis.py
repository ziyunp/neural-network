import numpy as np 
import matplotlib.pyplot as plt 

from part2_claim_classifier import ClaimClassifier

from sklearn.preprocessing import MinMaxScaler, scale

def plot_distribution(attr, label, i):
    zoom_in_percentile_range = (0, 99.9)
    cutoffs_attr = np.percentile(attr, zoom_in_percentile_range)
    non_outliers_mask = (
        np.all(np.array(attr > cutoffs_attr[0]).reshape(len(attr), 1), axis=1) &
        np.all(np.array(attr < cutoffs_attr[1]).reshape(len(attr), 1), axis=1))

    plt.figure(figsize=(6, 8))
    
    plt.subplot(411)
    plt.title("Attr {}".format(i))
    plt.ylabel("Distribution")
    plt.hist(attr, bins=40)
    
    plt.subplot(412)
    plt.ylabel("Label")
    plt.scatter(attr, label)
    
    plt.subplot(413)
    plt.ylabel("Distribution")
    plt.hist(attr[non_outliers_mask], bins=40)
    
    plt.subplot(414)
    plt.xlabel("Attribute Value")
    plt.ylabel("Label")
    # plt.title("After zoom in by 99.9%")
    plt.scatter(attr[non_outliers_mask], label[non_outliers_mask])
    
    plt.show()

def plot_pair_distribution(attr1, attr2, label):
    plt.figure()
    x = [label == 1]
    y = [label == 0]
    plt.scatter(attr1[x], attr2[x], s = 5, marker='o')
    plt.scatter(attr1[y], attr2[y], s = 5, marker='x')
    plt.show()


# drv_age1, vh_age, vh_cyl, vh_din, pol_bonus, vh_sale_begin, vh_sale_end, 
# vh_value, vh_speed, claim_amount, made_claim
dataset = np.genfromtxt('part2_training_data.csv', delimiter=',', skip_header=1)
label = dataset[:, -1]
attr = dataset[:, :-2]
# scaler = MinMaxScaler()
# scaler.fit(dataset[:, :-2])
# attr = scaler.transform(dataset[:, :-2])

# Single Attribute Distribution
# for i in range(len(attr[0, :])):
#     plot_distribution(attr[:, i], label, i)

# Data analysis
# mask = (dataset[:, 2] != 0) & (dataset[:, 8] != 25)
# print("Before: ", len(attr))
# attr = attr[mask]
# print("After: ", len(attr))
for i in range(len(attr[0, :])):
    print("Attr {} -- max : {}    min : {}    mean : {}   1-99% : {}   0.01-99.9% : {}"\
        .format(i, max(attr[:, i]), min(attr[:, i]), np.mean(attr[:, i]), \
        np.percentile(attr[:, i], [0.1, 99]), np.percentile(attr[:, i], [0.01, 99.9])))

# for i in range(len(attr[0, :])):
#     for j in range(i+1, len(attr[0, :])):
#         plot_pair_distribution(attr[:, i], attr[:, j], label)


# for data in dataset:
#     if data[5] < data[6]:
#         print(data)
